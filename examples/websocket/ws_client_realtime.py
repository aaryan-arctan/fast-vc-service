"""
Real-time voice conversion client using WebSocket.

Captures audio from a selected input device (microphone), streams it to
the voice conversion server over WebSocket, and plays the converted audio
back through a selected output device (speaker).

Usage:
    cd fast-vc-service
    uv run examples/websocket/ws_client_realtime.py

    # With custom server URL and Opus encoding:
    uv run examples/websocket/ws_client_realtime.py --url ws://myserver:18822/ws --encoding OPUS
"""

import asyncio
import websockets
import sounddevice as sd
import soundfile as sf
import numpy as np
import json
import queue
import uuid
import sys
import argparse
from datetime import datetime
from pathlib import Path
from loguru import logger

# Optional Opus support
try:
    import opuslib

    HAS_OPUS = True
except ImportError:
    HAS_OPUS = False


# ---------------------------------------------------------------------------
# Device selection helpers
# ---------------------------------------------------------------------------


def list_devices():
    """Query and categorise audio devices into input and output lists.

    Returns:
        (input_devices, output_devices) where each is a list of
        (display_number, device_index, device_info_dict).
    """
    devices = sd.query_devices()
    input_devices = []
    output_devices = []

    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            input_devices.append((len(input_devices) + 1, idx, dev))
        if dev["max_output_channels"] > 0:
            output_devices.append((len(output_devices) + 1, idx, dev))

    return input_devices, output_devices


def print_device_list(devices, kind: str):
    """Pretty-print a list of audio devices."""
    print(f"\n{kind} Devices:")
    print("-" * 60)
    for display_num, dev_idx, dev in devices:
        sr = int(dev["default_samplerate"])
        ch = (
            dev["max_input_channels"] if kind == "Input" else dev["max_output_channels"]
        )
        print(f"  [{display_num}] {dev['name']}  (index={dev_idx}, {ch}ch, {sr} Hz)")
    print("-" * 60)


def select_device(devices, kind: str) -> int:
    """Prompt the user to select a device by display number.

    Returns the actual sounddevice device index.
    """
    while True:
        try:
            choice = int(input(f"Select {kind.lower()} device [1-{len(devices)}]: "))
            if 1 <= choice <= len(devices):
                _, dev_idx, dev = devices[choice - 1]
                print(f"  -> Selected: {dev['name']}")
                return dev_idx
            print(f"  Please enter a number between 1 and {len(devices)}")
        except ValueError:
            print("  Invalid input. Please enter a number.")
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(0)


def list_and_select_devices():
    """Interactive device selection. Returns (input_device_idx, output_device_idx)."""
    input_devices, output_devices = list_devices()

    if not input_devices:
        logger.error("No input (microphone) devices found!")
        sys.exit(1)
    if not output_devices:
        logger.error("No output (speaker) devices found!")
        sys.exit(1)

    print_device_list(input_devices, "Input")
    input_dev_idx = select_device(input_devices, "Input")

    print_device_list(output_devices, "Output")
    output_dev_idx = select_device(output_devices, "Output")

    return input_dev_idx, output_dev_idx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def generate_session_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique = uuid.uuid4().hex[:6]
    return f"RT{timestamp}-{unique}"


def calc_chunk_frame(
    sample_rate: int, encoding: str, chunk_time_ms: int, frame_duration_ms: int
) -> int:
    """Number of samples per chunk."""
    if encoding == "OPUS":
        return int(frame_duration_ms / 1000 * sample_rate)
    else:  # PCM
        return int(chunk_time_ms / 1000 * sample_rate)


# ---------------------------------------------------------------------------
# Core real-time session
# ---------------------------------------------------------------------------


async def run_realtime_session(
    # server
    url: str = "ws://localhost:18822/ws",
    api_key: str = "test-key",
    # devices (sounddevice indices)
    input_device: int = 0,
    output_device: int = 1,
    # encoding
    encoding: str = "PCM",
    samplerate: int = 16000,
    samplerate_out: int = 22050,
    # PCM
    chunk_time_ms: int = 20,
    # Opus
    bitrate: int = 128_000,
    frame_duration_ms: int = 20,
):
    """
    Run a real-time voice conversion session.

    Captures from *input_device*, sends to the WS server, receives converted
    audio, and plays it on *output_device*.  Runs until the user presses
    Ctrl+C.
    """

    session_id = generate_session_id()
    logger.info(f"Session ID: {session_id}")

    # Validate Opus availability
    if encoding == "OPUS" and not HAS_OPUS:
        logger.error(
            "Opus encoding requested but opuslib is not installed. "
            "Install it with: pip install opuslib"
        )
        return

    # Calculate chunk frame size
    chunk_frame = calc_chunk_frame(
        samplerate, encoding, chunk_time_ms, frame_duration_ms
    )
    logger.info(
        f"Chunk frame size: {chunk_frame} samples "
        f"({chunk_frame / samplerate * 1000:.1f} ms)"
    )

    # Opus encoder (if needed)
    opus_encoder = None
    if encoding == "OPUS":
        opus_encoder = opuslib.Encoder(samplerate, 1, opuslib.APPLICATION_AUDIO)
        opus_encoder.bitrate = bitrate
        logger.info(
            f"Opus encoder: bitrate={bitrate}, frame_duration={frame_duration_ms}ms"
        )

    # Thread-safe queues for bridging sounddevice callbacks <-> asyncio
    mic_queue: queue.Queue = queue.Queue()
    playback_queue: queue.Queue = queue.Queue()

    # Flag used to signal shutdown
    running = True

    # Collect received audio for saving
    received_audio_chunks: list = []

    # Persistent buffer for playback (keeps leftover bytes between callbacks)
    playback_buffer = bytearray()

    # --- sounddevice callbacks (run on PortAudio thread) ---

    def mic_callback(indata, frames, time_info, status):
        if status:
            logger.warning(f"Mic input status: {status}")
        # indata is a numpy array of shape (frames, 1), dtype int16
        mic_queue.put(bytes(indata))

    def playback_callback(outdata, frames, time_info, status):
        nonlocal playback_buffer
        if status and (playback_queue.qsize() > 0 or len(playback_buffer) > 0):
            logger.warning(f"Playback output status: {status}")
        
        bytes_needed = frames * 2  # int16 = 2 bytes per sample
        
        # Pull chunks from queue into buffer until we have enough
        while len(playback_buffer) < bytes_needed:
            try:
                chunk = playback_queue.get_nowait()
                playback_buffer.extend(chunk)
            except queue.Empty:
                break
        
        # Extract exactly what we need
        if len(playback_buffer) >= bytes_needed:
            data = bytes(playback_buffer[:bytes_needed])
            del playback_buffer[:bytes_needed]  # Remove used bytes, keep the rest
        else:
            # Not enough data - use what we have and pad with silence
            data = bytes(playback_buffer) + b"\x00" * (bytes_needed - len(playback_buffer))
            playback_buffer.clear()
        
        outdata[:] = np.frombuffer(data, dtype=np.int16).reshape(-1, 1)

    # --- Open audio streams ---

    input_stream = sd.InputStream(
        device=input_device,
        samplerate=samplerate,
        channels=1,
        dtype="int16",
        blocksize=chunk_frame,
        callback=mic_callback,
    )

    output_stream = sd.OutputStream(
        device=output_device,
        samplerate=samplerate_out,
        channels=1,
        dtype="int16",
        blocksize=1024,
        callback=playback_callback,
    )

    # --- WebSocket connection ---

    try:
        async with websockets.connect(
            url,
            ping_interval=600,
            ping_timeout=600,
            close_timeout=600,
            max_size=None,
        ) as websocket:
            # 1. Send config
            config = {
                "type": "config",
                "session_id": session_id,
                "api_key": api_key,
                "sample_rate": samplerate,
                "sample_rate_out": samplerate_out,
                "bit_depth": 16,
                "channels": 1,
                "encoding": encoding,
            }
            if encoding == "OPUS":
                config["opus_frame_duration"] = frame_duration_ms

            await websocket.send(json.dumps(config))
            logger.info(f"Sent config: {json.dumps(config)}")

            # 2. Wait for ready
            response = await websocket.recv()
            response_data = json.loads(response)
            if response_data.get("type") != "ready":
                logger.error(f"Unexpected server response: {response_data}")
                return
            logger.info(f"Server ready: {response_data.get('message')}")

            # 3. Start audio streams
            input_stream.start()
            output_stream.start()
            logger.info("Audio streams started. Streaming... Press Ctrl+C to stop.")

            # 4. Async tasks

            async def send_loop():
                """Read mic chunks from queue and send over websocket."""
                nonlocal running
                loop = asyncio.get_event_loop()
                while running:
                    try:
                        # Block in executor so we don't block the event loop
                        chunk_bytes = await loop.run_in_executor(
                            None, mic_queue.get, True, 0.1
                        )
                    except queue.Empty:
                        continue

                    if not running:
                        break

                    try:
                        if encoding == "OPUS":
                            chunk_bytes = opus_encoder.encode(chunk_bytes, chunk_frame)
                        await websocket.send(chunk_bytes)
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("WebSocket closed during send")
                        break
                    except Exception as e:
                        logger.error(f"Send error: {e}")
                        break

            async def receive_and_play():
                """Receive converted audio from websocket and queue for playback."""
                nonlocal running
                while running:
                    try:
                        response = await websocket.recv()
                    except websockets.exceptions.ConnectionClosed:
                        logger.info("WebSocket connection closed")
                        break
                    except asyncio.CancelledError:
                        break

                    if isinstance(response, bytes):
                        # Raw int16 PCM from server -> push to playback queue
                        playback_queue.put(response)
                        received_audio_chunks.append(response)
                    else:
                        try:
                            data = json.loads(response)
                            msg_type = data.get("type")
                            if msg_type == "complete":
                                logger.info(f"Server complete: {data}")
                                break
                            elif msg_type == "error":
                                logger.error(f"Server error: {data}")
                                break
                            else:
                                logger.info(f"Server message: {data}")
                        except json.JSONDecodeError:
                            logger.warning(f"Non-JSON text from server: {response}")

            send_task = asyncio.create_task(send_loop())
            recv_task = asyncio.create_task(receive_and_play())

            # 5. Wait for Ctrl+C
            stop_event = asyncio.Event()

            loop = asyncio.get_event_loop()
            try:
                import signal as signal_mod

                loop.add_signal_handler(signal_mod.SIGINT, stop_event.set)
                use_signal_handler = True
            except NotImplementedError:
                # Windows doesn't support add_signal_handler in some cases
                use_signal_handler = False

            if use_signal_handler:
                await stop_event.wait()
            else:
                # Fallback: poll so KeyboardInterrupt can be raised
                try:
                    while running:
                        await asyncio.sleep(0.5)
                except KeyboardInterrupt:
                    pass

            # 6. Shutdown
            logger.info("Stopping...")
            running = False

            # Stop mic capture first
            input_stream.stop()
            input_stream.close()
            logger.info("Input stream closed")

            # Send end signal to server
            try:
                await websocket.send(json.dumps({"type": "end"}))
                logger.info("Sent end signal to server")
            except websockets.exceptions.ConnectionClosed:
                pass

            # Wait for server to send complete (with timeout)
            try:
                await asyncio.wait_for(recv_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for server completion")
            except asyncio.CancelledError:
                pass

            # Cancel remaining tasks
            send_task.cancel()
            recv_task.cancel()
            for t in [send_task, recv_task]:
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

            # Let remaining playback drain briefly
            await asyncio.sleep(0.5)

            output_stream.stop()
            output_stream.close()
            logger.info("Output stream closed")

    except ConnectionRefusedError:
        logger.error(f"Could not connect to {url}. Is the server running?")
    except Exception as e:
        logger.error(f"Session error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Ensure streams are closed even on unexpected errors
        try:
            if input_stream.active:
                input_stream.stop()
                input_stream.close()
        except Exception:
            pass
        try:
            if output_stream.active:
                output_stream.stop()
                output_stream.close()
        except Exception:
            pass

    # Save received audio to file
    if received_audio_chunks:
        output_dir = Path("outputs/realtime")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{session_id}_received_{samplerate_out}hz.wav"
        
        all_bytes = b"".join(received_audio_chunks)
        audio_data = np.frombuffer(all_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        sf.write(output_path, audio_data, samplerate_out)
        logger.info(f"Saved received audio ({len(audio_data)} samples @ {samplerate_out}Hz): {output_path}")

    logger.info(f"Session {session_id} ended.")


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time voice conversion WebSocket client"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="ws://localhost:18822/ws",
        help="WebSocket server URL (default: ws://localhost:18822/ws)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="test-key",
        help="API key for authentication (default: test-key)",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="PCM",
        choices=["PCM", "OPUS"],
        help="Audio encoding format (default: PCM)",
    )
    parser.add_argument(
        "--samplerate",
        type=int,
        default=16000,
        choices=[8000, 12000, 16000, 24000, 48000],
        help="Sample rate for mic capture / server input (default: 16000)",
    )
    parser.add_argument(
        "--samplerate-out",
        type=int,
        default=22050,
        choices=[8000, 16000, 22050, 48000],
        help="Output sample rate from server (default: 22050)",
    )
    parser.add_argument(
        "--chunk-time",
        type=int,
        default=20,
        help="Chunk time in ms for PCM encoding (default: 20)",
    )
    parser.add_argument(
        "--bitrate",
        type=int,
        default=128000,
        help="Bitrate for OPUS encoding (default: 128000)",
    )
    parser.add_argument(
        "--frame-duration",
        type=int,
        default=20,
        help="Frame duration in ms for OPUS encoding (default: 20)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    # Validate Opus availability early
    if args.encoding == "OPUS" and not HAS_OPUS:
        logger.error(
            "Opus encoding requested but opuslib is not installed. "
            "Install it with: pip install opuslib"
        )
        sys.exit(1)

    # Interactive device selection
    input_dev_idx, output_dev_idx = list_and_select_devices()

    print(f"\nStarting real-time voice conversion session...")
    print(f"  Server:      {args.url}")
    print(f"  Encoding:    {args.encoding}")
    print(f"  Sample rate: {args.samplerate} Hz (in) / {args.samplerate_out} Hz (out)")
    print(f"  Input dev:   index {input_dev_idx}")
    print(f"  Output dev:  index {output_dev_idx}")
    print()

    asyncio.run(
        run_realtime_session(
            url=args.url,
            api_key=args.api_key,
            input_device=input_dev_idx,
            output_device=output_dev_idx,
            encoding=args.encoding,
            samplerate=args.samplerate,
            samplerate_out=args.samplerate_out,
            chunk_time_ms=args.chunk_time,
            bitrate=args.bitrate,
            frame_duration_ms=args.frame_duration,
        )
    )
