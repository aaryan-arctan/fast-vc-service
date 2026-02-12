# WebSocket Client Examples

Three WebSocket clients for the voice conversion service, covering different use cases.

## Prerequisites

Make sure the VC server is running and accessible. All commands below assume you are in the `fast-vc-service` project root.

```bash
cd fast-vc-service
```

---

## 1. Real-Time Device Streaming (`ws_client_realtime.py`)

Captures audio from a microphone, streams it to the server for voice conversion, and plays the converted audio back through a speaker — all in real time.

### Quick Start

```bash
uv run examples/websocket/ws_client_realtime.py
```

On launch you will be prompted to select an input and output device:

```
Input Devices:
------------------------------------------------------------
  [1] MacBook Pro Microphone  (index=0, 1ch, 48000 Hz)
  [2] External USB Mic        (index=3, 2ch, 44100 Hz)
------------------------------------------------------------
Select input device [1-2]: 1
  -> Selected: MacBook Pro Microphone

Output Devices:
------------------------------------------------------------
  [1] MacBook Pro Speakers    (index=1, 2ch, 48000 Hz)
  [2] External Headphones     (index=4, 2ch, 44100 Hz)
------------------------------------------------------------
Select output device [1-2]: 2
  -> Selected: External Headphones
```

After selection, the client connects to the server and begins streaming. Press **Ctrl+C** to stop.

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `--url` | `ws://localhost:8042/ws` | WebSocket server URL |
| `--api-key` | `test-key` | API key for authentication |
| `--encoding` | `PCM` | Audio encoding: `PCM` or `OPUS` |
| `--samplerate` | `16000` | Mic capture sample rate (Hz). Choices: 8000, 12000, 16000, 24000, 48000 |
| `--samplerate-out` | `22050` | Server output sample rate (Hz). Choices: 8000, 16000, 22050 |
| `--chunk-time` | `20` | Chunk duration in ms (PCM mode) |
| `--bitrate` | `128000` | Opus bitrate in bps |
| `--frame-duration` | `20` | Opus frame duration in ms |

### Examples

```bash
# Default PCM at 16 kHz
uv run examples/websocket/ws_client_realtime.py

# Opus encoding with custom server
uv run examples/websocket/ws_client_realtime.py \
    --url ws://myserver:8042/ws \
    --encoding OPUS \
    --bitrate 64000

# Higher quality output
uv run examples/websocket/ws_client_realtime.py \
    --samplerate 48000 \
    --samplerate-out 22050
```

---

## 2. File-Based Streaming (`ws_client.py`)

Reads WAV files from disk, streams them to the server simulating real-time playback, and saves the converted output to WAV files. Supports multiple files processed in parallel.

### Quick Start

```bash
uv run examples/websocket/ws_client.py
```

Configuration is done by editing the `get_inputs()` function inside the script. Key parameters in the `Inputs` model:

| Parameter | Default | Description |
|---|---|---|
| `url` | `ws://localhost:8042/ws` | WebSocket server URL |
| `api_key` | `test-key` | API key |
| `src_wavs` | `["resources/srcs/female-0-16khz.wav"]` | List of input WAV file paths |
| `output_wav_dir` | `outputs/ws_client` | Directory for converted output files |
| `max_workers` | `2` | Max parallel WebSocket clients |
| `real_time` | `True` | Simulate real-time send pacing |
| `encoding` | `PCM` | `PCM` or `OPUS` |
| `samplerate` | `16000` | Input sample rate sent to server |
| `samplerate_out` | `22050` | Requested output sample rate |
| `save_timeline` | `False` | Save send/receive timeline JSON |

### Example: Process a Directory of WAV Files

Edit `get_inputs()` in the script:

```python
def get_inputs():
    src_wavs = [str(p) for p in Path("my_audio/").glob("*.wav")]
    return Inputs(
        src_wavs=src_wavs,
        max_workers=5,
        encoding=EncodingEnum.PCM,
        samplerate=SampleRateEnum.SR_16000,
        samplerate_out=22050,
    )
```

---

## 3. Simple Protocol Client (`simple_protocol_ws_client.py`)

A minimal client using the simplified WebSocket protocol (`signal: "start"` / `signal: "end"` instead of `type: "config"` / `type: "end"`). Uses CLI arguments via argparse.

### Quick Start

```bash
uv run examples/websocket/simple_protocol_ws_client.py \
    --input resources/srcs/female-0-16khz.wav \
    --output outputs/converted.wav
```

---

## WebSocket Protocol Summary

All clients follow the same general flow:

1. **Connect** to `ws://<host>/ws`
2. **Send config** (JSON) with session ID, sample rate, encoding, etc.
3. **Receive ready** signal from server
4. **Stream audio** as binary WebSocket frames (PCM int16 or Opus packets)
5. **Send end** signal (JSON)
6. **Receive complete** signal with stats from server

Audio format:
- **Input**: mono, int16 PCM bytes (or Opus-encoded packets)
- **Output**: mono, int16 PCM bytes at the requested `sample_rate_out`
