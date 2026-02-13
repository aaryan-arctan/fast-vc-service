"""LiveKit audio processor for real-time voice conversion.

Bridges LiveKit's audio transport (48kHz int16, 20ms frames) with the
fast-vc-service core pipeline (RealtimeVoiceConversion.chunk_vc()).

Data flow:
    LiveKit AudioStream (48kHz, 20ms frames, BVC noise-cancelled)
    → Ring buffer accumulation (N frames = block_time)
    → Resample 48kHz → 16kHz (SAMPLERATE_IN)
    → chunk_vc() (core VC pipeline)
    → session.out_data (22050Hz or ws_sr_out float32)
    → Resample to 48kHz
    → Split into 20ms frames → rtc.AudioSource.capture_frame()
"""

import asyncio
import time
import uuid
import numpy as np
import torch
import torchaudio.functional as AF
from loguru import logger

from livekit import rtc

from fast_vc_service.realtime_vc import RealtimeVoiceConversion
from fast_vc_service.session import Session, EventType


# LiveKit constants
LIVEKIT_SAMPLE_RATE = 48000
LIVEKIT_CHANNELS = 1
LIVEKIT_FRAME_DURATION_MS = 20
LIVEKIT_FRAME_SAMPLES = LIVEKIT_SAMPLE_RATE * LIVEKIT_FRAME_DURATION_MS // 1000  # 960


class LiveKitAudioProcessor:
    """Processes audio from a LiveKit AudioStream through the VC pipeline
    and publishes converted audio back to the LiveKit room.
    
    Each instance handles one participant's audio track.
    """
    
    def __init__(
        self,
        realtime_vc: RealtimeVoiceConversion,
        session: Session,
    ):
        """Initialize the LiveKit audio processor.
        
        Args:
            realtime_vc: The singleton RealtimeVoiceConversion instance.
            session: A per-connection Session for VC state.
        """
        self.realtime_vc = realtime_vc
        self.session = session
        self.cfg = realtime_vc.cfg
        
        self._stream_active = False
        
        # Input buffering at 48kHz
        # block_time (default 0.5s) → number of 48kHz samples per block
        self._block_samples_48k = int(self.cfg.block_time * LIVEKIT_SAMPLE_RATE)
        self._ring_buffer = np.zeros(self._block_samples_48k * 2, dtype=np.float32)
        self._ring_fill = 0
        
        # Output sample rate: session may request a specific ws_sr_out,
        # but for LiveKit we always output at 48kHz
        self._output_sr = session.ws_sr_out  # The SR that chunk_vc writes to session.out_data
        
        logger.info(
            f"{session.session_id} | LiveKitAudioProcessor initialized: "
            f"block_time={self.cfg.block_time}s, "
            f"block_48k={self._block_samples_48k}, "
            f"vc_in_sr={self.cfg.SAMPLERATE_IN}, "
            f"vc_out_sr={self._output_sr}"
        )
    
    async def process_stream(
        self,
        audio_stream: rtc.AudioStream,
        room: rtc.Room,
        participant_identity: str,
    ):
        """Main processing loop: receive → buffer → VC → publish.
        
        Args:
            audio_stream: LiveKit AudioStream from the subscribed track.
            room: The LiveKit Room to publish converted audio into.
            participant_identity: Identity string of the source participant.
        """
        if self._stream_active:
            logger.warning(
                f"{self.session.session_id} | process_stream already active, skipping"
            )
            return
        
        self._stream_active = True
        sid = self.session.session_id
        logger.info(f"{sid} | Starting LiveKit audio processing for {participant_identity}")
        
        # Create output audio source and track
        source = rtc.AudioSource(LIVEKIT_SAMPLE_RATE, LIVEKIT_CHANNELS)
        track = rtc.LocalAudioTrack.create_audio_track(
            f"vc-{participant_identity}", source
        )
        
        # Publish converted audio track
        try:
            await room.local_participant.publish_track(track)
            logger.info(f"{sid} | Published VC track: {track.name}")
        except Exception as e:
            logger.error(f"{sid} | Failed to publish track: {e}")
            self._stream_active = False
            return
        
        # Send warmup silence to prime WebRTC buffering
        await self._send_silence(source, frames=5)
        
        try:
            async for event in audio_stream:
                frame = event.frame
                # Convert LiveKit int16 PCM to float32 [-1, 1]
                samples_int16 = np.frombuffer(frame.data, dtype=np.int16)
                samples_float = samples_int16.astype(np.float32) / 32768.0
                
                # Accumulate into ring buffer
                n = len(samples_float)
                space = len(self._ring_buffer) - self._ring_fill
                if n > space:
                    # Should not happen normally, but handle overflow
                    logger.warning(
                        f"{sid} | Ring buffer overflow: fill={self._ring_fill}, "
                        f"incoming={n}, capacity={len(self._ring_buffer)}"
                    )
                    # Keep only the most recent data
                    self._ring_buffer[:self._ring_fill] = self._ring_buffer[:self._ring_fill]
                    drop = n - space
                    self._ring_buffer[:self._ring_fill - drop] = \
                        self._ring_buffer[drop:self._ring_fill]
                    self._ring_fill -= drop
                
                self._ring_buffer[self._ring_fill:self._ring_fill + n] = samples_float
                self._ring_fill += n
                
                # Process complete blocks
                while self._ring_fill >= self._block_samples_48k:
                    block_48k = self._ring_buffer[:self._block_samples_48k].copy()
                    
                    # Shift remaining data
                    remaining = self._ring_fill - self._block_samples_48k
                    if remaining > 0:
                        self._ring_buffer[:remaining] = \
                            self._ring_buffer[self._block_samples_48k:self._ring_fill]
                    self._ring_fill = remaining
                    
                    # Resample 48kHz → 16kHz (SAMPLERATE_IN)
                    block_16k = self._resample(
                        block_48k, LIVEKIT_SAMPLE_RATE, self.cfg.SAMPLERATE_IN
                    )
                    
                    # Run VC pipeline
                    t0 = time.perf_counter()
                    time_msg = self.realtime_vc.chunk_vc(block_16k, self.session)
                    vc_time = time.perf_counter() - t0
                    
                    if vc_time * 1000 > self.cfg.vc_slow_threshold:
                        logger.warning(f"{sid} | {time_msg} | [VC_SLOW]")
                    else:
                        logger.info(f"{sid} | {time_msg}")
                    
                    # Get output from session
                    out_data = self.session.out_data
                    if out_data is None or len(out_data) == 0:
                        logger.warning(f"{sid} | No output data from chunk_vc")
                        continue
                    
                    # Record recv event
                    output_duration_ms = len(out_data) / self._output_sr * 1000
                    self.session.record_event(EventType.RECV, output_duration_ms)
                    
                    # Resample output → 48kHz
                    out_48k = self._resample(
                        out_data, self._output_sr, LIVEKIT_SAMPLE_RATE
                    )
                    
                    # Send as 20ms LiveKit frames
                    await self._send_frames(source, out_48k)
        
        except asyncio.CancelledError:
            logger.info(f"{sid} | Audio processing cancelled for {participant_identity}")
        except Exception as e:
            logger.error(f"{sid} | Error in audio processing loop: {e}", exc_info=True)
        finally:
            # Flush remaining buffer
            await self._flush_remaining(source)
            
            self._stream_active = False
            logger.info(f"{sid} | LiveKit audio processing ended for {participant_identity}")
            
            # Save and cleanup session
            asyncio.create_task(self.session.async_save_and_cleanup())
            logger.info(f"{sid} | Session save and cleanup task created")
    
    def _resample(self, audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
        """Resample audio using torchaudio (GPU-accelerated when available).
        
        Args:
            audio: Input float32 numpy array, 1D.
            from_sr: Source sample rate.
            to_sr: Target sample rate.
            
        Returns:
            Resampled float32 numpy array.
        """
        if from_sr == to_sr:
            return audio
        
        tensor = torch.from_numpy(audio).float()
        resampled = AF.resample(tensor, from_sr, to_sr)
        return resampled.numpy()
    
    async def _send_frames(self, source: rtc.AudioSource, audio_48k: np.ndarray):
        """Split audio into 20ms frames and send via LiveKit AudioSource.
        
        Args:
            source: LiveKit AudioSource to capture frames into.
            audio_48k: Float32 audio at 48kHz.
        """
        # Convert to int16 PCM
        int16_data = (np.clip(audio_48k, -1.0, 1.0) * 32767).astype(np.int16)
        
        total_samples = len(int16_data)
        offset = 0
        
        while offset < total_samples:
            end = min(offset + LIVEKIT_FRAME_SAMPLES, total_samples)
            frame_data = int16_data[offset:end]
            
            # Pad last frame if needed
            if len(frame_data) < LIVEKIT_FRAME_SAMPLES:
                frame_data = np.pad(
                    frame_data,
                    (0, LIVEKIT_FRAME_SAMPLES - len(frame_data)),
                    mode='constant'
                )
            
            frame = rtc.AudioFrame(
                data=frame_data.tobytes(),
                sample_rate=LIVEKIT_SAMPLE_RATE,
                num_channels=LIVEKIT_CHANNELS,
                samples_per_channel=LIVEKIT_FRAME_SAMPLES,
            )
            await source.capture_frame(frame)
            offset = end
    
    async def _send_silence(self, source: rtc.AudioSource, frames: int = 5):
        """Send silent frames to prime WebRTC buffering.
        
        Args:
            source: LiveKit AudioSource.
            frames: Number of 20ms silent frames to send.
        """
        silence = np.zeros(LIVEKIT_FRAME_SAMPLES, dtype=np.int16)
        for _ in range(frames):
            frame = rtc.AudioFrame(
                data=silence.tobytes(),
                sample_rate=LIVEKIT_SAMPLE_RATE,
                num_channels=LIVEKIT_CHANNELS,
                samples_per_channel=LIVEKIT_FRAME_SAMPLES,
            )
            await source.capture_frame(frame)
    
    async def _flush_remaining(self, source: rtc.AudioSource):
        """Process and send any remaining audio in the ring buffer."""
        sid = self.session.session_id
        
        if self._ring_fill <= 0:
            return
        
        logger.info(
            f"{sid} | Flushing remaining {self._ring_fill} samples "
            f"({self._ring_fill / LIVEKIT_SAMPLE_RATE * 1000:.1f}ms)"
        )
        
        try:
            # Pad to full block size
            block_48k = np.zeros(self._block_samples_48k, dtype=np.float32)
            block_48k[:self._ring_fill] = self._ring_buffer[:self._ring_fill]
            self._ring_fill = 0
            
            # Resample and process
            block_16k = self._resample(
                block_48k, LIVEKIT_SAMPLE_RATE, self.cfg.SAMPLERATE_IN
            )
            self.realtime_vc.chunk_vc(block_16k, self.session)
            
            out_data = self.session.out_data
            if out_data is not None and len(out_data) > 0:
                out_48k = self._resample(
                    out_data, self._output_sr, LIVEKIT_SAMPLE_RATE
                )
                await self._send_frames(source, out_48k)
        except Exception as e:
            logger.error(f"{sid} | Error flushing remaining audio: {e}", exc_info=True)
