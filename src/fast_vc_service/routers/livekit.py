"""LiveKit router for real-time voice conversion via LiveKit audio transport.

Provides a POST /call endpoint matching the arctan-inference-server contract,
so the LiveKit server / media gateway requires no changes.

Usage:
    POST /call
    {
        "livekit_ws_url": "wss://...",
        "livekit_token": "...",
        "livekit_room": "room-name"
    }
"""

import asyncio
import uuid

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from loguru import logger

from livekit import rtc
from livekit.plugins import noise_cancellation

from fast_vc_service.livekit_audio_processor import LiveKitAudioProcessor

# Stored at module level so the app.state reference can be set once at startup
_realtime_vc = None


def set_realtime_vc(realtime_vc):
    """Called once from create_app() to inject the RealtimeVoiceConversion instance."""
    global _realtime_vc
    _realtime_vc = realtime_vc


livekit_router = APIRouter()


class CallRequest(BaseModel):
    livekit_ws_url: str
    livekit_token: str
    livekit_room: str


@livekit_router.post("/call")
async def join_room(request: CallRequest = Body(...)):
    """Join LiveKit room with provided token and room."""
    livekit_ws_url = request.livekit_ws_url
    livekit_token = request.livekit_token
    livekit_room = request.livekit_room

    # Validate required parameters
    if not livekit_ws_url:
        raise HTTPException(status_code=400, detail="livekit_ws_url is required")
    if not livekit_token:
        raise HTTPException(status_code=400, detail="livekit_token is required")
    if not livekit_room:
        raise HTTPException(status_code=400, detail="livekit_room is required")

    logger.info(f"Connecting to LiveKit WS: {livekit_ws_url}")
    logger.info(f"Joining room: {livekit_room}")
    logger.info(f"Using provided token: {livekit_token[:20]}...")  # Log first 20 chars for security

    room = rtc.Room()

    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        logger.info(
            "participant connected: %s %s", participant.sid, participant.identity
        )

    # Track active processors to prevent duplicates
    active_processors = {}

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        logger.info("track subscribed: %s", publication.sid)
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            # Create unique key for this participant's audio track
            processor_key = f"{participant.identity}_{publication.sid}"

            # Check if processor already exists for this track
            if processor_key in active_processors:
                logger.warning(
                    f"Audio processor already exists for {processor_key}, skipping duplicate creation"
                )
                return

            # Clean up any existing processor for this participant (different track)
            participant_prefix = f"{participant.identity}_"
            for key in list(active_processors.keys()):
                if key.startswith(participant_prefix) and key != processor_key:
                    logger.info(f"Cleaning up old processor for participant {participant.identity}")
                    old = active_processors.pop(key)
                    if hasattr(old, '_stream_active'):
                        old._stream_active = False

            # Create audio stream with BVC noise cancellation on inbound audio
            audio_stream = rtc.AudioStream.from_track(
                track=track,
                noise_cancellation=noise_cancellation.BVC(),
            )
            logger.info(f"AudioStream created with BVC noise cancellation for {processor_key}")

            # Create session and processor
            per_session_id = f"lk-{uuid.uuid4().hex[:12]}_{participant.identity}"
            session = _realtime_vc.create_session(session_id=per_session_id)

            processor = LiveKitAudioProcessor(
                realtime_vc=_realtime_vc,
                session=session,
            )

            # Store processor reference
            active_processors[processor_key] = processor

            async def cleanup_processor():
                try:
                    await processor.process_stream(audio_stream, room, participant.identity)
                finally:
                    # Clean up when processing is done
                    if processor_key in active_processors:
                        del active_processors[processor_key]
                        logger.info(f"Cleaned up processor for {processor_key}")

            asyncio.create_task(cleanup_processor())
            logger.info(f"Created new audio processor for {processor_key}")

    await room.connect(livekit_ws_url, livekit_token)
    logger.info(f"Connected to room: {room.name}")

    for identity, participant in room.remote_participants.items():
        logger.info(f"identity: {identity}")
        logger.info(f"participant sid: {participant.sid}, identity: {participant.identity}")
        logger.info(f"participant track publications: {participant.track_publications}")
        for tid, publication in participant.track_publications.items():
            logger.info(
                f"\ttrack id: {tid}, kind: {publication.kind}, "
                f"name: {publication.name}, source: {publication.source}"
            )
