from contextlib import asynccontextmanager

from fastapi import FastAPI
import torch
import uvicorn
import os
from loguru import logger
from pydantic import BaseModel
import traceback
import multiprocessing

from fast_vc_service.config import Config
from fast_vc_service.routers import base_router, websocket_router, tools_router, livekit_router, set_realtime_vc
from fast_vc_service.logging_config import LoggingSetup
from fast_vc_service.realtime_vc import RealtimeVoiceConversion
from fast_vc_service.tools.session_data_manager import SessionDataManager

def _get_work_id() -> str:
    """获取当前worker的index，从1开始的计数
    
    在多卡部署多实例的场景下，会使用该 wid
    """
    process_name = multiprocessing.current_process().name
    logger.info(f"process_name: {process_name}")
    if "SpawnProcess" in process_name:
        wid = int(process_name.split('-')[-1])
        return wid
    
    # pid 兜底
    wid = os.getpid()
    logger.info(f"faild to get wid, use pid instead. process_name:{process_name}, pid: {wid}")
    return str(wid)
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup/shutdown lifecycle."""
    # --- startup (nothing extra needed, create_app already initialises) ---
    yield
    # --- shutdown ---
    logger.info("Shutdown: cleaning up resources...")

    # 1. Release CUDA models held by the Singleton
    try:
        realtime_vc = getattr(app.state, "realtime_vc", None)
        if realtime_vc is not None:
            # Drop model references so CUDA memory can be freed
            if hasattr(realtime_vc, "models") and isinstance(realtime_vc.models, dict):
                realtime_vc.models.clear()
            torch.cuda.empty_cache()
            logger.info("Shutdown: CUDA cache cleared and model references released")
    except Exception:
        logger.warning(f"Shutdown: error releasing CUDA resources: {traceback.format_exc()}")

    # 2. Drain and remove loguru sinks so their non-daemon enqueue threads exit
    try:
        logger.info("Shutdown: removing loguru sinks...")
        logger.remove()  # sends sentinel to every enqueue thread
    except Exception:
        pass  # loguru may already be partially torn down


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    # Initialize logging
    cfg = Config().get_config()
    worker_id = _get_work_id()
    LoggingSetup.setup(cfg.app.log_dir, worker_id=worker_id)
    logger.info("-" * 21 + "initilizing service" + "-" * 21)
    
    # Create FastAPI application
    app = FastAPI(
        title="Realtime Voice Conversion Service",
        description="Voice Conversion Service API",
        lifespan=lifespan,
    )
    
    # Initialize realtime voice conversion service
    app.state.cfg = cfg
    
    logger.info("initializing SessionDataManager...")
    app.state.session_data_manager = SessionDataManager(
        search_dir=cfg.realtime_vc.save_dir
    )
    
    logger.info("initializing class: RealtimeVoiceConversion ...")
    try:
        cfg.realtime_vc.device = cfg.realtime_vc.device[ (int(worker_id)-1) % len(cfg.realtime_vc.device) ]
        
        app.state.realtime_vc = RealtimeVoiceConversion(
            cfg.realtime_vc,
            cfg.models,
        )
    except Exception as e:
        logger.error(f"faild to initialize RealtimeVoiceConversion: {traceback.format_exc()}")
        raise
    
    # Inject realtime_vc into LiveKit module so it can be accessed without Request
    set_realtime_vc(app.state.realtime_vc)
    
    # Include routers
    logger.info("registering routers...")
    app.include_router(base_router)
    app.include_router(websocket_router)
    app.include_router(tools_router)
    app.include_router(livekit_router)
    
    logger.info("-" * 21 + "service initialized" + "-" * 21)
    return app

def main() -> None:
    """Main function to run the FastAPI application."""
    app_config = Config().get_config().app
    logger.info(f"Starting fast vc service on {app_config.host}:{app_config.port}")
    logger.info(f"Number of workers: {app_config.workers}")
    
    uvicorn.run(
        "fast_vc_service.app:create_app",
        host=app_config.host,
        port=app_config.port,
        workers=app_config.workers,
        factory=True,
        loop="uvloop",  # Use uvloop for better async I/O performance
        log_config=None,  # Forbid default logging config
        ws_ping_interval=600,
        ws_ping_timeout=600,
    )
    
if __name__ == "__main__":
    main()