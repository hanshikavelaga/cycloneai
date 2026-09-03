import threading
import time
from datetime import datetime, timedelta
import logging
from backend.app.services.live_sync import execute_sync
from backend.app.services.status_cache import status_manager

logger = logging.getLogger("scheduler")

# Background thread and cancellation event
_scheduler_thread = None
_stop_event = threading.Event()

def _scheduler_loop():
    """
    Asynchronous loop that runs in a background thread, executing
    the synchronization checks every 5 minutes (300 seconds).
    """
    logger.info("Background Live Ingestion Scheduler thread started.")
    status_manager.add_log_message("Background Ingestion Scheduler initialized successfully.")
    
    while not _stop_event.is_set():
        try:
            # Calculate sync timestamps
            last_sync_dt = datetime.utcnow()
            next_sync_dt = last_sync_dt + timedelta(minutes=5)
            
            status_manager.update_sync_times(
                last_sync=last_sync_dt.strftime("%Y-%m-%d %H:%M:%S UTC"),
                next_sync=next_sync_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            )
            
            # Execute Ingestion & Sync
            execute_sync()
            
        except Exception as e:
            logger.error(f"Error inside scheduler execution loop: {e}")
            status_manager.add_log_message(f"Scheduler Loop Error: {e}")
            
        # Sleep in 1-second chunks to check the stop event responsively
        for _ in range(300):
            if _stop_event.is_set():
                break
            time.sleep(1)
            
    logger.info("Background Live Ingestion Scheduler thread stopped.")
    status_manager.add_log_message("Background Ingestion Scheduler stopped.")

def start_scheduler():
    """
    Spawns the background scheduler loop.
    """
    global _scheduler_thread, _stop_event
    with threading.Lock():
        if _scheduler_thread is not None and _scheduler_thread.is_alive():
            logger.warning("Scheduler is already running.")
            return
            
        _stop_event.clear()
        _scheduler_thread = threading.Thread(target=_scheduler_loop, daemon=True)
        _scheduler_thread.start()
        logger.info("Started background live sync task scheduler.")

def stop_scheduler():
    """
    Stops the background scheduler loop.
    """
    global _scheduler_thread, _stop_event
    _stop_event.set()
    if _scheduler_thread is not None:
        _scheduler_thread.join(timeout=5)
        logger.info("Stopped background live sync task scheduler.")
