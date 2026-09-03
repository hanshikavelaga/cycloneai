import threading
from datetime import datetime

class SyncStatusManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._status = {
            "gdacs_status": "DISCONNECTED",
            "gdacs_last_success": None,
            "gdacs_last_error": None,
            "gdacs_records_received": 0,
            "gdacs_records_stored": 0,
            
            "noaa_status": "DISCONNECTED",
            "noaa_last_success": None,
            "noaa_last_error": None,
            "noaa_records_received": 0,
            "noaa_records_stored": 0,
            
            "last_sync": None,
            "next_sync": None,
            "logs": []
        }

    def update_gdacs(self, status, last_success=None, last_error=None, received=0, stored=0):
        with self._lock:
            self._status["gdacs_status"] = status
            if last_success:
                self._status["gdacs_last_success"] = last_success
            if last_error:
                self._status["gdacs_last_error"] = last_error
            self._status["gdacs_records_received"] = received
            self._status["gdacs_records_stored"] = stored
            self._add_log(f"GDACS Status updated: {status} (Rx: {received}, Stored: {stored})")

    def update_noaa(self, status, last_success=None, last_error=None, received=0, stored=0):
        with self._lock:
            self._status["noaa_status"] = status
            if last_success:
                self._status["noaa_last_success"] = last_success
            if last_error:
                self._status["noaa_last_error"] = last_error
            self._status["noaa_records_received"] = received
            self._status["noaa_records_stored"] = stored
            self._add_log(f"NOAA Status updated: {status} (Rx: {received}, Stored: {stored})")

    def update_sync_times(self, last_sync, next_sync):
        with self._lock:
            self._status["last_sync"] = last_sync
            self._status["next_sync"] = next_sync

    def add_log_message(self, message):
        with self._lock:
            self._add_log(message)

    def _add_log(self, message):
        timestamp = datetime.utcnow().strftime("%H:%M:%S UTC")
        log_entry = f"[{timestamp}] {message}"
        self._status["logs"].append(log_entry)
        # Keep only the last 50 logs to prevent memory leak
        if len(self._status["logs"]) > 50:
            self._status["logs"].pop(0)

    def get_status(self):
        with self._lock:
            return self._status.copy()

# Global thread-safe status manager instance
status_manager = SyncStatusManager()
