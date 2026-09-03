import sys
import os

# Append project root directory to path to resolve backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.app.services.live_sync import execute_sync
from backend.app.services.status_cache import status_manager

def main():
    print("====================================================")
    print("CYCLONEAI - DATA INGESTION SYNC MANUAL TEST RUN")
    print("====================================================")
    
    print("\n[1] Starting Sync Execution...")
    try:
        execute_sync()
        print("[SUCCESS] Sync executed successfully.")
    except Exception as e:
        print(f"[ERROR] Sync execution failed: {e}")
        
    print("\n[2] Reading Cache System Status & Metrics...")
    status = status_manager.get_status()
    
    print(f"GDACS Status: {status['gdacs_status']}")
    print(f"GDACS Last Success: {status['gdacs_last_success']}")
    print(f"GDACS Last Error: {status['gdacs_last_error']}")
    print(f"GDACS Records Received: {status['gdacs_records_received']}")
    print(f"GDACS Records Stored: {status['gdacs_records_stored']}")
    
    print("-" * 50)
    print(f"NOAA Status: {status['noaa_status']}")
    print(f"NOAA Last Success: {status['noaa_last_success']}")
    print(f"NOAA Last Error: {status['noaa_last_error']}")
    print(f"NOAA Records Received: {status['noaa_records_received']}")
    print(f"NOAA Records Stored: {status['noaa_records_stored']}")
    
    print("\n[3] Showing Thread-Safe Synchronization Logs:")
    for log in status["logs"]:
        print(f"  {log}")
        
    print("\n====================================================")
    print("TEST COMPLETED")
    print("====================================================")

if __name__ == "__main__":
    main()
