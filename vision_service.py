#!/usr/bin/env python3
"""
Vision Service - Reliable Windows service for screen capture with cursor
Based on the working llm_backup_service.py pattern
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# For Windows Service
try:
    import win32serviceutil
    import win32service
    import win32event
    import servicemanager
    SERVICE_AVAILABLE = True
except ImportError:
    SERVICE_AVAILABLE = False
    print("Warning: Windows service modules not available. Running in standalone mode.")

from capture.gdi_screen_capture import GDIScreenCapture


class VisionService:
    """Base vision service class (works with or without Windows service framework)"""
    
    def __init__(self):
        self.is_running = True
        self.capture = GDIScreenCapture()
        
        # Configuration
        self.session_folder = Path("claude_session")
        self.current_view = self.session_folder / "current_view.png"
        self.current_info = self.session_folder / "current_view_info.json"
        self.service_status = self.session_folder / "service" / "service_status.json"
        
        self.interval = 10  # 10 seconds between captures
        self.frame_count = 0
        
        # Ensure directories exist
        self.session_folder.mkdir(exist_ok=True)
        (self.session_folder / "service").mkdir(exist_ok=True)
        
    def capture_frame(self):
        """Capture current screen frame"""
        try:
            # Capture screen with cursor
            screen_data = self.capture.capture_primary_monitor(include_cursor=True)
            
            # Save capture
            self.capture.save_capture(screen_data, str(self.current_view))
            
            # Update info
            info_data = {
                "capture_time": datetime.now().isoformat(),
                "frame_count": self.frame_count,
                "image_path": str(self.current_view),
                "status": "active"
            }
            
            with open(self.current_info, 'w') as f:
                json.dump(info_data, f, indent=2)
            
            # Update service status
            status = {
                "running": True,
                "frame_count": self.frame_count,
                "last_capture": datetime.now().isoformat(),
                "interval": self.interval
            }
            
            with open(self.service_status, 'w') as f:
                json.dump(status, f, indent=2)
            
            self.frame_count += 1
            return True
            
        except Exception as e:
            print(f"Capture failed: {e}")
            return False
    
    def run_loop(self):
        """Main service loop"""
        print(f"Vision service started. Capturing every {self.interval} seconds.")
        print(f"Output: {self.current_view}")
        
        while self.is_running:
            self.capture_frame()
            
            # Sleep or wait for stop signal
            for i in range(self.interval):
                if not self.is_running:
                    break
                time.sleep(1)
        
        print("Vision service stopped.")
    
    def stop(self):
        """Stop the service"""
        self.is_running = False


if SERVICE_AVAILABLE:
    class WindowsVisionService(win32serviceutil.ServiceFramework, VisionService):
        """Windows Service version"""
        _svc_name_ = "AIVisionService"
        _svc_display_name_ = "AI Vision Service"
        _svc_description_ = "Screen capture service with cursor visibility for AI vision"

        def __init__(self, args):
            win32serviceutil.ServiceFramework.__init__(self, args)
            VisionService.__init__(self)
            self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)

        def SvcStop(self):
            self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
            win32event.SetEvent(self.hWaitStop)
            self.stop()

        def SvcDoRun(self):
            servicemanager.LogMsg(servicemanager.EVENTLOG_INFORMATION_TYPE,
                                servicemanager.PYS_SERVICE_STARTED,
                                (self._svc_name_, ''))
            self.run_loop_windows()
        
        def run_loop_windows(self):
            """Windows service main loop with proper event handling"""
            print(f"Windows Vision Service started. Capturing every {self.interval} seconds.")
            
            while self.is_running:
                self.capture_frame()
                
                # Wait with proper Windows event handling
                if win32event.WaitForSingleObject(self.hWaitStop, self.interval * 1000) == win32event.WAIT_OBJECT_0:
                    break
            
            print("Windows Vision Service stopped.")


def install_service():
    """Install Windows service"""
    if not SERVICE_AVAILABLE:
        print("Error: Windows service modules not available")
        return
        
    print("Installing AI Vision Service...")
    win32serviceutil.InstallService(
        win32serviceutil.GetServiceClassString(WindowsVisionService),
        WindowsVisionService._svc_name_,
        WindowsVisionService._svc_display_name_,
        startType=win32service.SERVICE_DEMAND_START
    )
    print("Installed! Commands:")
    print("  net start AIVisionService    - Start service")
    print("  net stop AIVisionService     - Stop service") 
    print("  sc delete AIVisionService    - Remove service")


def main():
    """Main entry point - defaults to standalone mode"""
    if len(sys.argv) > 1:
        if sys.argv[1] == 'install':
            install_service()
            return
        elif sys.argv[1] == 'test':
            # Test mode - single capture
            print("Testing vision capture...")
            service = VisionService()
            success = service.capture_frame()
            if success:
                print(f"Success! Capture saved to: {service.current_view}")
            else:
                print("Capture failed!")
            return
        elif sys.argv[1] == 'service':
            # Windows service mode
            if SERVICE_AVAILABLE:
                servicemanager.Initialize()
                servicemanager.PrepareToHostSingle(WindowsVisionService)
                servicemanager.StartServiceCtrlDispatcher()
            else:
                print("Error: Windows service modules not available")
            return
        else:
            # Handle other Windows service commands (start, stop, etc.)
            if SERVICE_AVAILABLE:
                win32serviceutil.HandleCommandLine(WindowsVisionService)
            else:
                print("Error: Windows service modules not available")
            return
    
    # DEFAULT: Run in standalone mode (no arguments needed)
    print("AI Vision Service - Enhanced Cursor Capture")
    print("Running in standalone mode...")
    service = VisionService()
    try:
        service.run_loop()
    except KeyboardInterrupt:
        print("\nStopping service...")
        service.stop()


if __name__ == '__main__':
    main()