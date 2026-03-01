"""
AI Session Manager
Handles communication between the AI system and screen capture for interactive sessions.
"""

import json
import time
import threading
import queue
import base64
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, asdict
from datetime import datetime
import os
import numpy as np
from PIL import Image
import io

# Import our capture modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from capture.gdi_screen_capture import GDIScreenCapture
from monitors.monitor_manager import MonitorManager
from regions.region_selector import RegionSelector


@dataclass
class SessionCommand:
    """Command structure for AI session control."""
    command: str
    parameters: Dict[str, Any]
    timestamp: float
    session_id: str
    command_id: str


@dataclass
class SessionResponse:
    """Response structure from session operations."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]]
    timestamp: float
    command_id: Optional[str] = None


@dataclass
class CaptureResult:
    """Result of a screen capture operation."""
    image_data: np.ndarray
    metadata: Dict[str, Any]
    timestamp: float
    region_name: Optional[str] = None


class AISessionManager:
    """Manages interactive sessions between AI and screen capture system."""
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or f"session_{int(time.time())}"
        self.active = False
        self.command_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # Initialize capture components
        self.screen_capture = GDIScreenCapture()
        self.monitor_manager = MonitorManager()
        self.region_selector = RegionSelector(self.screen_capture, self.monitor_manager)
        
        # Session state
        self.current_mode = "idle"  # idle, monitoring, interactive
        self.capture_thread = None
        self.command_thread = None
        self.settings = {
            "capture_interval": 0.5,  # Default 2 FPS for AI interaction
            "auto_save_images": True,
            "image_format": "PNG",
            "max_image_size": (1920, 1080),
            "compression_quality": 85
        }
        
        # Callback for external communication (e.g., to Claude Code)
        self.external_callback: Optional[Callable] = None
        
        # Image storage
        self.session_dir = f"sessions/{self.session_id}"
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Command handlers
        self.command_handlers = {
            "start_monitoring": self._handle_start_monitoring,
            "stop_monitoring": self._handle_stop_monitoring,
            "capture_now": self._handle_capture_now,
            "capture_region": self._handle_capture_region,
            "list_monitors": self._handle_list_monitors,
            "list_regions": self._handle_list_regions,
            "create_region": self._handle_create_region,
            "set_settings": self._handle_set_settings,
            "get_status": self._handle_get_status,
            "show_me_something": self._handle_show_me_something,
        }
    
    def start_session(self, external_callback: Optional[Callable] = None) -> SessionResponse:
        """Start the interactive session."""
        self.external_callback = external_callback
        self.active = True
        
        # Start command processing thread
        self.command_thread = threading.Thread(target=self._process_commands, daemon=True)
        self.command_thread.start()
        
        response = SessionResponse(
            success=True,
            message=f"AI Session {self.session_id} started successfully",
            data={"session_id": self.session_id, "session_dir": self.session_dir},
            timestamp=time.time()
        )
        
        self._log_session_event("SESSION_STARTED", {"session_id": self.session_id})
        return response
    
    def stop_session(self) -> SessionResponse:
        """Stop the interactive session."""
        self.active = False
        
        # Stop monitoring if active
        if self.current_mode == "monitoring":
            self._stop_monitoring_internal()
        
        response = SessionResponse(
            success=True,
            message=f"AI Session {self.session_id} stopped",
            data={"total_commands": self.command_queue.qsize()},
            timestamp=time.time()
        )
        
        self._log_session_event("SESSION_STOPPED", {"session_id": self.session_id})
        return response
    
    def send_command(self, command: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Send a command to the session. Returns command ID for tracking."""
        command_id = f"cmd_{int(time.time() * 1000)}"
        
        session_command = SessionCommand(
            command=command,
            parameters=parameters or {},
            timestamp=time.time(),
            session_id=self.session_id,
            command_id=command_id
        )
        
        self.command_queue.put(session_command)
        return command_id
    
    def get_response(self, timeout: float = 5.0) -> Optional[SessionResponse]:
        """Get the next response from the session."""
        try:
            return self.response_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def _process_commands(self):
        """Process commands in the background thread."""
        while self.active:
            try:
                # Get command with timeout
                try:
                    command = self.command_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process command
                handler = self.command_handlers.get(command.command)
                if handler:
                    try:
                        response = handler(command)
                        response.command_id = command.command_id
                    except Exception as e:
                        response = SessionResponse(
                            success=False,
                            message=f"Error executing {command.command}: {str(e)}",
                            data=None,
                            timestamp=time.time(),
                            command_id=command.command_id
                        )
                else:
                    response = SessionResponse(
                        success=False,
                        message=f"Unknown command: {command.command}",
                        data=None,
                        timestamp=time.time(),
                        command_id=command.command_id
                    )
                
                # Send response
                self.response_queue.put(response)
                self._log_session_event("COMMAND_PROCESSED", {
                    "command": command.command,
                    "success": response.success,
                    "command_id": command.command_id
                })
                
            except Exception as e:
                print(f"Error in command processing: {e}")
    
    def _handle_start_monitoring(self, command: SessionCommand) -> SessionResponse:
        """Handle start monitoring command."""
        if self.current_mode == "monitoring":
            return SessionResponse(
                success=False,
                message="Already monitoring",
                data=None,
                timestamp=time.time()
            )
        
        # Extract parameters
        interval = command.parameters.get("interval", self.settings["capture_interval"])
        region_name = command.parameters.get("region", "primary_monitor")
        
        # Start monitoring
        self.current_mode = "monitoring"
        self.capture_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(interval, region_name), 
            daemon=True
        )
        self.capture_thread.start()
        
        return SessionResponse(
            success=True,
            message=f"Started monitoring {region_name} at {interval}s intervals",
            data={
                "mode": "monitoring",
                "interval": interval,
                "region": region_name
            },
            timestamp=time.time()
        )
    
    def _handle_stop_monitoring(self, command: SessionCommand) -> SessionResponse:
        """Handle stop monitoring command."""
        if self.current_mode != "monitoring":
            return SessionResponse(
                success=False,
                message="Not currently monitoring",
                data=None,
                timestamp=time.time()
            )
        
        self._stop_monitoring_internal()
        
        return SessionResponse(
            success=True,
            message="Stopped monitoring",
            data={"mode": "idle"},
            timestamp=time.time()
        )
    
    def _handle_capture_now(self, command: SessionCommand) -> SessionResponse:
        """Handle immediate capture command."""
        try:
            # Get capture parameters
            region_name = command.parameters.get("region", "primary_monitor")
            save_image = command.parameters.get("save", True)
            
            # Perform capture
            if region_name == "primary_monitor":
                image_data = self.screen_capture.capture_primary_monitor()
                metadata = {"type": "primary_monitor", "size": image_data.shape}
            else:
                # Try to capture named region
                region = self.region_selector.get_region_by_name(region_name)
                if region:
                    image_data = self.screen_capture.capture_screen_region(
                        region.x, region.y, region.width, region.height
                    )
                    metadata = {
                        "type": "region",
                        "region_name": region_name,
                        "size": image_data.shape,
                        "position": (region.x, region.y)
                    }
                else:
                    return SessionResponse(
                        success=False,
                        message=f"Region '{region_name}' not found",
                        data=None,
                        timestamp=time.time()
                    )
            
            # Process and save image
            result = self._process_captured_image(image_data, metadata, region_name, save_image)
            
            # Notify external callback if set
            if self.external_callback:
                self.external_callback("image_captured", result)
            
            return SessionResponse(
                success=True,
                message="Capture completed successfully",
                data={
                    "image_path": result.get("image_path"),
                    "metadata": metadata,
                    "timestamp": time.time()
                },
                timestamp=time.time()
            )
            
        except Exception as e:
            return SessionResponse(
                success=False,
                message=f"Capture failed: {str(e)}",
                data=None,
                timestamp=time.time()
            )
    
    def _handle_create_region(self, command: SessionCommand) -> SessionResponse:
        """Handle create custom region command."""
        try:
            # Get region parameters
            params = command.parameters
            region_id = params.get("id")
            region_name = params.get("name", f"Custom Region {region_id}")
            
            if not region_id:
                return SessionResponse(
                    success=False,
                    message="Region ID is required",
                    data=None,
                    timestamp=time.time(),
                    command_id=command.command_id
                )
            
            # Create region definition
            region_def = {
                "id": region_id,
                "name": region_name,
                "x": params.get("x", 0),
                "y": params.get("y", 0),
                "width": params.get("width", 800), 
                "height": params.get("height", 600),
                "created_at": time.time(),
                "custom": True
            }
            
            # Store region (in a real implementation, this would be persisted)
            if not hasattr(self, 'custom_regions'):
                self.custom_regions = {}
            
            self.custom_regions[region_id] = region_def
            
            return SessionResponse(
                success=True,
                message=f"Created custom region: {region_name}",
                data={"region": region_def},
                timestamp=time.time(),
                command_id=command.command_id
            )
            
        except Exception as e:
            return SessionResponse(
                success=False,
                message=f"Failed to create region: {str(e)}",
                data=None,
                timestamp=time.time(),
                command_id=command.command_id
            )

    def _handle_set_settings(self, command: SessionCommand) -> SessionResponse:
        """Handle settings update command."""
        try:
            settings = command.parameters.get("settings", {})
            
            # Update capture settings
            if "capture_format" in settings:
                self.capture_format = settings["capture_format"]
            
            if "compression_quality" in settings:
                self.compression_quality = settings["compression_quality"]
                
            if "monitoring_interval" in settings:
                self.monitoring_interval = settings["monitoring_interval"]
            
            # Store settings (in a real implementation, this would be persisted)
            if not hasattr(self, 'current_settings'):
                self.current_settings = {}
            
            self.current_settings.update(settings)
            
            return SessionResponse(
                success=True,
                message=f"Updated {len(settings)} settings",
                data={"settings": self.current_settings},
                timestamp=time.time(),
                command_id=command.command_id
            )
            
        except Exception as e:
            return SessionResponse(
                success=False,
                message=f"Failed to update settings: {str(e)}",
                data=None,
                timestamp=time.time(),
                command_id=command.command_id
            )

    def _handle_list_regions(self, command: SessionCommand) -> SessionResponse:
        """Handle list available regions command."""
        try:
            # Get monitor information
            monitors = self.monitor_manager.get_monitors()
            
            # Create region suggestions based on monitors
            regions = []
            for i, monitor in enumerate(monitors):
                regions.append({
                    "id": f"monitor_{i}",
                    "name": f"Monitor {i+1}",
                    "x": monitor.get("left", 0),
                    "y": monitor.get("top", 0), 
                    "width": monitor.get("width", 1920),
                    "height": monitor.get("height", 1080),
                    "primary": monitor.get("primary", False)
                })
            
            # Add some common region presets
            primary_monitor = next((m for m in monitors if m.get("primary")), monitors[0] if monitors else {})
            if primary_monitor:
                w = primary_monitor.get("width", 1920)
                h = primary_monitor.get("height", 1080)
                
                regions.extend([
                    {
                        "id": "center_quarter",
                        "name": "Center Quarter",
                        "x": w//4, "y": h//4,
                        "width": w//2, "height": h//2
                    },
                    {
                        "id": "top_half", 
                        "name": "Top Half",
                        "x": 0, "y": 0,
                        "width": w, "height": h//2
                    },
                    {
                        "id": "bottom_half",
                        "name": "Bottom Half", 
                        "x": 0, "y": h//2,
                        "width": w, "height": h//2
                    }
                ])
            
            return SessionResponse(
                success=True,
                message=f"Found {len(regions)} available regions",
                data={"regions": regions},
                timestamp=time.time(),
                command_id=command.command_id
            )
            
        except Exception as e:
            return SessionResponse(
                success=False,
                message=f"Failed to list regions: {str(e)}",
                data=None,
                timestamp=time.time(),
                command_id=command.command_id
            )

    def _handle_capture_region(self, command: SessionCommand) -> SessionResponse:
        """Handle region-based capture command."""
        try:
            # Get region parameters
            region = command.parameters.get("region")
            if not region:
                return SessionResponse(
                    success=False,
                    message="No region specified for capture",
                    data=None,
                    timestamp=time.time(),
                    command_id=command.command_id
                )
            
            # Capture the specified region
            x = region.get("x", 0)
            y = region.get("y", 0) 
            width = region.get("width", 800)
            height = region.get("height", 600)
            
            image_data = self.screen_capture.capture_region(x, y, width, height)
            
            metadata = {
                "type": "region_capture",
                "region": region,
                "size": image_data.shape,
                "capture_time": time.time(),
                "purpose": command.parameters.get("purpose", "region_analysis")
            }
            
            # Process and save
            result = self._process_captured_image(image_data, metadata, "region_capture", 
                                               command.parameters.get("copy_to_workspace", False))
            
            return SessionResponse(
                success=True,
                message=f"Region captured: {width}x{height} at ({x},{y})",
                data=result,
                timestamp=time.time(),
                command_id=command.command_id
            )
            
        except Exception as e:
            return SessionResponse(
                success=False,
                message=f"Region capture failed: {str(e)}",
                data=None,
                timestamp=time.time(),
                command_id=command.command_id
            )
    
    def _handle_show_me_something(self, command: SessionCommand) -> SessionResponse:
        """Handle 'show me something' command - user wants to show the AI something."""
        try:
            # Capture current screen
            image_data = self.screen_capture.capture_primary_monitor()
            metadata = {
                "type": "user_requested",
                "purpose": "show_ai",
                "size": image_data.shape,
                "message": command.parameters.get("message", "User wants to show something")
            }
            
            # Process and save
            result = self._process_captured_image(image_data, metadata, "user_show", True)
            
            # Create a special response for the AI
            ai_message = f"📸 User is showing me their screen! "
            ai_message += f"Captured {image_data.shape[1]}x{image_data.shape[0]} image. "
            
            if command.parameters.get("message"):
                ai_message += f"User says: '{command.parameters['message']}'"
            
            # Notify external callback
            if self.external_callback:
                self.external_callback("user_showing_screen", {
                    "image_path": result.get("image_path"),
                    "metadata": metadata,
                    "ai_message": ai_message
                })
            
            return SessionResponse(
                success=True,
                message=ai_message,
                data=result,
                timestamp=time.time()
            )
            
        except Exception as e:
            return SessionResponse(
                success=False,
                message=f"Failed to capture what you're showing: {str(e)}",
                data=None,
                timestamp=time.time()
            )
    
    def _handle_list_monitors(self, command: SessionCommand) -> SessionResponse:
        """Handle list monitors command."""
        monitors = self.monitor_manager.get_monitors()
        monitor_info = []
        
        for monitor in monitors:
            monitor_info.append({
                "index": monitor.index,
                "name": monitor.name,
                "resolution": f"{monitor.width}x{monitor.height}",
                "position": (monitor.left, monitor.top),
                "is_primary": monitor.is_primary,
                "dpi": f"{monitor.dpi_x}x{monitor.dpi_y}"
            })
        
        return SessionResponse(
            success=True,
            message=f"Found {len(monitors)} monitor(s)",
            data={"monitors": monitor_info},
            timestamp=time.time()
        )
    
    def _handle_get_status(self, command: SessionCommand) -> SessionResponse:
        """Handle get status command."""
        status = {
            "session_id": self.session_id,
            "active": self.active,
            "mode": self.current_mode,
            "settings": self.settings,
            "monitors_count": len(self.monitor_manager.get_monitors()),
            "regions_count": len(self.region_selector.regions),
            "session_dir": self.session_dir
        }
        
        return SessionResponse(
            success=True,
            message=f"Session status for {self.session_id}",
            data=status,
            timestamp=time.time()
        )
    
    def _monitoring_loop(self, interval: float, region_name: str):
        """Background monitoring loop."""
        capture_count = 0
        
        while self.current_mode == "monitoring" and self.active:
            try:
                # Perform capture based on region
                if region_name == "primary_monitor":
                    image_data = self.screen_capture.capture_primary_monitor()
                    metadata = {"type": "monitoring", "region": "primary_monitor"}
                else:
                    region = self.region_selector.get_region_by_name(region_name)
                    if region:
                        image_data = self.screen_capture.capture_screen_region(
                            region.x, region.y, region.width, region.height
                        )
                        metadata = {"type": "monitoring", "region": region_name}
                    else:
                        print(f"Region {region_name} not found, switching to primary monitor")
                        image_data = self.screen_capture.capture_primary_monitor()
                        metadata = {"type": "monitoring", "region": "primary_monitor"}
                
                capture_count += 1
                metadata.update({
                    "capture_count": capture_count,
                    "size": image_data.shape,
                    "monitoring_session": self.session_id
                })
                
                # Process image
                result = self._process_captured_image(
                    image_data, metadata, f"monitor_{capture_count:06d}", 
                    self.settings["auto_save_images"]
                )
                
                # Notify external callback
                if self.external_callback:
                    self.external_callback("monitoring_capture", result)
                
                # Wait for next capture
                time.sleep(interval)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(1)  # Wait before retry
    
    def _stop_monitoring_internal(self):
        """Internal method to stop monitoring."""
        self.current_mode = "idle"
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
    
    def _process_captured_image(self, image_data: np.ndarray, metadata: Dict[str, Any], 
                              name_prefix: str, save: bool) -> Dict[str, Any]:
        """Process captured image data and optionally save it."""
        result = {
            "shape": image_data.shape,
            "dtype": str(image_data.dtype),
            "metadata": metadata,
            "timestamp": time.time()
        }
        
        if save:
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{name_prefix}_{timestamp}.png"
            filepath = os.path.join(self.session_dir, filename)
            
            # Save image
            pil_image = Image.fromarray(image_data)
            
            # Resize if too large
            if pil_image.size[0] > self.settings["max_image_size"][0] or \
               pil_image.size[1] > self.settings["max_image_size"][1]:
                pil_image.thumbnail(self.settings["max_image_size"], Image.Resampling.LANCZOS)
            
            pil_image.save(filepath, "PNG", optimize=True)
            
            result["image_path"] = filepath
            result["filename"] = filename
            result["file_size"] = os.path.getsize(filepath)
        
        # Create base64 encoded thumbnail for quick viewing
        pil_image = Image.fromarray(image_data)
        pil_image.thumbnail((200, 150), Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        thumbnail_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        result["thumbnail"] = thumbnail_b64
        
        return result
    
    def _log_session_event(self, event_type: str, data: Dict[str, Any]):
        """Log session events to file."""
        log_entry = {
            "timestamp": time.time(),
            "event_type": event_type,
            "data": data,
            "session_id": self.session_id
        }
        
        log_file = os.path.join(self.session_dir, "session.log")
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")


# Convenience functions for external integration
def create_ai_session(session_id: Optional[str] = None) -> AISessionManager:
    """Create and start a new AI session."""
    session = AISessionManager(session_id)
    session.start_session()
    return session


def quick_capture_for_ai(message: str = "Quick capture") -> Dict[str, Any]:
    """Quick function to capture screen for AI analysis."""
    session = create_ai_session()
    try:
        command_id = session.send_command("capture_now", {"save": True, "message": message})
        response = session.get_response(timeout=10.0)
        
        if response and response.success:
            return response.data
        else:
            return {"error": response.message if response else "Timeout"}
    finally:
        session.stop_session()


if __name__ == "__main__":
    # Test the session manager
    print("Testing AI Session Manager...")
    
    session = create_ai_session("test_session")
    
    # Test commands
    commands_to_test = [
        ("get_status", {}),
        ("list_monitors", {}),
        ("capture_now", {"save": True}),
        ("show_me_something", {"message": "Testing the session system"}),
    ]
    
    for cmd, params in commands_to_test:
        print(f"\nSending command: {cmd}")
        command_id = session.send_command(cmd, params)
        
        response = session.get_response(timeout=10.0)
        if response:
            print(f"Response: {response.message}")
            if response.data:
                print(f"Data: {list(response.data.keys())}")
        else:
            print("No response received")
    
    print("\nStopping session...")
    session.stop_session()
    print("Test completed!")