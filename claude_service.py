#!/usr/bin/env python3
"""
Claude Background Service - Continuous Vision Streaming
Runs as a background service that streams images for Claude with automatic archiving.
"""

import os
import sys
import time
import json
import shutil
import threading
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import queue
import signal
import subprocess
import tempfile
import stat

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from capture.gdi_screen_capture import GDIScreenCapture
from config.vision_config import VisionConfigManager, get_config_manager


class ClaudeVisionService:
    """Background service for Claude vision streaming."""
    
    def __init__(self, session_folder: Optional[str] = None):
        self.capture = GDIScreenCapture()
        self.previous_screen = None  # Store previous screen capture
        
        # Service configuration
        self.service_name = "ClaudeVisionService"
        
        # Use secure base directory if no session folder specified
        if session_folder:
            self.session_folder = Path(session_folder)
        else:
            self.session_folder = self.get_secure_base_dir() / "claude_session"
        
        self.service_folder = self.session_folder / "service"
        
        # Create directories with secure permissions
        for folder in [self.session_folder, self.service_folder]:
            self.create_secure_directory(folder)
        
        # Service files
        self.current_view = self.session_folder / "current_view.png"
        self.current_info = self.session_folder / "current_view_info.json"
        self.service_status = self.service_folder / "service_status.json"
        self.service_log = self.service_folder / "service.log"
        self.command_file = self.service_folder / "commands.json"
        self.response_file = self.service_folder / "response.json"
        
        # Claude interaction files
        self.claude_viewed = self.session_folder / "VIEWED.txt"
        self.claude_ready = self.session_folder / "READY.txt"
        self.claude_message = self.session_folder / "MESSAGE.txt"
        
        # Archive management
        self.archive_folder = self.session_folder / "archive"
        self.create_secure_directory(self.archive_folder)
        
        # Retention policy settings
        self.max_archive_files = 500
        self.max_archive_age_days = 7
        
        # Service state
        self.running = False
        self.streaming = False
        self.stream_interval = 10.0  # Reduced flickering - capture every 10 seconds
        self.frame_count = 0
        self.last_capture_time = 0
        self.current_frame_id = None  # Current frame identifier for race condition prevention
        self.frame_sequence = 0  # Monotonic sequence counter
        
        # Threading
        self.service_thread: Optional[threading.Thread] = None
        self.command_queue = queue.Queue()
        
        # Logging
        self.setup_logging()
        
        self.log("Claude Vision Service initialized")
        self.log(f"Session folder: {self.session_folder}")
    
    def get_secure_base_dir(self) -> Path:
        """Get secure base directory in %LOCALAPPDATA%\AIVision."""
        try:
            # Get %LOCALAPPDATA% on Windows, fallback to home directory
            if os.name == 'nt':
                local_appdata = os.environ.get('LOCALAPPDATA')
                if local_appdata:
                    base_dir = Path(local_appdata) / "AIVision"
                else:
                    base_dir = Path.home() / "AppData" / "Local" / "AIVision"
            else:
                base_dir = Path.home() / ".config" / "ai-vision"
            
            # Create directory with secure ACLs
            if not base_dir.exists():
                self.create_secure_directory(base_dir)
                self.log(f"Created secure base directory: {base_dir}")
            
            return base_dir
            
        except Exception as e:
            self.log(f"Failed to create secure base directory, using current dir: {e}", "warning")
            return Path(".")  # Fallback to current directory
    
    def create_secure_directory(self, path: Path):
        """Create directory with locked DACL (current user + SYSTEM only)."""
        try:
            # Create directory first
            path.mkdir(parents=True, exist_ok=True)
            
            if os.name == 'nt':
                # Try to set Windows ACLs
                try:
                    import win32security
                    import win32api
                    import ntsecuritycon
                    
                    # Get current user SID
                    current_user = win32api.GetUserName()
                    user_sid, _, _ = win32security.LookupAccountName(None, current_user)
                    
                    # Get SYSTEM SID
                    system_sid = win32security.ConvertStringSidToSid('S-1-5-18')
                    
                    # Create new DACL
                    dacl = win32security.ACL()
                    
                    # Add full control for current user
                    dacl.AddAccessAllowedAce(
                        win32security.ACL_REVISION,
                        ntsecuritycon.FILE_ALL_ACCESS,
                        user_sid
                    )
                    
                    # Add full control for SYSTEM
                    dacl.AddAccessAllowedAce(
                        win32security.ACL_REVISION,
                        ntsecuritycon.FILE_ALL_ACCESS,
                        system_sid
                    )
                    
                    # Create security descriptor
                    security_desc = win32security.SECURITY_DESCRIPTOR()
                    security_desc.SetSecurityDescriptorDacl(1, dacl, 0)
                    
                    # Apply to directory
                    win32security.SetFileSecurity(
                        str(path),
                        win32security.DACL_SECURITY_INFORMATION,
                        security_desc
                    )
                    
                    self.log(f"Applied Windows ACL security to: {path}")
                    
                except ImportError:
                    self.log("win32security not available, using basic permissions", "warning")
                    self._set_basic_permissions(path)
                except Exception as e:
                    self.log(f"Failed to set Windows ACLs: {e}, using basic permissions", "warning")
                    self._set_basic_permissions(path)
            else:
                # Unix-like systems: set 700 permissions
                os.chmod(path, stat.S_IRWXU)
                self.log(f"Set Unix permissions (700) for: {path}")
                
        except Exception as e:
            self.log(f"Failed to secure directory {path}: {e}", "error")
    
    def _set_basic_permissions(self, path: Path):
        """Set basic restrictive permissions as fallback."""
        try:
            # Set restrictive permissions (owner only)
            if os.name == 'nt':
                # Windows: Remove inheritance and set owner-only access
                os.chmod(path, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
            else:
                # Unix: 700 permissions
                os.chmod(path, stat.S_IRWXU)
            
            self.log(f"Set basic restrictive permissions for: {path}")
        except Exception as e:
            self.log(f"Failed to set basic permissions: {e}", "warning")
    
    def migrate_existing_session(self, old_session_path: Path):
        """Migrate existing session to secure location."""
        if not old_session_path.exists():
            return
        
        try:
            self.log(f"Migrating session from {old_session_path} to {self.session_folder}")
            
            # Create secure destination
            self.create_secure_directory(self.session_folder.parent)
            
            # Copy files
            if self.session_folder.exists():
                backup_path = self.session_folder.with_suffix('.backup')
                shutil.move(str(self.session_folder), str(backup_path))
                self.log(f"Backed up existing session to: {backup_path}")
            
            shutil.copytree(str(old_session_path), str(self.session_folder))
            
            # Secure the migrated directory
            self.create_secure_directory(self.session_folder)
            for subdir in self.session_folder.rglob('*'):
                if subdir.is_dir():
                    self.create_secure_directory(subdir)
            
            self.log(f"Session migration completed successfully")
            
        except Exception as e:
            self.log(f"Session migration failed: {e}", "error")
    
    def atomic_write_json(self, path: Path, obj: Dict[str, Any]):
        """Atomically write JSON data to a file using temp-then-rename."""
        try:
            # Write to temporary file in the same directory
            temp_path = path.parent / f"{path.stem}_temp{path.suffix}"
            
            with open(temp_path, 'w') as f:
                json.dump(obj, f, indent=2)
                f.flush()  # Ensure data is written to disk
                os.fsync(f.fileno())  # Force write to storage
            
            # Atomic rename with retry for Windows file locking
            self._atomic_rename_with_retry(temp_path, path)
            
        except Exception as e:
            # Cleanup temp file if it exists
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except:
                pass
            raise e
    
    def atomic_write_image(self, image_data, path: Path):
        """Atomically write image data to a file using temp-then-rename."""
        try:
            # Create temp file with proper image extension
            temp_path = path.parent / f"{path.stem}_temp{path.suffix}"
            
            # Use the capture's save method to write to temp file
            self.capture.save_capture(image_data, str(temp_path))
            
            # Atomic rename with retry and exponential backoff for Windows file locking
            self._atomic_rename_with_retry(temp_path, path)
            
        except Exception as e:
            # Cleanup temp file if it exists
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except:
                pass
            raise e
    
    def _atomic_rename_with_retry(self, temp_path: Path, target_path: Path, max_retries: int = 3):
        """Perform atomic rename with retry and reduced flickering for Windows."""
        import random
        
        # First, try to minimize viewer flickering by using a very fast copy operation
        # instead of rename when possible
        try:
            # If target exists, try to overwrite it directly with minimal interruption
            if target_path.exists():
                # Use Windows API for faster file replacement with minimal flicker
                import ctypes
                from ctypes import wintypes
                
                try:
                    # Use ReplaceFile API which is designed to minimize flicker
                    kernel32 = ctypes.windll.kernel32
                    result = kernel32.ReplaceFileW(
                        ctypes.c_wchar_p(str(target_path)),  # existing file
                        ctypes.c_wchar_p(str(temp_path)),    # replacement file
                        None,  # backup file (not used)
                        0,     # flags
                        None,  # exclude
                        None   # reserved
                    )
                    if result:
                        return  # Success with minimal flicker
                except:
                    pass  # Fall back to standard method
        except:
            pass
        
        # Fall back to standard atomic rename with retry
        for attempt in range(max_retries):
            try:
                os.replace(temp_path, target_path)
                return  # Success
            except OSError as e:
                if e.errno == 32 and attempt < max_retries - 1:  # File in use, not last attempt
                    # Reduced delay for faster updates: 10ms, 20ms, 40ms
                    delay = (10 + random.randint(0, 5)) * (2 ** attempt) / 1000.0
                    time.sleep(delay)
                    continue
                else:
                    raise  # Re-raise on last attempt or different error
    
    def setup_logging(self):
        """Set up service logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.service_log),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.service_name)
    
    def log(self, message: str, level: str = "info"):
        """Log a message."""
        if hasattr(self, 'logger') and self.logger:
            if level == "error":
                self.logger.error(message)
            elif level == "warning":
                self.logger.warning(message)
            else:
                self.logger.info(message)
        else:
            print(f"[{level.upper()}] {message}")
    
    def update_service_status(self):
        """Update the service status file."""
        status = {
            "service_name": self.service_name,
            "running": self.running,
            "streaming": self.streaming,
            "frame_count": self.frame_count,
            "stream_interval": self.stream_interval,
            "last_capture": self.last_capture_time,
            # Remove absolute paths - only use relative/anonymized identifiers
            "session_id": self.session_folder.name,  # Just the folder name, not full path
            "archive_count": len(list(self.archive_folder.glob("view_*.png"))) if self.archive_folder.exists() else 0,
            "pid": os.getpid(),
            "start_time": getattr(self, 'start_time', time.time()),
            "uptime_seconds": time.time() - getattr(self, 'start_time', time.time()),
            "current_frame_id": self.current_frame_id,
            "frame_sequence": self.frame_sequence,
            # Additional non-sensitive fields
            "build_version": "1.0.0",
            "service_mode": "background_streaming",
            "retention_policy": {
                "max_files": self.max_archive_files,
                "max_age_days": self.max_archive_age_days
            },
            "for_claude": {
                "current_image": "current_view.png (or .jpg/.webp depending on format)",
                "to_archive": "Create VIEWED.txt file when done viewing (optionally include frame_id)",
                "to_message": "Write to MESSAGE.txt to communicate",
                "status_check": "Check this service_status.json file",
                "frame_tracking": {
                    "current_frame_id": self.current_frame_id,
                    "frame_sequence": self.frame_sequence,
                    "usage": "Write frame_id to VIEWED.txt for precise tracking"
                }
            }
        }
        
        try:
            self.atomic_write_json(self.service_status, status)
        except Exception as e:
            self.log(f"Failed to update status: {e}", "error")
    
    def capture_frame(self) -> bool:
        """Capture current screen frame with per-frame ID tracking."""
        try:
            # Capture screen normally with enhanced cursor
            screen_data = self.capture.capture_primary_monitor(include_cursor=True)
            
            # Generate unique frame ID to prevent race conditions
            import uuid
            import hashlib
            
            # Create frame ID combining timestamp, sequence, and content hash
            capture_timestamp = time.time()
            self.frame_sequence += 1
            
            # Create content hash for verification (first 1KB of image data)
            content_sample = screen_data.flatten()[:1024].tobytes()
            content_hash = hashlib.md5(content_sample).hexdigest()[:12]
            
            # Generate unique frame ID
            frame_id = f"frame_{int(capture_timestamp * 1000000)}_{self.frame_sequence:06d}_{content_hash}"
            self.current_frame_id = frame_id
            
            # Save as fixed filename (always current_view.png) - atomically
            self.atomic_write_image(screen_data, self.current_view)
            
            # Update info file with frame tracking
            info_data = {
                "image_file": "current_view.png",
                "frame_number": self.frame_count,
                "frame_id": frame_id,
                "frame_sequence": self.frame_sequence,
                "capture_time": capture_timestamp,
                "datetime": datetime.now().isoformat(),
                "screen_size": f"{screen_data.shape[1]}x{screen_data.shape[0]}",
                "service_mode": "background_streaming",
                "content_hash": content_hash,
                "frame_verification": {
                    "timestamp_us": int(capture_timestamp * 1000000),
                    "sequence": self.frame_sequence,
                    "hash_sample": content_hash
                },
                "for_claude": "This image updates every few seconds. Create VIEWED.txt when done viewing to archive. Optionally include the frame_id for precise tracking."
            }
            
            # Atomically write info file
            self.atomic_write_json(self.current_info, info_data)
            
            self.frame_count += 1
            self.last_capture_time = capture_timestamp
            
            return True
            
        except Exception as e:
            self.log(f"Capture failed: {e}", "error")
            return False
    
    def check_claude_viewed(self) -> Optional[str]:
        """Check if Claude has viewed the current image and return frame ID if specified."""
        if not self.claude_viewed.exists():
            return None
        
        try:
            # Try to read frame ID from VIEWED.txt for verification
            with open(self.claude_viewed, 'r') as f:
                viewed_content = f.read().strip()
            
            # Check if it contains a frame ID
            if viewed_content and viewed_content.startswith('frame_'):
                return viewed_content  # Return the specific frame ID that was viewed
            else:
                return "viewed"  # Generic viewed signal
                
        except Exception:
            # File exists but couldn't read - treat as generic viewed
            return "viewed"
    
    def archive_current_view(self) -> bool:
        """Archive the current view to the archive folder (legacy method)."""
        return self.archive_current_view_with_id("legacy")
    
    def archive_current_view_with_id(self, viewed_frame_id: str) -> bool:
        """Archive the current view with frame ID verification."""
        if not self.current_view.exists():
            return False
        
        try:
            # Verify frame ID matches current frame if specified
            if viewed_frame_id != "viewed" and viewed_frame_id != "legacy" and self.current_frame_id:
                if viewed_frame_id != self.current_frame_id:
                    self.log(f"Frame ID mismatch: viewed={viewed_frame_id}, current={self.current_frame_id}", "warning")
                    # Still archive but note the mismatch
            
            # Read current info to get frame metadata
            frame_info = {}
            if self.current_info.exists():
                try:
                    with open(self.current_info, 'r') as f:
                        frame_info = json.load(f)
                except Exception as e:
                    self.log(f"Failed to read frame info: {e}", "warning")
            
            # Create archive filename with frame ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            frame_suffix = f"_{viewed_frame_id}" if viewed_frame_id not in ["viewed", "legacy"] else ""
            
            # Get file extension from current view
            file_ext = self.current_view.suffix
            archive_name = f"view_{timestamp}_frame_{self.frame_count:06d}{frame_suffix}{file_ext}"
            archive_path = self.archive_folder / archive_name
            
            # Copy to archive
            shutil.copy2(self.current_view, archive_path)
            
            # Archive enhanced info file with verification data
            if frame_info or self.current_info.exists():
                archive_info_name = f"view_{timestamp}_frame_{self.frame_count:06d}{frame_suffix}_info.json"
                archive_info_path = self.archive_folder / archive_info_name
                
                # Add archive metadata
                archive_metadata = frame_info.copy() if frame_info else {}
                archive_metadata.update({
                    "archived_at": time.time(),
                    "archived_datetime": datetime.now().isoformat(),
                    "viewed_frame_id": viewed_frame_id,
                    "current_frame_id": self.current_frame_id,
                    "frame_id_match": (viewed_frame_id == self.current_frame_id) if self.current_frame_id else None,
                    "archive_filename": archive_name,
                    "original_filename": self.current_view.name
                })
                
                self.atomic_write_json(archive_info_path, archive_metadata)
            
            # Remove Claude's viewed signal
            try:
                self.claude_viewed.unlink()
            except:
                pass
            
            self.log(f"Archived: {archive_name}")
            
            # Cleanup old archives after archiving
            self.cleanup_archives()
            
            return True
            
        except Exception as e:
            self.log(f"Archive failed: {e}", "error")
            return False
    
    def check_claude_message(self) -> Optional[str]:
        """Check if Claude has left a message."""
        if self.claude_message.exists():
            try:
                with open(self.claude_message, 'r') as f:
                    message = f.read().strip()
                
                # Remove message file after reading
                self.claude_message.unlink()
                
                return message
            except:
                pass
        
        return None
    
    def handle_claude_message(self, message: str):
        """Handle a message from Claude."""
        self.log(f"Claude message: {message}")
        
        # Simple command processing
        message_lower = message.lower()
        
        if "stop" in message_lower or "pause" in message_lower:
            self.streaming = False
            self.log("Streaming paused by Claude")
            
        elif "start" in message_lower or "resume" in message_lower:
            self.streaming = True
            self.log("Streaming resumed by Claude")
            
        elif "faster" in message_lower:
            self.stream_interval = max(1.0, self.stream_interval - 1.0)
            self.log(f"Interval decreased to {self.stream_interval}s")
            
        elif "slower" in message_lower:
            self.stream_interval = min(10.0, self.stream_interval + 1.0)
            self.log(f"Interval increased to {self.stream_interval}s")
            
        elif "capture" in message_lower:
            self.capture_frame()
            self.log("Manual capture triggered by Claude")
    
    def process_commands(self):
        """Process any pending commands."""
        if self.command_file.exists():
            try:
                with open(self.command_file, 'r') as f:
                    command = json.load(f)
                
                # Remove command file
                self.command_file.unlink()
                
                # Process command
                cmd_type = command.get('type')
                
                if cmd_type == 'stop_service':
                    self.running = False
                    self.log("Service stop requested")
                    
                elif cmd_type == 'start_streaming':
                    self.streaming = True
                    self.stream_interval = command.get('interval', 4.0)
                    self.log(f"Streaming started (interval: {self.stream_interval}s)")
                    
                elif cmd_type == 'stop_streaming':
                    self.streaming = False
                    self.log("Streaming stopped")
                    
                elif cmd_type == 'change_interval':
                    self.stream_interval = command.get('interval', 4.0)
                    self.log(f"Interval changed to {self.stream_interval}s")
                
                # Create response
                response = {
                    "command": command,
                    "processed_at": time.time(),
                    "success": True
                }
                
                self.atomic_write_json(self.response_file, response)
                    
            except Exception as e:
                self.log(f"Command processing error: {e}", "error")
    
    def service_loop(self):
        """Main service loop."""
        self.log("Service loop started")
        last_frame_time = 0
        
        while self.running:
            try:
                # Process any commands
                self.process_commands()
                
                # Check for Claude messages
                claude_msg = self.check_claude_message()
                if claude_msg:
                    self.handle_claude_message(claude_msg)
                
                # Streaming logic
                if self.streaming:
                    current_time = time.time()
                    
                    # Time for next frame?
                    if current_time - last_frame_time >= self.stream_interval:
                        # Check if Claude viewed the previous image
                        viewed_frame_id = self.check_claude_viewed()
                        if viewed_frame_id:
                            if self.archive_current_view_with_id(viewed_frame_id):
                                self.log(f"Image archived after Claude viewed (frame: {viewed_frame_id})")
                            else:
                                self.log("Image archive skipped - frame ID mismatch or error", "warning")
                        
                        # Capture new frame
                        if self.capture_frame():
                            last_frame_time = current_time
                            self.log(f"Frame {self.frame_count} captured")
                        else:
                            self.log("Frame capture failed", "error")
                
                # Update service status
                self.update_service_status()
                
                # Brief sleep to prevent excessive CPU usage
                time.sleep(0.5)
                
            except Exception as e:
                self.log(f"Service loop error: {e}", "error")
                time.sleep(1)
        
        self.log("Service loop ended")
    
    def cleanup_archives(self):
        """Clean up old archive files based on retention policy."""
        try:
            # Get all archive files (PNG and JSON)
            archive_files = list(self.archive_folder.glob("view_*.png")) + list(self.archive_folder.glob("view_*_info.json"))
            
            if not archive_files:
                return
            
            # Sort by modification time (oldest first)
            archive_files.sort(key=lambda f: f.stat().st_mtime)
            
            current_time = time.time()
            files_deleted = 0
            
            # Age-based cleanup: remove files older than max_archive_age_days
            max_age_seconds = self.max_archive_age_days * 24 * 3600
            for file_path in archive_files[:]:
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                        archive_files.remove(file_path)
                        files_deleted += 1
                        self.log(f"Deleted old archive: {file_path.name} (age: {file_age/3600:.1f}h)")
                    except Exception as e:
                        self.log(f"Failed to delete {file_path.name}: {e}", "error")
            
            # Count-based cleanup: remove oldest files if we exceed max_archive_files
            # Only count PNG files for the file limit (info files are paired)
            png_files = [f for f in archive_files if f.suffix == '.png']
            if len(png_files) > self.max_archive_files:
                files_to_remove = len(png_files) - self.max_archive_files
                
                # Remove oldest PNG files and their corresponding info files
                for i in range(files_to_remove):
                    png_file = png_files[i]
                    info_file = png_file.with_name(png_file.stem + '_info.json')
                    
                    try:
                        png_file.unlink()
                        files_deleted += 1
                        self.log(f"Deleted excess archive: {png_file.name}")
                        
                        if info_file.exists():
                            info_file.unlink()
                            self.log(f"Deleted excess archive info: {info_file.name}")
                    except Exception as e:
                        self.log(f"Failed to delete {png_file.name}: {e}", "error")
            
            if files_deleted > 0:
                self.log(f"Archive cleanup completed: {files_deleted} files deleted")
                
        except Exception as e:
            self.log(f"Archive cleanup failed: {e}", "error")
    
    def start_service(self):
        """Start the background service."""
        if self.running:
            self.log("Service already running", "warning")
            return False
        
        self.running = True
        self.start_time = time.time()
        
        # Create ready signal for Claude
        with open(self.claude_ready, 'w') as f:
            f.write(f"Service started at {datetime.now().isoformat()}")
        
        # Start service thread
        self.service_thread = threading.Thread(target=self.service_loop, daemon=True)
        self.service_thread.start()
        
        self.log("Claude Vision Service started")
        return True
    
    def stop_service(self):
        """Stop the background service."""
        self.running = False
        self.streaming = False
        
        if self.service_thread and self.service_thread.is_alive():
            self.service_thread.join(timeout=5.0)
        
        # Archive final image if needed
        viewed_frame_id = self.check_claude_viewed()
        if self.current_view.exists() and viewed_frame_id:
            self.archive_current_view_with_id(viewed_frame_id)
        
        # Update final status
        self.update_service_status()
        
        self.log("Claude Vision Service stopped")
        return True
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status."""
        return {
            "running": self.running,
            "streaming": self.streaming,
            "frame_count": self.frame_count,
            "interval": self.stream_interval,
            "uptime": time.time() - getattr(self, 'start_time', time.time()),
            "current_view_exists": self.current_view.exists(),
            "archive_count": len(list(self.archive_folder.glob("view_*.png"))),
            "claude_ready": self.claude_ready.exists(),
            "awaiting_claude_view": not self.claude_viewed.exists(),
            "current_frame_id": self.current_frame_id,
            "frame_sequence": self.frame_sequence
        }


def main():
    """Main service control function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Claude Vision Background Service")
    
    parser.add_argument('command', choices=['start', 'stop', 'status', 'daemon'], 
                       help='Service command')
    
    parser.add_argument('--session', '-s', default="claude_session",
                       help='Session folder path')
    
    parser.add_argument('--interval', '-i', type=float, default=4.0,
                       help='Stream interval in seconds')
    
    parser.add_argument('--detach', '-d', action='store_true',
                       help='Run as detached background process')
    
    parser.add_argument('--max-size', type=int, default=1024,
                       help='Maximum file size in KB')
    
    parser.add_argument('--auto-optimize', action='store_true', default=True,
                       help='Enable auto format optimization')
    
    args = parser.parse_args()
    
    service = ClaudeVisionService(args.session)
    
    if args.command == 'start':
        if args.detach:
            # Run as detached background process
            print("🚀 Starting Claude Vision Service in background...")
            
            # Create a script to run the service
            service_script = Path("run_claude_service.py")
            with open(service_script, 'w') as f:
                f.write(f"""
import sys
sys.path.append('.')
from claude_service import ClaudeVisionService

service = ClaudeVisionService("{args.session}")
service.start_service()
service.streaming = True
service.stream_interval = {args.interval}
service.max_file_size_kb = {getattr(args, 'max_size', 1024)}
service.auto_optimize_format = {getattr(args, 'auto_optimize', True)}

try:
    while service.running:
        import time
        time.sleep(1)
except KeyboardInterrupt:
    service.stop_service()
""")
            
            # Start the service in background
            subprocess.Popen([sys.executable, str(service_script)], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL,
                           creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
            
            print("✅ Service started in background")
            print(f"📁 Session folder: {service.session_folder}")
            print(f"👁️ Current view: {service.current_view}")
            print("🔧 Use 'python claude_service.py status' to check status")
        else:
            # Run in foreground
            service.start_service()
            service.streaming = True
            service.stream_interval = args.interval
            
            print("🚀 Claude Vision Service started (foreground mode)")
            print("📋 Claude Instructions:")
            print(f"   - Watch: {service.current_view}")
            print(f"   - After viewing, create: {service.claude_viewed}")
            print(f"   - To message service, write to: {service.claude_message}")
            print("   - Press Ctrl+C to stop")
            
            try:
                while service.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n⏸️ Stopping service...")
                service.stop_service()
    
    elif args.command == 'stop':
        # Send stop command via file
        command = {"type": "stop_service", "timestamp": time.time()}
        
        with open(service.command_file, 'w') as f:
            json.dump(command, f, indent=2)
        
        print("🛑 Stop command sent to service")
    
    elif args.command == 'status':
        if service.service_status.exists():
            with open(service.service_status, 'r') as f:
                status = json.load(f)
            
            print("📊 Claude Vision Service Status")
            print("=" * 40)
            print(f"Running: {'🟢 Yes' if status.get('running') else '🔴 No'}")
            print(f"Streaming: {'🟢 Yes' if status.get('streaming') else '🔴 No'}")
            print(f"Frame count: {status.get('frame_count', 0)}")
            print(f"Interval: {status.get('stream_interval', 0)}s")
            print(f"Uptime: {status.get('uptime_seconds', 0):.0f}s")
            print(f"Current view: {status.get('current_view', 'N/A')}")
            print(f"PID: {status.get('pid', 'N/A')}")
            
            # Image settings info
            img_settings = status.get('image_settings', {})
            if img_settings:
                print(f"\n📷 Image Settings:")
                print(f"  Format: {img_settings.get('format', 'N/A')}")
                print(f"  Quality: {img_settings.get('quality', 'N/A')}")
                print(f"  Max size: {img_settings.get('max_file_size_kb', 'N/A')}KB")
                print(f"  Auto optimize: {'Yes' if img_settings.get('auto_optimize') else 'No'}")
        else:
            print("❌ Service not running (no status file found)")


if __name__ == "__main__":
    main()