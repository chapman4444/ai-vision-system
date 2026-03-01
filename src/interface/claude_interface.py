"""
Claude Code Interface for AI Vision System
Creates seamless interaction between Claude and the screen capture system.
"""

import os
import json
import time
import threading
import queue
from typing import Optional, Dict, Any, List
from datetime import datetime
import shutil
from pathlib import Path
import tempfile

from session.ai_session_manager import AISessionManager


class ClaudeInterface:
    """Interface for Claude Code to interact with the vision system."""
    
    def __init__(self, workspace_path: str):
        """
        Initialize Claude interface.
        
        Args:
            workspace_path: Path to Claude Code workspace for image sharing
        """
        self.workspace_path = Path(workspace_path)
        self.session_manager: Optional[AISessionManager] = None
        self.interface_dir = Path("ai_vision_interface")
        self.commands_dir = self.interface_dir / "commands"
        self.responses_dir = self.interface_dir / "responses"
        self.images_dir = self.interface_dir / "images"
        
        # Create interface directories
        for dir_path in [self.interface_dir, self.commands_dir, self.responses_dir, self.images_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Command processing
        self.command_processor = None
        self.processing_active = False
        
        # State tracking
        self.last_command_time = 0
        self.vision_active = False
        self.monitoring_mode = False
        
        self._create_interface_files()
    
    def atomic_write_json(self, path: Path, obj: Dict[str, Any]):
        """Atomically write JSON data to a file using temp-then-rename."""
        try:
            # Write to temporary file in the same directory
            temp_path = path.with_suffix(path.suffix + '.tmp')
            
            with open(temp_path, 'w') as f:
                json.dump(obj, f, indent=2)
                f.flush()  # Ensure data is written to disk
                os.fsync(f.fileno())  # Force write to storage
            
            # Atomic rename (Windows supports this for files in same directory)
            os.replace(temp_path, path)
            
        except Exception as e:
            # Cleanup temp file if it exists
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except:
                pass
            raise e
    
    def _create_interface_files(self):
        """Create the interface control files."""
        
        # Create command templates
        command_examples = {
            "start_vision": {
                "command": "start_monitoring",
                "parameters": {
                    "interval": 2.0,
                    "region": "primary_monitor"
                },
                "description": "Start continuous screen monitoring"
            },
            "stop_vision": {
                "command": "stop_monitoring", 
                "parameters": {},
                "description": "Stop screen monitoring"
            },
            "capture_now": {
                "command": "capture_now",
                "parameters": {
                    "save": True,
                    "copy_to_workspace": True
                },
                "description": "Take immediate screenshot"
            },
            "show_claude": {
                "command": "show_me_something",
                "parameters": {
                    "message": "User wants to show Claude something",
                    "copy_to_workspace": True
                },
                "description": "User wants to show Claude their screen"
            },
            "get_status": {
                "command": "get_status",
                "parameters": {},
                "description": "Get current vision system status"
            }
        }
        
        # Save command examples atomically
        self.atomic_write_json(self.interface_dir / "command_examples.json", command_examples)
        
        # Create README for Claude
        readme_content = """# AI Vision System - Claude Interface

This interface allows Claude Code to interact with the AI Vision System.

## Quick Commands

### For Claude to See Your Screen:
1. Run: `python start_vision_interface.py`
2. Create a command file to capture: `echo '{"command": "show_claude", "message": "Look at this!"}' > commands/show_claude.json`
3. Claude will receive the image in the workspace

### For Continuous Monitoring:
- Start: `echo '{"command": "start_vision", "interval": 1.0}' > commands/monitor.json`  
- Stop: `echo '{"command": "stop_vision"}' > commands/stop.json`

### Status Check:
- `echo '{"command": "get_status"}' > commands/status.json`

## Files:
- `commands/` - Drop JSON command files here
- `responses/` - System responses appear here  
- `images/` - Captured images for Claude
- `workspace_images/` - Images copied to Claude workspace

## Command Structure:
```json
{
    "command": "command_name",
    "parameters": {
        "key": "value"
    },
    "timestamp": 1234567890,
    "copy_to_workspace": true
}
```
"""
        
        with open(self.interface_dir / "README.md", "w") as f:
            f.write(readme_content)
        
        # Create status file
        self._update_status_file("Interface initialized")
    
    def start_interface(self) -> bool:
        """Start the Claude interface system."""
        try:
            # Start session manager
            self.session_manager = AISessionManager("claude_session")
            self.session_manager.start_session(self._handle_session_callback)
            
            # Start command processing
            self.processing_active = True
            self.command_processor = threading.Thread(target=self._process_command_files, daemon=True)
            self.command_processor.start()
            
            self.vision_active = True
            self._update_status_file("Vision interface active - Ready for Claude commands")
            
            print("🤖 Claude Interface Started!")
            print(f"📁 Interface directory: {self.interface_dir.absolute()}")
            print(f"📁 Workspace path: {self.workspace_path.absolute()}")
            print("\n📝 To interact:")
            print("   1. Drop command JSON files in the 'commands' folder")
            print("   2. Check 'responses' folder for results")
            print("   3. Images will be copied to workspace for Claude")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start interface: {e}")
            return False
    
    def stop_interface(self):
        """Stop the Claude interface system."""
        self.processing_active = False
        self.vision_active = False
        
        if self.session_manager:
            self.session_manager.stop_session()
        
        self._update_status_file("Interface stopped")
        print("🛑 Claude Interface Stopped")
    
    def _process_command_files(self):
        """Process command files dropped in the commands directory."""
        while self.processing_active:
            try:
                # Check for command files
                command_files = list(self.commands_dir.glob("*.json"))
                
                for command_file in command_files:
                    try:
                        # Read and parse command
                        with open(command_file, 'r') as f:
                            command_data = json.load(f)
                        
                        # Add timestamp if not present
                        if "timestamp" not in command_data:
                            command_data["timestamp"] = time.time()
                        
                        # Process command
                        self._execute_command(command_data, command_file.stem)
                        
                        # Move processed command to responses directory
                        processed_file = self.responses_dir / f"processed_{command_file.name}"
                        shutil.move(str(command_file), str(processed_file))
                        
                    except Exception as e:
                        print(f"❌ Error processing {command_file}: {e}")
                        # Move failed command
                        failed_file = self.responses_dir / f"failed_{command_file.name}"
                        try:
                            shutil.move(str(command_file), str(failed_file))
                        except:
                            pass
                
                # Sleep before next check
                time.sleep(0.5)
                
            except Exception as e:
                print(f"❌ Error in command processing loop: {e}")
                time.sleep(1)
    
    def _execute_command(self, command_data: Dict[str, Any], command_id: str):
        """Execute a command from Claude."""
        if not self.session_manager:
            return
        
        command = command_data.get("command")
        parameters = command_data.get("parameters", {})
        
        print(f"🎯 Executing command: {command} (ID: {command_id})")
        
        # Send command to session manager
        session_command_id = self.session_manager.send_command(command, parameters)
        
        # Wait for response
        response = self.session_manager.get_response(timeout=15.0)
        
        if response:
            # Create response file
            response_data = {
                "command_id": command_id,
                "session_command_id": session_command_id,
                "success": response.success,
                "message": response.message,
                "data": response.data,
                "timestamp": response.timestamp,
                "processed_at": time.time()
            }
            
            # Save response atomically
            response_file = self.responses_dir / f"response_{command_id}_{int(time.time())}.json"
            self.atomic_write_json(response_file, response_data)
            
            # Handle image copying if needed
            if response.success and response.data:
                self._handle_image_response(response.data, command_data, command_id)
            
            print(f"✅ Command completed: {response.message}")
            self._update_status_file(f"Last command: {command} - {response.message}")
            
        else:
            print(f"❌ Command timeout: {command}")
            # Create timeout response
            timeout_response = {
                "command_id": command_id,
                "success": False,
                "message": "Command timed out",
                "timestamp": time.time()
            }
            
            timeout_file = self.responses_dir / f"timeout_{command_id}_{int(time.time())}.json"
            self.atomic_write_json(timeout_file, timeout_response)
    
    def _handle_image_response(self, response_data: Dict[str, Any], 
                             command_data: Dict[str, Any], command_id: str):
        """Handle responses that contain images."""
        image_path = response_data.get("image_path")
        if not image_path or not os.path.exists(image_path):
            return
        
        # Copy to interface images directory
        image_filename = f"{command_id}_{int(time.time())}.png"
        interface_image_path = self.images_dir / image_filename
        shutil.copy2(image_path, interface_image_path)
        
        # Copy to Claude workspace if requested
        if command_data.get("copy_to_workspace", True) or command_data.get("parameters", {}).get("copy_to_workspace", True):
            self._copy_to_workspace(interface_image_path, command_id)
        
        print(f"📸 Image saved: {image_filename}")
    
    def _copy_to_workspace(self, image_path: Path, command_id: str):
        """Copy image to Claude Code workspace."""
        try:
            if not self.workspace_path.exists():
                self.workspace_path.mkdir(parents=True, exist_ok=True)
            
            # Create workspace filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            workspace_filename = f"vision_{command_id}_{timestamp}.png"
            workspace_image_path = self.workspace_path / workspace_filename
            
            # Copy image
            shutil.copy2(image_path, workspace_image_path)
            
            # Create info file for Claude
            info_data = {
                "image_file": workspace_filename,
                "command_id": command_id,
                "timestamp": time.time(),
                "datetime": datetime.now().isoformat(),
                "message": f"Vision system captured screen for command: {command_id}"
            }
            
            info_file = self.workspace_path / f"vision_{command_id}_{timestamp}_info.json"
            self.atomic_write_json(info_file, info_data)
            
            print(f"📋 Image copied to Claude workspace: {workspace_filename}")
            
        except Exception as e:
            print(f"❌ Failed to copy to workspace: {e}")
    
    def _handle_session_callback(self, event_type: str, data: Dict[str, Any]):
        """Handle callbacks from the session manager."""
        if event_type == "monitoring_capture":
            # Handle continuous monitoring
            if self.monitoring_mode:
                self._copy_monitoring_image(data)
        elif event_type == "user_showing_screen":
            # Handle user showing screen to Claude
            print(f"👁️ User showing screen: {data.get('ai_message', 'Screen capture')}")
    
    def _copy_monitoring_image(self, data: Dict[str, Any]):
        """Handle monitoring mode images."""
        image_path = data.get("image_path")
        if image_path and os.path.exists(image_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            workspace_filename = f"monitoring_{timestamp}.png"
            workspace_path = self.workspace_path / workspace_filename
            
            try:
                shutil.copy2(image_path, workspace_path)
                print(f"📊 Monitoring image: {workspace_filename}")
            except Exception as e:
                print(f"❌ Failed to copy monitoring image: {e}")
    
    def _update_status_file(self, message: str):
        """Update the status file for external monitoring."""
        status = {
            "vision_active": self.vision_active,
            "monitoring_mode": self.monitoring_mode,
            "last_update": time.time(),
            "datetime": datetime.now().isoformat(),
            "message": message,
            # Remove absolute paths - use relative/anonymized identifiers
            "interface_name": self.interface_dir.name,
            "workspace_name": self.workspace_path.name,
            "build_version": "1.0.0",
            "session_id": f"vision_{int(time.time() % 10000)}"
        }
        
        status_file = self.interface_dir / "status.json"
        self.atomic_write_json(status_file, status)
    
    def create_quick_command(self, command_name: str, **kwargs) -> Path:
        """Create a quick command file for immediate execution."""
        command_data = {
            "command": command_name,
            "parameters": kwargs,
            "timestamp": time.time(),
            "copy_to_workspace": True
        }
        
        command_file = self.commands_dir / f"quick_{command_name}_{int(time.time())}.json"
        self.atomic_write_json(command_file, command_data)
        
        print(f"📝 Created quick command: {command_file.name}")
        return command_file


def main():
    """Main function to run the Claude interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Claude AI Vision Interface")
    parser.add_argument("--workspace", "-w", default="claude_workspace", 
                       help="Path to Claude Code workspace")
    parser.add_argument("--command", "-c", help="Quick command to execute")
    parser.add_argument("--message", "-m", default="Quick capture", 
                       help="Message for quick commands")
    
    args = parser.parse_args()
    
    # Initialize interface
    interface = ClaudeInterface(args.workspace)
    
    if args.command:
        # Execute quick command
        if not interface.start_interface():
            return 1
        
        try:
            if args.command == "capture":
                interface.create_quick_command("capture_now", message=args.message)
            elif args.command == "show":
                interface.create_quick_command("show_me_something", message=args.message)
            elif args.command == "status":
                interface.create_quick_command("get_status")
            elif args.command == "start":
                interface.create_quick_command("start_monitoring", interval=2.0)
            elif args.command == "stop":
                interface.create_quick_command("stop_monitoring")
            else:
                print(f"❌ Unknown quick command: {args.command}")
                return 1
            
            # Wait a bit for processing
            time.sleep(3)
            
        finally:
            interface.stop_interface()
    
    else:
        # Run interactive mode
        if not interface.start_interface():
            return 1
        
        try:
            print("\n🎮 Interactive Mode:")
            print("   Type 'capture' to take a screenshot")
            print("   Type 'show' to show Claude your screen")  
            print("   Type 'start' to begin monitoring")
            print("   Type 'stop' to stop monitoring")
            print("   Type 'status' to check system status")
            print("   Type 'quit' to exit")
            
            while True:
                user_input = input("\n> ").strip().lower()
                
                if user_input == "quit":
                    break
                elif user_input == "capture":
                    interface.create_quick_command("capture_now")
                elif user_input == "show":
                    message = input("Message for Claude (optional): ").strip()
                    interface.create_quick_command("show_me_something", 
                                                 message=message or "User showing screen")
                elif user_input == "start":
                    interface.create_quick_command("start_monitoring", interval=2.0)
                    interface.monitoring_mode = True
                elif user_input == "stop":
                    interface.create_quick_command("stop_monitoring")
                    interface.monitoring_mode = False
                elif user_input == "status":
                    interface.create_quick_command("get_status")
                else:
                    print("❓ Unknown command. Try: capture, show, start, stop, status, quit")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        
        finally:
            interface.stop_interface()
    
    return 0


if __name__ == "__main__":
    exit(main())