#!/usr/bin/env python3
"""
Start Vision Interface for Claude Code
Quick launcher for the AI Vision System interface.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from interface.claude_interface import ClaudeInterface


def main():
    """Main launcher function."""
    print("🤖 AI Vision System - Claude Interface Launcher")
    print("=" * 50)
    
    parser = argparse.ArgumentParser(
        description="Start AI Vision Interface for Claude Code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with default workspace
  python start_vision_interface.py
  
  # Specify custom workspace
  python start_vision_interface.py --workspace /path/to/claude/workspace
  
  # Quick capture
  python start_vision_interface.py --quick capture
  
  # Show Claude your screen
  python start_vision_interface.py --quick show --message "Look at this cool feature!"
  
  # Start monitoring mode
  python start_vision_interface.py --quick start
        """
    )
    
    parser.add_argument(
        "--workspace", "-w", 
        default="claude_workspace",
        help="Path to Claude Code workspace directory (default: claude_workspace)"
    )
    
    parser.add_argument(
        "--quick", "-q",
        choices=["capture", "show", "start", "stop", "status"],
        help="Execute a quick command and exit"
    )
    
    parser.add_argument(
        "--message", "-m",
        default="Quick capture from vision interface",
        help="Message for capture commands"
    )
    
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=2.0,
        help="Monitoring interval in seconds (default: 2.0)"
    )
    
    args = parser.parse_args()
    
    # Validate workspace path
    workspace_path = Path(args.workspace).resolve()
    
    print(f"📁 Workspace: {workspace_path}")
    print(f"🔧 Interface directory will be: {Path.cwd() / 'ai_vision_interface'}")
    
    # Create workspace if it doesn't exist
    try:
        workspace_path.mkdir(parents=True, exist_ok=True)
        print("✅ Workspace ready")
    except Exception as e:
        print(f"❌ Failed to create workspace: {e}")
        return 1
    
    # Initialize interface
    try:
        interface = ClaudeInterface(str(workspace_path))
    except Exception as e:
        print(f"❌ Failed to initialize interface: {e}")
        return 1
    
    # Execute quick command or start interactive mode
    if args.quick:
        print(f"🚀 Executing quick command: {args.quick}")
        
        if not interface.start_interface():
            return 1
        
        try:
            if args.quick == "capture":
                interface.create_quick_command("capture_now", message=args.message)
                print("📸 Screenshot command queued - check workspace for image!")
                
            elif args.quick == "show":
                interface.create_quick_command("show_me_something", message=args.message)
                print("👁️ 'Show Claude' command queued - capturing your screen!")
                
            elif args.quick == "start":
                interface.create_quick_command("start_monitoring", interval=args.interval)
                interface.monitoring_mode = True
                print(f"▶️ Monitoring started - capturing every {args.interval}s")
                print("   Images will appear in your workspace")
                print("   Run with --quick stop to end monitoring")
                
            elif args.quick == "stop":
                interface.create_quick_command("stop_monitoring")
                print("⏹️ Stop monitoring command queued")
                
            elif args.quick == "status":
                interface.create_quick_command("get_status")
                print("ℹ️ Status check queued - check responses folder")
            
            # Wait for command processing
            import time
            print("⏳ Processing command...")
            time.sleep(5)
            
            print("✅ Command processed!")
            print(f"📂 Check these locations:")
            print(f"   - Workspace images: {workspace_path}")
            print(f"   - Interface responses: {Path.cwd() / 'ai_vision_interface' / 'responses'}")
            
        except KeyboardInterrupt:
            print("\n⏸️ Interrupted by user")
        except Exception as e:
            print(f"❌ Error executing quick command: {e}")
            return 1
        finally:
            interface.stop_interface()
    
    else:
        # Interactive mode
        print("\n🎮 Starting Interactive Mode...")
        print("   This will monitor the 'commands' folder for JSON files")
        print("   Images will be automatically copied to your workspace")
        
        if not interface.start_interface():
            return 1
        
        try:
            # Display interface instructions
            interface_dir = Path.cwd() / "ai_vision_interface"
            print(f"\n📋 Interface Ready!")
            print(f"   Commands folder: {interface_dir / 'commands'}")
            print(f"   Responses folder: {interface_dir / 'responses'}")
            print(f"   Workspace folder: {workspace_path}")
            
            print("\n💡 Quick Actions:")
            print("   📸 Capture Now:")
            print(f"      echo '{{\"command\": \"capture_now\"}}' > \"{interface_dir / 'commands' / 'capture.json'}\"")
            
            print("\n   👁️ Show Claude Screen:")
            print(f"      echo '{{\"command\": \"show_me_something\", \"parameters\": {{\"message\": \"Look at this!\"}}}}' > \"{interface_dir / 'commands' / 'show.json'}\"")
            
            print("\n   ▶️ Start Monitoring:")
            print(f"      echo '{{\"command\": \"start_monitoring\", \"parameters\": {{\"interval\": 2.0}}}}' > \"{interface_dir / 'commands' / 'monitor.json'}\"")
            
            print("\n   ⏹️ Stop Monitoring:")
            print(f"      echo '{{\"command\": \"stop_monitoring\"}}' > \"{interface_dir / 'commands' / 'stop.json'}\"")
            
            print("\n🔄 The system is now running and monitoring for commands...")
            print("   Press Ctrl+C to stop")
            
            # Keep running until interrupted
            while True:
                import time
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n👋 Shutting down interface...")
        except Exception as e:
            print(f"❌ Error in interactive mode: {e}")
            return 1
        finally:
            interface.stop_interface()
    
    print("✅ Vision Interface stopped successfully")
    return 0


if __name__ == "__main__":
    exit(main())