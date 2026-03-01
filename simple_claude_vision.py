#!/usr/bin/env python3
"""
Simple Claude Vision Interface
A working, simplified version for immediate use.
"""

import sys
import os
import time
import json
import shutil
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from capture.simple_screen_capture import SimpleScreenCapture


class SimpleClaude:
    """Simple interface for Claude to see your screen."""
    
    def __init__(self, workspace_path: str = "claude_workspace"):
        self.capture = SimpleScreenCapture()
        self.workspace = Path(workspace_path)
        self.workspace.mkdir(exist_ok=True)
        
        print(f"📁 Claude workspace: {self.workspace.absolute()}")
    
    def capture_for_claude(self, message: str = "Screen capture") -> str:
        """Capture screen and save for Claude."""
        print(f"📸 Capturing screen: {message}")
        
        try:
            # Capture full screen
            screen_data = self.capture.capture_primary_monitor()
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"claude_vision_{timestamp}.png"
            filepath = self.workspace / filename
            
            # Save image
            self.capture.save_capture(screen_data, str(filepath))
            
            # Create info file for Claude
            info_data = {
                "image_file": filename,
                "message": message,
                "timestamp": time.time(),
                "datetime": datetime.now().isoformat(),
                "screen_size": f"{screen_data.shape[1]}x{screen_data.shape[0]}",
                "captured_region": "full_screen"
            }
            
            info_file = self.workspace / f"claude_vision_{timestamp}_info.json"
            with open(info_file, 'w') as f:
                json.dump(info_data, f, indent=2)
            
            print(f"✅ Image saved for Claude: {filename}")
            print(f"📋 Info file: {info_file.name}")
            
            return str(filepath)
            
        except Exception as e:
            print(f"❌ Capture failed: {e}")
            return ""
    
    def capture_region_for_claude(self, x: int, y: int, width: int, height: int, 
                                message: str = "Region capture") -> str:
        """Capture specific region and save for Claude."""
        print(f"📸 Capturing region {width}x{height} at ({x},{y}): {message}")
        
        try:
            # Capture region
            region_data = self.capture.capture_screen_region(x, y, width, height)
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"claude_region_{timestamp}.png"
            filepath = self.workspace / filename
            
            # Save image
            self.capture.save_capture(region_data, str(filepath))
            
            # Create info file
            info_data = {
                "image_file": filename,
                "message": message,
                "timestamp": time.time(),
                "datetime": datetime.now().isoformat(),
                "region_size": f"{width}x{height}",
                "region_position": f"({x},{y})",
                "captured_region": "custom_region"
            }
            
            info_file = self.workspace / f"claude_region_{timestamp}_info.json"
            with open(info_file, 'w') as f:
                json.dump(info_data, f, indent=2)
            
            print(f"✅ Region saved for Claude: {filename}")
            return str(filepath)
            
        except Exception as e:
            print(f"❌ Region capture failed: {e}")
            return ""
    
    def show_claude(self, message: str = "Look at this!") -> str:
        """Show Claude your current screen - optimized for showing something specific."""
        print(f"👁️ Showing Claude: {message}")
        return self.capture_for_claude(f"USER SHOWING: {message}")
    
    def monitor_for_claude(self, duration: float = 30.0, interval: float = 2.0):
        """Monitor screen for Claude for a specific duration."""
        print(f"▶️ Starting monitoring for {duration}s (every {interval}s)")
        
        start_time = time.time()
        capture_count = 0
        
        try:
            while (time.time() - start_time) < duration:
                capture_count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Capture screen
                screen_data = self.capture.capture_primary_monitor()
                filename = f"claude_monitor_{timestamp}_{capture_count:03d}.png"
                filepath = self.workspace / filename
                
                # Save with minimal info
                self.capture.save_capture(screen_data, str(filepath))
                print(f"📊 Monitor capture {capture_count}: {filename}")
                
                # Wait for next capture
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n⏹️ Monitoring stopped by user")
        
        print(f"✅ Monitoring complete: {capture_count} captures saved")


def main():
    """Main function with command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Claude Vision System")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Capture command
    capture_parser = subparsers.add_parser('capture', help='Capture screen for Claude')
    capture_parser.add_argument('--message', '-m', default='Screen capture', 
                               help='Message describing what to capture')
    
    # Show command  
    show_parser = subparsers.add_parser('show', help='Show Claude something on screen')
    show_parser.add_argument('message', nargs='?', default='Look at this!',
                            help='What you want to show Claude')
    
    # Region command
    region_parser = subparsers.add_parser('region', help='Capture specific region')
    region_parser.add_argument('x', type=int, help='Left position')
    region_parser.add_argument('y', type=int, help='Top position') 
    region_parser.add_argument('width', type=int, help='Width')
    region_parser.add_argument('height', type=int, help='Height')
    region_parser.add_argument('--message', '-m', default='Region capture',
                              help='Message for the region')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor screen for Claude')
    monitor_parser.add_argument('--duration', '-d', type=float, default=30.0,
                               help='Duration to monitor (seconds)')
    monitor_parser.add_argument('--interval', '-i', type=float, default=2.0, 
                               help='Capture interval (seconds)')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test the system')
    
    # Workspace argument for all commands
    parser.add_argument('--workspace', '-w', default='claude_workspace',
                       help='Claude workspace directory')
    
    args = parser.parse_args()
    
    if not args.command:
        # Interactive mode if no command given
        print("🤖 Simple Claude Vision - Interactive Mode")
        print("=" * 50)
        
        claude = SimpleClaude(args.workspace)
        
        while True:
            print("\n🎮 Commands:")
            print("  1. capture - Take screenshot for Claude")
            print("  2. show - Show Claude your screen")
            print("  3. monitor - Start monitoring mode")
            print("  4. test - Test capture")
            print("  5. quit - Exit")
            
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == '1' or choice.lower() == 'capture':
                message = input("Message (optional): ").strip() or "Screen capture"
                claude.capture_for_claude(message)
                
            elif choice == '2' or choice.lower() == 'show':
                message = input("What do you want to show Claude?: ").strip() or "Look at this!"
                claude.show_claude(message)
                
            elif choice == '3' or choice.lower() == 'monitor':
                duration = input("Duration in seconds (30): ").strip()
                duration = float(duration) if duration else 30.0
                claude.monitor_for_claude(duration)
                
            elif choice == '4' or choice.lower() == 'test':
                print("🧪 Testing screen capture...")
                width, height = claude.capture.get_screen_dimensions()
                print(f"Screen: {width}x{height}")
                
                test_data = claude.capture.capture_screen_region(0, 0, 400, 300)
                claude.capture.save_capture(test_data, "test_capture.png")
                print("✅ Test capture saved: test_capture.png")
                
            elif choice == '5' or choice.lower() == 'quit':
                break
                
            else:
                print("❓ Invalid choice")
        
        print("👋 Goodbye!")
        return
    
    # Command line mode
    claude = SimpleClaude(args.workspace)
    
    if args.command == 'capture':
        claude.capture_for_claude(args.message)
        
    elif args.command == 'show':
        claude.show_claude(args.message)
        
    elif args.command == 'region':
        claude.capture_region_for_claude(args.x, args.y, args.width, args.height, args.message)
        
    elif args.command == 'monitor':
        claude.monitor_for_claude(args.duration, args.interval)
        
    elif args.command == 'test':
        print("🧪 Testing Simple Claude Vision...")
        
        # Test screen dimensions
        width, height = claude.capture.get_screen_dimensions()
        print(f"✅ Screen: {width}x{height}")
        
        # Test capture
        test_file = claude.capture_for_claude("System test capture")
        if test_file:
            print("✅ Test capture successful!")
        else:
            print("❌ Test capture failed")


if __name__ == "__main__":
    main()