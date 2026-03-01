#!/usr/bin/env python3
"""
Silent Streaming Screen Capture Viewer with Auto-Service Start
Always-on-top viewer that automatically starts the vision service.
Uses .pyw extension to run without console window.
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import time
import threading
from pathlib import Path
import sys
import subprocess
import queue

# Add src to path for service imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the vision service
from capture.gdi_screen_capture import GDIScreenCapture
import json
from datetime import datetime
import random
import math


class CalibrationWindow:
    """Full-screen calibration for LLMs - starts immediately, no buttons needed"""
    
    def __init__(self, parent):
        self.parent = parent
        self.window = tk.Toplevel(parent)
        self.window.title("LLM Mouse Calibration")
        
        # Get primary monitor size
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        
        # Full screen setup
        self.window.geometry(f"{screen_width}x{screen_height}+0+0")
        self.window.configure(bg='black')
        self.window.attributes('-topmost', True)
        self.window.attributes('-fullscreen', True)
        self.window.overrideredirect(True)  # Remove window decorations
        
        # Calibration data
        self.targets = []
        self.current_target = 0
        self.results = []
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Create full-screen canvas
        self.canvas = tk.Canvas(self.window, bg='black', highlightthickness=0, 
                               width=screen_width, height=screen_height)
        self.canvas.pack(fill='both', expand=True)
        
        # Bind events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.window.bind("<Escape>", self.cancel_calibration)
        self.canvas.focus_set()  # Enable keyboard focus
        
        # Start calibration immediately
        self.start_calibration()
        
    def start_calibration(self):
        """Start the calibration process - full screen 9-point grid"""
        self.targets.clear()
        self.results.clear()
        self.current_target = 0
        
        # Create 9 targets in a 3x3 grid using full screen
        margin = 150  # Larger margin for full screen
        
        for row in range(3):
            for col in range(3):
                x = margin + col * (self.screen_width - 2*margin) // 2
                y = margin + row * (self.screen_height - 2*margin) // 2
                self.targets.append({'x': x, 'y': y, 'clicked': False})
        
        self.show_next_target()
        
    def show_next_target(self):
        """Show the next calibration target"""
        self.canvas.delete("all")
        
        if self.current_target < len(self.targets):
            target = self.targets[self.current_target]
            
            # Draw larger target for full screen (red circle with white center)
            self.canvas.create_oval(target['x']-40, target['y']-40, target['x']+40, target['y']+40, 
                                   fill='red', outline='white', width=4)
            self.canvas.create_oval(target['x']-8, target['y']-8, target['x']+8, target['y']+8, 
                                   fill='white', outline='black', width=2)
            
            # Show completed targets as green dots
            for i, prev_target in enumerate(self.targets[:self.current_target]):
                self.canvas.create_oval(prev_target['x']-20, prev_target['y']-20, 
                                       prev_target['x']+20, prev_target['y']+20, 
                                       fill='green', outline='white', width=3)
            
            # Show target number in top-left corner
            self.canvas.create_text(50, 50, text=f"Target {self.current_target + 1}/9", 
                                   fill='white', font=('Arial', 20), anchor='nw')
            
            # Show ESC hint in top-right corner  
            self.canvas.create_text(self.screen_width - 50, 50, text="Press ESC to cancel", 
                                   fill='gray', font=('Arial', 16), anchor='ne')
        else:
            # All targets completed - save and close
            self.save_results()
            self.close_window()
    
    def on_canvas_click(self, event):
        """Handle canvas click"""
        if self.current_target < len(self.targets):
            target = self.targets[self.current_target]
            click_x, click_y = event.x, event.y
            
            # Calculate accuracy
            distance = math.sqrt((click_x - target['x'])**2 + (click_y - target['y'])**2)
            accuracy = max(0, 100 - distance)
            
            # Store result
            result = {
                'target': self.current_target + 1,
                'target_x': target['x'],
                'target_y': target['y'], 
                'click_x': click_x,
                'click_y': click_y,
                'distance': distance,
                'accuracy': accuracy
            }
            self.results.append(result)
            
            # Mark target as clicked
            target['clicked'] = True
            self.current_target += 1
            
            if self.current_target < len(self.targets):
                self.show_next_target()
            else:
                # All targets completed - save and close
                self.save_results()
                self.close_window()
    
    def save_results(self):
        """Save calibration results to viewer directory"""
        try:
            results_data = {
                'timestamp': datetime.now().isoformat(),
                'screen_size': {'width': self.screen_width, 'height': self.screen_height},
                'results': self.results,
                'summary': {
                    'total_targets': len(self.results),
                    'avg_accuracy': sum(r['accuracy'] for r in self.results) / len(self.results) if self.results else 0,
                    'avg_distance': sum(r['distance'] for r in self.results) / len(self.results) if self.results else 0
                }
            }
            
            # Save to viewer directory (same as viewer.pyw location)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"calibration_results_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(results_data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving calibration results: {e}")
    
    def cancel_calibration(self, event=None):
        """Cancel calibration (ESC key pressed)"""
        self.close_window()
    
    def close_window(self):
        """Close calibration window"""
        self.window.destroy()


class StreamViewer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI Vision Stream")
        self.root.attributes('-topmost', True)  # Always on top
        self.root.configure(bg='black')
        
        # Hide from taskbar (optional)
        self.root.attributes('-toolwindow', True)
        
        # Initialize vision service
        self.capture = GDIScreenCapture()
        self.service_running = True
        self.frame_count = 0
        
        # Paths
        self.session_folder = Path("claude_session")
        self.session_folder.mkdir(exist_ok=True)
        (self.session_folder / "service").mkdir(exist_ok=True)
        
        # Stream image path
        self.image_path = Path("claude_session/current_view.png")
        
        # Image display
        self.image_label = tk.Label(self.root, bg='black', text="Waiting for stream...", 
                                   fg='white', font=('Arial', 12))
        self.image_label.pack(expand=True, fill='both')
        
        # Status bar
        self.status_frame = tk.Frame(self.root, bg='#404040', height=22)
        self.status_frame.pack(fill='x', side='bottom')
        self.status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(self.status_frame, text="Waiting for stream", 
                                    bg='#404040', fg='white', font=('Arial', 8))
        self.status_label.pack(side='left', padx=5, pady=2)
        
        # Calibrate button
        self.calibrate_btn = tk.Button(self.status_frame, text="Calibrate", 
                                      command=self.open_calibration,
                                      bg='#660066', fg='white', bd=1, font=('Arial', 8, 'bold'),
                                      height=1)
        self.calibrate_btn.pack(side='right', padx=2)
        
        # Tell Claude button
        self.claude_btn = tk.Button(self.status_frame, text="Tell Claude", 
                                   command=self.tell_claude,
                                   bg='#006600', fg='white', bd=1, font=('Arial', 8, 'bold'),
                                   height=1)
        self.claude_btn.pack(side='right', padx=2)
        
        # Close button (small X in top right of status bar)
        self.close_btn = tk.Button(self.status_frame, text="×", command=self.on_closing,
                                  bg='#404040', fg='white', bd=0, font=('Arial', 10, 'bold'),
                                  width=2, height=1)
        self.close_btn.pack(side='right', padx=2)
        
        # Control variables
        self.running = True
        self.last_modified = 0
        self.current_image = None
        
        # Thread-safe communication
        self.gui_queue = queue.Queue()
        self.gui_lock = threading.Lock()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self.monitor_image, daemon=True)
        self.monitor_thread.start()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Start GUI update processor
        self.process_gui_updates()
        
    def monitor_image(self):
        """Monitor the stream image file for changes."""
        while self.running:
            try:
                if self.image_path.exists():
                    # Check if file has been modified
                    modified_time = self.image_path.stat().st_mtime
                    
                    if modified_time != self.last_modified:
                        self.last_modified = modified_time
                        self.load_and_display_image()
                        
                else:
                    # File doesn't exist yet
                    self.queue_gui_update('status', {'message': "Waiting for stream..."})
                    
            except Exception as e:
                self.queue_gui_update('status', {'message': f"Error: {str(e)[:30]}"})
                
            time.sleep(0.5)  # Check every 500ms
            
    def load_and_display_image(self):
        """Load and display the stream image."""
        try:
            # Load image
            pil_image = Image.open(self.image_path)
            
            # Calculate dimensions - 800px width, maintain aspect ratio
            original_width, original_height = pil_image.size
            target_width = 800
            aspect_ratio = original_height / original_width
            target_height = int(target_width * aspect_ratio)
            
            # Resize image
            resized_image = pil_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.current_image = ImageTk.PhotoImage(resized_image)
            
            # Queue display update for main thread
            timestamp = time.strftime("%H:%M:%S")
            status_msg = f"Updated {timestamp} | {original_width}×{original_height} → {target_width}×{target_height}"
            
            self.queue_gui_update('display', {
                'image': self.current_image,
                'width': target_width, 
                'height': target_height,
                'status': status_msg
            })
            
        except Exception as e:
            self.update_status(f"Error: {str(e)[:30]}")
            
    def update_display(self, width, height):
        """Update the image display in the main thread."""
        if self.current_image:
            self.image_label.configure(image=self.current_image, text="")
            
            # Resize window to fit image + status bar
            window_height = height + 22  # Add status bar height
            self.root.geometry(f"{width}x{window_height}")
            
    def queue_gui_update(self, update_type, data):
        """Thread-safe method to queue GUI updates."""
        try:
            self.gui_queue.put((update_type, data), block=False)
        except queue.Full:
            pass  # Skip update if queue is full
    
    def process_gui_updates(self):
        """Process queued GUI updates in the main thread."""
        try:
            while True:
                update_type, data = self.gui_queue.get_nowait()
                
                if update_type == 'status':
                    self.status_label.configure(text=data['message'])
                    
                elif update_type == 'display':
                    if data['image']:
                        self.image_label.configure(image=data['image'], text="")
                        # Resize window to fit image + status bar
                        window_height = data['height'] + 22
                        self.root.geometry(f"{data['width']}x{window_height}")
                    
                    if 'status' in data:
                        self.status_label.configure(text=data['status'])
                        
        except queue.Empty:
            pass
        
        # Schedule next update check
        self.root.after(50, self.process_gui_updates)  # Check every 50ms
    
    def update_status(self, message):
        """Thread-safe status update."""
        self.queue_gui_update('status', {'message': message})
    
    def tell_claude(self):
        """Send current_view.png command to active command prompt."""
        try:
            # Get absolute path to current_view.png
            image_path = Path("E:/Devel/ai-vision-system/claude_session/current_view.png").resolve()
            
            # Create the claude command
            claude_command = f"claude look at {image_path}"
            
            # Use PowerShell to send to active window
            # First, get the active window, then send keystrokes
            powershell_script = f'''
            # Get the current active window
            Add-Type -TypeDefinition @"
                using System;
                using System.Runtime.InteropServices;
                public class Win32 {{
                    [DllImport("user32.dll")]
                    public static extern IntPtr GetForegroundWindow();
                    [DllImport("user32.dll")]
                    public static extern int GetWindowText(IntPtr hWnd, System.Text.StringBuilder text, int count);
                    [DllImport("user32.dll")]
                    public static extern bool SetForegroundWindow(IntPtr hWnd);
                }}
"@

            # Find command prompt or PowerShell windows
            $processes = Get-Process | Where-Object {{ $_.MainWindowTitle -match "(Command Prompt|PowerShell|Windows Terminal)" }}
            
            if ($processes.Count -gt 0) {{
                $targetWindow = $processes[0].MainWindowHandle
                [Win32]::SetForegroundWindow($targetWindow)
                Start-Sleep -Milliseconds 200
                
                # Send the command
                Add-Type -AssemblyName System.Windows.Forms
                [System.Windows.Forms.SendKeys]::SendWait("{claude_command}{{ENTER}}")
            }} else {{
                # No command prompt found, try to send to any active window
                Add-Type -AssemblyName System.Windows.Forms
                [System.Windows.Forms.SendKeys]::SendWait("{claude_command}{{ENTER}}")
            }}
            '''
            
            # Execute PowerShell script
            subprocess.run([
                "powershell", "-WindowStyle", "Hidden", "-Command", powershell_script
            ], check=False, capture_output=True)
            
            # Update status
            self.update_status("Sent to Claude!")
            
            # Reset status after 2 seconds
            self.root.after(2000, lambda: self.update_status("Stream active"))
            
        except Exception as e:
            self.update_status(f"Error: {str(e)[:30]}")
            # Reset status after 3 seconds
            self.root.after(3000, lambda: self.update_status("Stream active"))
        
    def capture_frame(self):
        """Capture current screen frame with cursor"""
        try:
            # Capture screen with cursor
            screen_data = self.capture.capture_primary_monitor(include_cursor=True)
            
            # Save capture
            self.capture.save_capture(screen_data, str(self.image_path))
            
            # Update info
            info_data = {
                "capture_time": datetime.now().isoformat(),
                "frame_count": self.frame_count,
                "image_path": str(self.image_path),
                "status": "active"
            }
            
            info_path = self.session_folder / "current_view_info.json"
            with open(info_path, 'w') as f:
                json.dump(info_data, f, indent=2)
            
            self.frame_count += 1
            return True
            
        except Exception as e:
            print(f"Capture failed: {e}")
            return False
    
    def service_loop(self):
        """Background service loop with change detection"""
        last_hash = None
        last_capture_time = 0
        min_interval = 1.0  # Minimum 1 second between captures
        max_interval = 5.0  # Maximum 5 seconds even if no changes
        
        while self.service_running:
            current_time = time.time()
            
            # Quick screen hash to detect changes
            try:
                # Capture a small sample for change detection
                sample_data = self.capture.capture_screen_region(0, 0, 200, 200, include_cursor=False)
                current_hash = hash(sample_data.tobytes())
                
                # Check if screen changed or max interval exceeded
                screen_changed = (last_hash is None or current_hash != last_hash)
                max_time_exceeded = (current_time - last_capture_time) > max_interval
                min_time_passed = (current_time - last_capture_time) > min_interval
                
                if (screen_changed or max_time_exceeded) and min_time_passed:
                    self.capture_frame()
                    last_hash = current_hash
                    last_capture_time = current_time
                
                time.sleep(0.5)  # Check every 500ms for changes
                
            except Exception as e:
                # Fallback to time-based capture if change detection fails
                self.capture_frame()
                time.sleep(5)  # Fallback interval
    
    def start_service(self):
        """Start the background capture service"""
        service_thread = threading.Thread(target=self.service_loop, daemon=True)
        service_thread.start()
    
    def open_calibration(self):
        """Open calibration window"""
        calibration_window = CalibrationWindow(self.root)
    
    def on_closing(self):
        """Handle window closing."""
        self.service_running = False  # Stop service
        self.running = False
        self.root.quit()
        self.root.destroy()
        
    def run(self):
        """Start the viewer and integrated service."""
        # Start the background capture service
        self.start_service()
        
        # Set initial window size
        self.root.geometry("800x472")  # 800x450 + 22 for status bar
        
        # Center window on screen
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (400)  # Center horizontally
        y = 50  # Near top of screen
        self.root.geometry(f"+{x}+{y}")
        
        # Update title to show it's integrated
        self.root.title("AI Vision Stream (Integrated Service)")
        
        self.root.mainloop()


if __name__ == "__main__":
    viewer = StreamViewer()
    viewer.run()