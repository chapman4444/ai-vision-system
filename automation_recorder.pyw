#!/usr/bin/env python3
"""
Automation Recorder - Floating Control Panel
Like Active Presenter but for UI automation workflows
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os
import time
import threading
from pathlib import Path
import sys
import json
from datetime import datetime
import math

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from capture.gdi_screen_capture import GDIScreenCapture


class FloatingControlPanel:
    """Active Presenter style floating control panel for automation recording"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Automation Recorder")
        self.root.geometry("300x80")
        self.root.configure(bg='#2b2b2b')
        self.root.attributes('-topmost', True)  # Always on top
        self.root.attributes('-toolwindow', True)  # Hide from taskbar
        self.root.overrideredirect(True)  # Remove window decorations
        
        # Recording state
        self.is_recording = False
        self.is_paused = False
        self.current_step = 0
        self.workflow_steps = []
        self.session_folder = None
        
        # Initialize capture system
        self.capture = GDIScreenCapture()
        
        # Create the floating panel UI
        self.create_floating_ui()
        
        # Make draggable
        self.make_draggable()
        
        # Position in top-right corner
        self.position_panel()
    
    def create_floating_ui(self):
        """Create the floating control panel UI"""
        # Main frame with rounded appearance
        main_frame = tk.Frame(self.root, bg='#2b2b2b', relief='raised', bd=1)
        main_frame.pack(fill='both', expand=True, padx=2, pady=2)
        
        # Title bar
        title_frame = tk.Frame(main_frame, bg='#404040', height=25)
        title_frame.pack(fill='x', pady=(0,5))
        title_frame.pack_propagate(False)
        
        # Title and close button
        title_label = tk.Label(title_frame, text="🎬 Automation Recorder", 
                              bg='#404040', fg='white', font=('Segoe UI', 9, 'bold'))
        title_label.pack(side='left', padx=10, pady=3)
        
        close_btn = tk.Button(title_frame, text="✕", command=self.close_panel,
                             bg='#404040', fg='white', bd=0, font=('Segoe UI', 8), width=3)
        close_btn.pack(side='right', padx=5)
        
        # Control buttons frame
        controls_frame = tk.Frame(main_frame, bg='#2b2b2b')
        controls_frame.pack(fill='x', padx=10, pady=5)
        
        # Record button
        self.record_btn = tk.Button(controls_frame, text="🔴 Record", 
                                   command=self.toggle_recording,
                                   bg='#dc3545', fg='white', font=('Segoe UI', 9, 'bold'),
                                   relief='flat', padx=15, pady=5)
        self.record_btn.pack(side='left', padx=2)
        
        # Pause button
        self.pause_btn = tk.Button(controls_frame, text="⏸️ Pause",
                                  command=self.toggle_pause,
                                  bg='#6c757d', fg='white', font=('Segoe UI', 9),
                                  relief='flat', padx=15, pady=5, state='disabled')
        self.pause_btn.pack(side='left', padx=2)
        
        # Stop button
        self.stop_btn = tk.Button(controls_frame, text="⏹️ Stop",
                                 command=self.stop_recording,
                                 bg='#6c757d', fg='white', font=('Segoe UI', 9),
                                 relief='flat', padx=15, pady=5, state='disabled')
        self.stop_btn.pack(side='left', padx=2)
        
        # Status frame
        status_frame = tk.Frame(main_frame, bg='#2b2b2b', height=20)
        status_frame.pack(fill='x', padx=10, pady=(0,5))
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(status_frame, text="Ready to record workflow", 
                                    bg='#2b2b2b', fg='#28a745', font=('Segoe UI', 8))
        self.status_label.pack(side='left')
        
        self.step_label = tk.Label(status_frame, text="Step: 0", 
                                  bg='#2b2b2b', fg='white', font=('Segoe UI', 8))
        self.step_label.pack(side='right')
    
    def make_draggable(self):
        """Make the panel draggable"""
        def start_move(event):
            self.x = event.x
            self.y = event.y
        
        def stop_move(event):
            self.x = None
            self.y = None
        
        def do_move(event):
            if hasattr(self, 'x') and self.x is not None:
                deltax = event.x - self.x
                deltay = event.y - self.y
                x = self.root.winfo_x() + deltax
                y = self.root.winfo_y() + deltay
                self.root.geometry(f"+{x}+{y}")
        
        # Bind to title frame for dragging
        title_widgets = [widget for widget in self.root.winfo_children()[0].winfo_children()[0].winfo_children()]
        for widget in title_widgets:
            if isinstance(widget, tk.Label):
                widget.bind("<Button-1>", start_move)
                widget.bind("<ButtonRelease-1>", stop_move)
                widget.bind("<B1-Motion>", do_move)
    
    def position_panel(self):
        """Position panel in top-right corner"""
        self.root.update_idletasks()
        screen_width = self.root.winfo_screenwidth()
        x = screen_width - 320  # 20px from right edge
        y = 50  # 50px from top
        self.root.geometry(f"+{x}+{y}")
    
    def toggle_recording(self):
        """Start/stop recording"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start recording automation workflow"""
        self.is_recording = True
        self.is_paused = False
        self.current_step = 0
        self.workflow_steps = []
        
        # Create session folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_folder = Path(f"automation_sessions/session_{timestamp}")
        self.session_folder.mkdir(parents=True, exist_ok=True)
        
        # Update UI
        self.record_btn.configure(text="🔴 Recording...", bg='#dc3545')
        self.pause_btn.configure(state='normal', bg='#ffc107')
        self.stop_btn.configure(state='normal', bg='#dc3545')
        self.status_label.configure(text="Recording workflow...", fg='#dc3545')
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self.recording_loop, daemon=True)
        self.recording_thread.start()
        
        messagebox.showinfo("Recording Started", 
                          f"Recording automation workflow...\n\n"
                          f"Session: {self.session_folder.name}\n"
                          f"Perform your actions - each screen change will be captured as a step.\n\n"
                          f"Click 'Stop' when finished.")
    
    def toggle_pause(self):
        """Pause/resume recording"""
        self.is_paused = not self.is_paused
        
        if self.is_paused:
            self.pause_btn.configure(text="▶️ Resume", bg='#28a745')
            self.status_label.configure(text="Recording paused...", fg='#ffc107')
        else:
            self.pause_btn.configure(text="⏸️ Pause", bg='#ffc107')
            self.status_label.configure(text="Recording workflow...", fg='#dc3545')
    
    def stop_recording(self):
        """Stop recording and save workflow"""
        self.is_recording = False
        self.is_paused = False
        
        # Update UI
        self.record_btn.configure(text="🔴 Record", bg='#dc3545')
        self.pause_btn.configure(state='disabled', bg='#6c757d')
        self.stop_btn.configure(state='disabled', bg='#6c757d')
        self.status_label.configure(text=f"Workflow saved! {len(self.workflow_steps)} steps", fg='#28a745')
        
        # Save workflow data
        self.save_workflow()
        
        # Show completion message
        messagebox.showinfo("Recording Complete", 
                          f"Automation workflow recorded!\n\n"
                          f"Steps captured: {len(self.workflow_steps)}\n"
                          f"Saved to: {self.session_folder}\n\n"
                          f"Ready to record next workflow.")
    
    def recording_loop(self):
        """Main recording loop - captures steps based on screen changes"""
        last_capture_time = 0
        step_interval = 2.0  # Capture every 2 seconds or on significant change
        
        while self.is_recording:
            if not self.is_paused:
                current_time = time.time()
                
                if (current_time - last_capture_time) >= step_interval:
                    self.capture_workflow_step()
                    last_capture_time = current_time
            
            time.sleep(0.5)
    
    def capture_workflow_step(self):
        """Capture a single workflow step"""
        try:
            self.current_step += 1
            
            # Capture screen with cursor
            screen_data = self.capture.capture_primary_monitor(include_cursor=True)
            
            # Save step image
            step_filename = f"step_{self.current_step:03d}.png"
            step_path = self.session_folder / step_filename
            self.capture.save_capture(screen_data, str(step_path))
            
            # Get cursor position for metadata
            import ctypes
            class POINT(ctypes.Structure):
                _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
            
            point = POINT()
            ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
            
            # Create step metadata
            step_data = {
                "step_number": self.current_step,
                "timestamp": datetime.now().isoformat(),
                "image_file": step_filename,
                "cursor_position": {"x": point.x, "y": point.y},
                "action_type": "screen_state",  # Can be enhanced later
                "description": f"Workflow step {self.current_step}",
                "annotations": []  # For future annotation system
            }
            
            self.workflow_steps.append(step_data)
            
            # Update UI
            self.step_label.configure(text=f"Step: {self.current_step}")
            
        except Exception as e:
            print(f"Step capture error: {e}")
    
    def save_workflow(self):
        """Save complete workflow data"""
        if not self.session_folder:
            return
        
        workflow_data = {
            "session_info": {
                "created": datetime.now().isoformat(),
                "total_steps": len(self.workflow_steps),
                "session_folder": str(self.session_folder),
                "recorder_version": "1.0"
            },
            "workflow_steps": self.workflow_steps,
            "metadata": {
                "screen_resolution": f"{self.capture.get_screen_dimensions()}",
                "capture_settings": {
                    "include_cursor": True,
                    "step_interval": 2.0
                }
            }
        }
        
        # Save workflow JSON
        workflow_file = self.session_folder / "workflow.json"
        with open(workflow_file, 'w') as f:
            json.dump(workflow_data, f, indent=2)
        
        print(f"Workflow saved: {workflow_file}")
    
    def close_panel(self):
        """Close the control panel"""
        if self.is_recording:
            response = messagebox.askyesno("Recording Active", 
                                         "Stop recording before closing?")
            if response:
                self.stop_recording()
            else:
                return
        
        self.root.quit()
        self.root.destroy()
    
    def run(self):
        """Start the floating control panel"""
        self.root.mainloop()


if __name__ == "__main__":
    panel = FloatingControlPanel()
    panel.run()