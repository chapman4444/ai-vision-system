#!/usr/bin/env python3
"""
Minimal AI Vision Viewer - Command Window Style
Clean, minimal interface with system menu and properties
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import os
import time
import threading
from pathlib import Path
import sys
import queue
import json
from datetime import datetime
import math

# Add src to path for service imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the vision service
from capture.gdi_screen_capture import GDIScreenCapture


class MinimalViewer:
    """Minimal command-window-style viewer"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI Vision")
        
        # Load/create settings
        self.settings_file = "viewer_settings.json"
        self.load_settings()
        
        # Initialize vision service
        self.capture = GDIScreenCapture()
        self.service_running = True
        self.frame_count = 0
        
        # Setup UI - minimal command window style
        self.setup_minimal_ui()
        
        # Image monitoring
        self.running = True
        self.last_modified = 0
        self.current_image = None
        
        # Thread-safe communication
        self.gui_queue = queue.Queue()
        
        # Start service and monitoring
        self.start_service()
        self.start_monitoring()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Process GUI updates
        self.process_gui_updates()
    
    def load_settings(self):
        """Load viewer settings"""
        default_settings = {
            "output_path": "claude_session/current_view.png",
            "archive_folder": "claude_session/archive",
            "monitor_target": "primary",  # primary, secondary, window, region
            "cursor_type": "cross",  # cross, cursor, none
            "cursor_size": 15,
            "cursor_thickness": 3,
            "window_width": 800,
            "window_height": 600
        }
        
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    self.settings = {**default_settings, **json.load(f)}
            else:
                self.settings = default_settings
                self.save_settings()
        except Exception:
            self.settings = default_settings
    
    def save_settings(self):
        """Save viewer settings"""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")
    
    def setup_minimal_ui(self):
        """Setup minimal command-window-style UI"""
        # Window setup
        width = self.settings["window_width"]
        height = self.settings["window_height"]
        self.root.geometry(f"{width}x{height}")
        self.root.configure(bg='black')
        
        # Remove default menu, we'll add our own
        self.root.config(menu=tk.Menu(self.root))
        
        # Create system menu (accessed via icon click)
        self.create_system_menu()
        
        # Title bar frame (minimal)
        self.title_frame = tk.Frame(self.root, bg='#2d2d30', height=30)
        self.title_frame.pack(fill='x', side='top')
        self.title_frame.pack_propagate(False)
        
        # System icon (clickable for menu)
        self.system_icon = tk.Button(self.title_frame, text="⚙", 
                                    command=self.show_system_menu,
                                    bg='#2d2d30', fg='white', bd=0, 
                                    font=('Segoe UI', 10), width=3)
        self.system_icon.pack(side='left', pady=5)
        
        # Title text
        self.title_label = tk.Label(self.title_frame, text="AI Vision Stream", 
                                   bg='#2d2d30', fg='white', font=('Segoe UI', 9))
        self.title_label.pack(side='left', padx=10, pady=7)
        
        # Window controls
        self.controls_frame = tk.Frame(self.title_frame, bg='#2d2d30')
        self.controls_frame.pack(side='right', padx=5)
        
        # Minimize button
        self.min_btn = tk.Button(self.controls_frame, text="─", command=self.minimize_window,
                                bg='#2d2d30', fg='white', bd=0, font=('Segoe UI', 8), width=3)
        self.min_btn.pack(side='left')
        
        # Close button
        self.close_btn = tk.Button(self.controls_frame, text="✕", command=self.on_closing,
                                  bg='#2d2d30', fg='white', bd=0, font=('Segoe UI', 8), width=3)
        self.close_btn.pack(side='left')
        
        # Main image area
        self.image_label = tk.Label(self.root, bg='black', text="Initializing vision stream...", 
                                   fg='white', font=('Consolas', 10))
        self.image_label.pack(expand=True, fill='both')
        
        # Status bar (minimal)
        self.status_frame = tk.Frame(self.root, bg='#007acc', height=20)
        self.status_frame.pack(fill='x', side='bottom')
        self.status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(self.status_frame, text="Ready", 
                                    bg='#007acc', fg='white', font=('Segoe UI', 8))
        self.status_label.pack(side='left', padx=5, pady=1)
        
        # Service status indicator
        self.service_status = tk.Label(self.status_frame, text="●", 
                                      bg='#007acc', fg='green', font=('Segoe UI', 10))
        self.service_status.pack(side='right', padx=5)
        
        # Make title bar draggable
        self.make_draggable(self.title_frame)
        self.make_draggable(self.title_label)
    
    def create_system_menu(self):
        """Create system menu (like right-click on taskbar icon)"""
        self.system_menu = tk.Menu(self.root, tearoff=0)
        
        # Service controls
        self.system_menu.add_command(label="Start Service", command=self.start_service_menu)
        self.system_menu.add_command(label="Stop Service", command=self.stop_service_menu)
        self.system_menu.add_command(label="Restart Service", command=self.restart_service)
        self.system_menu.add_separator()
        
        # System service controls
        self.system_menu.add_command(label="Install as Windows Service", command=self.install_service)
        self.system_menu.add_command(label="Remove Windows Service", command=self.remove_service)
        self.system_menu.add_separator()
        
        # Other options
        self.system_menu.add_command(label="Properties...", command=self.show_properties)
        self.system_menu.add_command(label="Calibrate", command=self.open_calibration)
        self.system_menu.add_separator()
        self.system_menu.add_command(label="Close", command=self.on_closing)
    
    def show_system_menu(self):
        """Show system menu when icon is clicked"""
        try:
            self.system_menu.post(self.system_icon.winfo_rootx(), 
                                 self.system_icon.winfo_rooty() + self.system_icon.winfo_height())
        except:
            pass
    
    def make_draggable(self, widget):
        """Make widget draggable for window movement"""
        def start_move(event):
            widget.x = event.x
            widget.y = event.y
        
        def stop_move(event):
            widget.x = None
            widget.y = None
        
        def do_move(event):
            if hasattr(widget, 'x') and widget.x is not None:
                deltax = event.x - widget.x
                deltay = event.y - widget.y
                x = self.root.winfo_x() + deltax
                y = self.root.winfo_y() + deltay
                self.root.geometry(f"+{x}+{y}")
        
        widget.bind("<Button-1>", start_move)
        widget.bind("<ButtonRelease-1>", stop_move)
        widget.bind("<B1-Motion>", do_move)
    
    def minimize_window(self):
        """Minimize window"""
        self.root.iconify()
    
    def show_properties(self):
        """Show properties dialog"""
        PropertiesDialog(self.root, self)
    
    def open_calibration(self):
        """Open full-screen calibration"""
        CalibrationWindow(self.root, self.settings)
    
    def start_service_menu(self):
        """Start service from menu"""
        if not self.service_running:
            self.service_running = True
            self.start_service()
            self.update_status("Service started", "green")
    
    def stop_service_menu(self):
        """Stop service from menu"""
        self.service_running = False
        self.update_status("Service stopped", "red")
    
    def restart_service(self):
        """Restart service"""
        self.service_running = False
        time.sleep(1)
        self.service_running = True
        self.start_service()
        self.update_status("Service restarted", "green")
    
    def install_service(self):
        """Install as Windows service"""
        messagebox.showinfo("Install Service", "Windows service installation not implemented in minimal version.")
    
    def remove_service(self):
        """Remove Windows service"""
        messagebox.showinfo("Remove Service", "Windows service removal not implemented in minimal version.")
    
    def update_status(self, message, color="white"):
        """Update status bar"""
        self.status_label.config(text=message)
        self.service_status.config(fg=color)
    
    def capture_frame(self):
        """Capture current screen frame with cursor"""
        try:
            # Get output path
            output_path = Path(self.settings["output_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Capture based on settings
            if self.settings["cursor_type"] == "none":
                screen_data = self.capture.capture_primary_monitor(include_cursor=False)
            else:
                screen_data = self.capture.capture_primary_monitor(include_cursor=True)
            
            # Save capture
            self.capture.save_capture(screen_data, str(output_path))
            
            self.frame_count += 1
            return True
            
        except Exception as e:
            print(f"Capture failed: {e}")
            return False
    
    def service_loop(self):
        """Background service loop with change detection"""
        last_hash = None
        last_capture_time = 0
        min_interval = 1.0
        max_interval = 5.0
        
        while self.service_running:
            current_time = time.time()
            
            try:
                # Quick screen hash to detect changes
                sample_data = self.capture.capture_screen_region(0, 0, 200, 200, include_cursor=False)
                current_hash = hash(sample_data.tobytes())
                
                # Check if screen changed
                screen_changed = (last_hash is None or current_hash != last_hash)
                max_time_exceeded = (current_time - last_capture_time) > max_interval
                min_time_passed = (current_time - last_capture_time) > min_interval
                
                if (screen_changed or max_time_exceeded) and min_time_passed:
                    self.capture_frame()
                    last_hash = current_hash
                    last_capture_time = current_time
                
                time.sleep(0.5)
                
            except Exception as e:
                self.capture_frame()
                time.sleep(5)
    
    def start_service(self):
        """Start the background capture service"""
        if hasattr(self, 'service_thread') and self.service_thread.is_alive():
            return
        
        self.service_thread = threading.Thread(target=self.service_loop, daemon=True)
        self.service_thread.start()
        self.update_status("Service running", "green")
    
    def start_monitoring(self):
        """Start image file monitoring"""
        self.monitor_thread = threading.Thread(target=self.monitor_image, daemon=True)
        self.monitor_thread.start()
    
    def monitor_image(self):
        """Monitor the stream image file for changes"""
        while self.running:
            try:
                image_path = Path(self.settings["output_path"])
                if image_path.exists():
                    modified_time = image_path.stat().st_mtime
                    if modified_time != self.last_modified:
                        self.last_modified = modified_time
                        self.load_and_display_image()
                else:
                    self.gui_queue.put(('status', 'Waiting for capture...'))
                    
            except Exception as e:
                self.gui_queue.put(('status', f'Error: {str(e)[:30]}'))
                
            time.sleep(0.5)
    
    def load_and_display_image(self):
        """Load and display the stream image"""
        try:
            image_path = Path(self.settings["output_path"])
            pil_image = Image.open(image_path)
            
            # Scale to fit window
            display_width = self.settings["window_width"]
            display_height = self.settings["window_height"] - 50  # Account for title bar and status
            
            original_width, original_height = pil_image.size
            scale = min(display_width / original_width, display_height / original_height)
            
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage and queue for GUI update
            photo = ImageTk.PhotoImage(pil_image)
            self.gui_queue.put(('image', photo, new_width, new_height))
            
        except Exception as e:
            self.gui_queue.put(('status', f'Display error: {str(e)[:30]}'))
    
    def process_gui_updates(self):
        """Process queued GUI updates"""
        try:
            while True:
                update_type, *data = self.gui_queue.get_nowait()
                
                if update_type == 'image':
                    photo, width, height = data
                    self.image_label.config(image=photo, text="")
                    self.image_label.image = photo
                    self.update_status(f"Frame {self.frame_count}")
                    
                elif update_type == 'status':
                    message = data[0]
                    self.update_status(message)
                    
        except queue.Empty:
            pass
        
        # Schedule next update
        self.root.after(100, self.process_gui_updates)
    
    def on_closing(self):
        """Handle window closing"""
        self.service_running = False
        self.running = False
        self.save_settings()
        self.root.quit()
        self.root.destroy()
    
    def run(self):
        """Start the viewer"""
        # Position window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (self.settings["window_width"] // 2)
        y = 100
        self.root.geometry(f"+{x}+{y}")
        
        self.root.mainloop()


class PropertiesDialog:
    """Properties dialog for viewer settings"""
    
    def __init__(self, parent, viewer):
        self.parent = parent
        self.viewer = viewer
        self.settings = viewer.settings.copy()
        
        # Create dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Properties")
        self.dialog.geometry("400x500")
        self.dialog.configure(bg='#f0f0f0')
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create property widgets"""
        # Notebook for tabs
        notebook = ttk.Notebook(self.dialog)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # General tab
        general_frame = ttk.Frame(notebook)
        notebook.add(general_frame, text="General")
        
        # Output settings
        ttk.Label(general_frame, text="Output Path:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.output_path = tk.StringVar(value=self.settings["output_path"])
        ttk.Entry(general_frame, textvariable=self.output_path, width=40).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(general_frame, text="Browse", command=self.browse_output).grid(row=0, column=2, padx=5)
        
        # Archive folder
        ttk.Label(general_frame, text="Archive Folder:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.archive_folder = tk.StringVar(value=self.settings["archive_folder"])
        ttk.Entry(general_frame, textvariable=self.archive_folder, width=40).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(general_frame, text="Browse", command=self.browse_archive).grid(row=1, column=2, padx=5)
        
        # Target selection
        ttk.Label(general_frame, text="Monitor Target:").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.monitor_target = tk.StringVar(value=self.settings["monitor_target"])
        monitor_combo = ttk.Combobox(general_frame, textvariable=self.monitor_target, 
                                   values=["primary", "secondary", "all", "window", "region"])
        monitor_combo.grid(row=2, column=1, sticky='w', padx=5, pady=5)
        
        # Cursor tab
        cursor_frame = ttk.Frame(notebook)
        notebook.add(cursor_frame, text="Cursor")
        
        # Cursor type
        ttk.Label(cursor_frame, text="Cursor Type:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.cursor_type = tk.StringVar(value=self.settings["cursor_type"])
        cursor_combo = ttk.Combobox(cursor_frame, textvariable=self.cursor_type,
                                  values=["cross", "cursor", "none"])
        cursor_combo.grid(row=0, column=1, sticky='w', padx=5, pady=5)
        
        # Cursor size
        ttk.Label(cursor_frame, text="Cursor Size:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.cursor_size = tk.IntVar(value=self.settings["cursor_size"])
        ttk.Scale(cursor_frame, from_=5, to=50, variable=self.cursor_size, orient='horizontal').grid(row=1, column=1, sticky='ew', padx=5, pady=5)
        
        # Cursor thickness
        ttk.Label(cursor_frame, text="Cursor Thickness:").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.cursor_thickness = tk.IntVar(value=self.settings["cursor_thickness"])
        ttk.Scale(cursor_frame, from_=1, to=10, variable=self.cursor_thickness, orient='horizontal').grid(row=2, column=1, sticky='ew', padx=5, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Button(button_frame, text="OK", command=self.ok_clicked).pack(side='right', padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.cancel_clicked).pack(side='right', padx=5)
    
    def browse_output(self):
        """Browse for output file"""
        filename = filedialog.asksaveasfilename(
            title="Select Output File",
            filetypes=[("PNG files", "*.png"), ("JPG files", "*.jpg"), ("All files", "*.*")]
        )
        if filename:
            self.output_path.set(filename)
    
    def browse_archive(self):
        """Browse for archive folder"""
        folder = filedialog.askdirectory(title="Select Archive Folder")
        if folder:
            self.archive_folder.set(folder)
    
    def ok_clicked(self):
        """Apply settings and close"""
        self.viewer.settings["output_path"] = self.output_path.get()
        self.viewer.settings["archive_folder"] = self.archive_folder.get()
        self.viewer.settings["monitor_target"] = self.monitor_target.get()
        self.viewer.settings["cursor_type"] = self.cursor_type.get()
        self.viewer.settings["cursor_size"] = self.cursor_size.get()
        self.viewer.settings["cursor_thickness"] = self.cursor_thickness.get()
        
        self.viewer.save_settings()
        self.dialog.destroy()
    
    def cancel_clicked(self):
        """Cancel without applying"""
        self.dialog.destroy()


class CalibrationWindow:
    """Full-screen calibration for LLMs"""
    
    def __init__(self, parent, settings):
        self.parent = parent
        self.settings = settings
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
        self.window.overrideredirect(True)
        
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
        self.canvas.focus_set()
        
        # Start calibration immediately
        self.start_calibration()
    
    def start_calibration(self):
        """Start calibration process"""
        self.targets.clear()
        self.results.clear()
        self.current_target = 0
        
        # Create 9 targets in 3x3 grid
        margin = 150
        
        for row in range(3):
            for col in range(3):
                x = margin + col * (self.screen_width - 2*margin) // 2
                y = margin + row * (self.screen_height - 2*margin) // 2
                self.targets.append({'x': x, 'y': y, 'clicked': False})
        
        self.show_next_target()
    
    def show_next_target(self):
        """Show next calibration target"""
        self.canvas.delete("all")
        
        if self.current_target < len(self.targets):
            target = self.targets[self.current_target]
            
            # Draw target
            size = self.settings.get("cursor_size", 15) * 2  # Use cursor size from settings
            self.canvas.create_oval(target['x']-size, target['y']-size, target['x']+size, target['y']+size, 
                                   fill='red', outline='white', width=4)
            self.canvas.create_oval(target['x']-8, target['y']-8, target['x']+8, target['y']+8, 
                                   fill='white', outline='black', width=2)
            
            # Show completed targets
            for prev_target in self.targets[:self.current_target]:
                self.canvas.create_oval(prev_target['x']-20, prev_target['y']-20, 
                                       prev_target['x']+20, prev_target['y']+20, 
                                       fill='green', outline='white', width=3)
            
            # Show progress
            self.canvas.create_text(50, 50, text=f"Target {self.current_target + 1}/9", 
                                   fill='white', font=('Arial', 20), anchor='nw')
            self.canvas.create_text(self.screen_width - 50, 50, text="Press ESC to cancel", 
                                   fill='gray', font=('Arial', 16), anchor='ne')
        else:
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
            
            target['clicked'] = True
            self.current_target += 1
            
            if self.current_target < len(self.targets):
                self.show_next_target()
            else:
                self.save_results()
                self.close_window()
    
    def save_results(self):
        """Save calibration results"""
        try:
            results_data = {
                'timestamp': datetime.now().isoformat(),
                'screen_size': {'width': self.screen_width, 'height': self.screen_height},
                'settings': self.settings,
                'results': self.results,
                'summary': {
                    'total_targets': len(self.results),
                    'avg_accuracy': sum(r['accuracy'] for r in self.results) / len(self.results) if self.results else 0,
                    'avg_distance': sum(r['distance'] for r in self.results) / len(self.results) if self.results else 0
                }
            }
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"calibration_results_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(results_data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving calibration results: {e}")
    
    def cancel_calibration(self, event=None):
        """Cancel calibration"""
        self.close_window()
    
    def close_window(self):
        """Close calibration window"""
        self.window.destroy()


if __name__ == "__main__":
    viewer = MinimalViewer()
    viewer.run()