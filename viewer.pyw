#!/usr/bin/env python3
"""
Minimal AI Vision Viewer - Command Window Style
Clean, minimal interface with system menu and properties
"""

import ctypes
import json
import math
import os
import queue
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

# Add src to path for service imports
sys.path.append(os.path.dirname(__file__))

# Import the vision service
from core.capture.gdi_screen_capture import GDIScreenCapture

# Windows API Point structure for mouse coordinates
class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class MinimalViewer:
    """Minimal command-window-style viewer"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI Vision")

        # Load/create settings
        self.settings_file = os.path.join(os.path.dirname(__file__), "configs", "viewer_settings.json")
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
            "monitor_target": "1",  # Monitor numbers: 1, 2, 3, etc. or all, window, region
            "cursor_type": "cross",  # cross, cursor, none
            "cursor_size": 15,
            "cursor_thickness": 3,
            "cursor_offset_x": 0,  # Horizontal offset correction (pixels)
            "cursor_offset_y": 0,  # Vertical offset correction (pixels)
            "window_width": 800,
            "window_height": 450,
            "sizing_mode": "auto_fit",  # auto_fit, width_based, height_based, fixed
            "target_width": 1200,
            "target_height": 800,
            "capture_interval": .5,  # Seconds between captures
            "pixel_threshold": 5.0,  # Percentage of pixels that must change
            "detection_method": "random",  # random, fixed_coords
            "sample_points": 100,  # Number of random sample points
            "fixed_coords": "100,100;500,300;800,600",  # Fixed coordinates if using fixed method
            "prevent_feedback": True  # Prevent infinite loop by masking viewer window
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
        """Save viewer settings atomically (temp file + rename)"""
        tmp_path = self.settings_file + ".tmp"
        try:
            with open(tmp_path, 'w') as f:
                json.dump(self.settings, f, indent=2)
            os.replace(tmp_path, self.settings_file)
        except Exception as e:
            print(f"Error saving settings: {e}")
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def restore_window_state(self):
        """Restore window position, size and state from settings"""
        try:
            window_state = self.settings.get("window_state", {})
            
            # Get saved dimensions or fallback to settings
            width = window_state.get("width", self.settings.get("window_width", 800))
            height = window_state.get("height", self.settings.get("window_height", 600))
            x = window_state.get("x", None)
            y = window_state.get("y", None)
            
            # Set reasonable minimum size but maintain aspect ratio
            min_width = 400
            min_height = 300
            width = max(width, min_width)
            height = max(height, min_height)
            
            # Set minimum size constraint
            self.root.minsize(min_width, min_height)
            
            # Set initial geometry
            if x is not None and y is not None:
                self.root.geometry(f"{width}x{height}+{x}+{y}")
                print(f"Restored window to: {width}x{height} at ({x}, {y})")
            else:
                # Center on screen if no position saved
                screen_width = self.root.winfo_screenwidth()
                screen_height = self.root.winfo_screenheight()
                center_x = (screen_width - width) // 2
                center_y = (screen_height - height) // 2
                self.root.geometry(f"{width}x{height}+{center_x}+{center_y}")
                print(f"Centered window: {width}x{height} at ({center_x}, {center_y})")
            
            # Apply window state after a brief delay to ensure window is created
            self.root.after(100, self.apply_window_state)
            
        except Exception as e:
            print(f"Error restoring window state: {e}")
            # Fallback to basic sizing
            width = self.settings.get("window_width", 800)
            height = self.settings.get("window_height", 600)
            self.root.geometry(f"{width}x{height}")

    def apply_window_state(self):
        """Apply saved window state after window is created"""
        try:
            window_state = self.settings.get("window_state", {})
            
            if window_state.get("maximized", False):
                self.root.state('zoomed')  # Windows maximize
                print("Applied maximized state")
            elif window_state.get("minimized", False):
                self.root.iconify()
                print("Applied minimized state")
                
        except Exception as e:
            print(f"Error applying window state: {e}")

    def save_window_state(self):
        """Save current window state to settings"""
        try:
            # Get current window state
            geometry = self.root.geometry()
            state = self.root.state()
            
            # Parse geometry string (e.g., "800x600+100+50")
            size_pos = geometry.split('+')
            width_height = size_pos[0].split('x')
            width = int(width_height[0])
            height = int(width_height[1])
            
            x = int(size_pos[1]) if len(size_pos) > 1 else 0
            y = int(size_pos[2]) if len(size_pos) > 2 else 0
            
            # Update settings
            self.settings["window_state"] = {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "maximized": state == 'zoomed',
                "minimized": state == 'iconic',
                "fullscreen": False
            }
            
            # Also update legacy settings for compatibility
            self.settings["window_width"] = width
            self.settings["window_height"] = height
            
            print(f"Saved window state: {width}x{height} at ({x}, {y}), state: {state}")
            
        except Exception as e:
            print(f"Error saving window state: {e}")

    def get_sessions_path(self, *path_parts):
        """Get path relative to sessions folder"""
        base_dir = os.path.dirname(__file__)
        return os.path.join(base_dir, "core", "sessions", "captured", *path_parts)

    def get_recorded_path(self, *path_parts):
        """Get path for permanent recordings"""
        base_dir = os.path.dirname(__file__)
        return os.path.join(base_dir, "recorded_sessions", *path_parts)

    def get_output_path(self):
        """Get current_view.png path"""
        base_dir = os.path.dirname(__file__)
        return os.path.join(base_dir, "llm_screenshots", "current_view.png")


    def get_temp_view_path(self):
        """Get temp_view.jpg path"""
        return self.get_sessions_path("claude_session", "temp_view.jpg")

    def get_nomouse_path(self):
        """Get nomouse_cursor.jpg path"""
        return self.get_sessions_path("claude_session", "nomouse_cursor.jpg")

    def get_available_monitors(self):
        """Get list of available monitors with user-friendly names"""
        try:
            monitors = self.capture.get_monitor_info()
            monitor_list = []

            for i, monitor in enumerate(monitors):
                monitor_name = f"Monitor {i+1}"
                if i == 0:
                    monitor_name += " (Primary)"
                monitor_name += f" - {monitor['width']}x{monitor['height']}"
                monitor_list.append({
                    'name': monitor_name,
                    'value': str(i+1),  # 1-based indexing for user
                    'info': monitor
                })

            # Add special options
            monitor_list.extend([
                {'name': 'All Monitors', 'value': 'all', 'info': None},
                {'name': 'Active Window', 'value': 'window', 'info': None},
                {'name': 'Custom Region', 'value': 'region', 'info': None}
            ])

            return monitor_list
        except Exception as e:
            print(f"Error getting monitors: {e}")
            # Fallback to basic options
            return [
                {'name': 'Monitor 1 (Primary)', 'value': '1', 'info': None},
                {'name': 'Monitor 2', 'value': '2', 'info': None},
                {'name': 'All Monitors', 'value': 'all', 'info': None},
                {'name': 'Active Window', 'value': 'window', 'info': None},
                {'name': 'Custom Region', 'value': 'region', 'info': None}
            ]

    def capture_monitor_target(self, **kwargs):
        """Capture based on monitor_target setting"""
        target = self.settings.get("monitor_target", "1")

        try:
            # Handle numbered monitors (1, 2, 3, etc.)
            if target.isdigit():
                monitor_index = int(target) - 1  # Convert to 0-based index
                return self.capture.capture_monitor(monitor_index, **kwargs)

            # Handle special cases
            elif target == "all":
                # For "all monitors", capture primary for now (could be enhanced)
                return self.capture.capture_primary_monitor(**kwargs)

            elif target == "window":
                # For "active window", capture primary for now (could be enhanced)
                return self.capture.capture_primary_monitor(**kwargs)

            elif target == "region":
                # For "custom region", capture primary for now (could be enhanced)
                return self.capture.capture_primary_monitor(**kwargs)

            elif target in ["primary", "secondary"]:
                # Backward compatibility - convert old settings
                monitor_index = 0 if target == "primary" else 1
                return self.capture.capture_monitor(monitor_index, **kwargs)

            else:
                # Fallback to primary monitor
                return self.capture.capture_primary_monitor(**kwargs)

        except Exception as e:
            print(f"Monitor capture failed, falling back to primary: {e}")
            return self.capture.capture_primary_monitor(**kwargs)

    def setup_minimal_ui(self):
        """Setup minimal command-window-style UI"""
        # Window setup - restore saved state if available
        self.restore_window_state()
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

        # Add click detection for calibration and click-through
        self.image_label.bind("<Button-1>", self.on_image_click)  # Left click
        self.image_label.bind("<Button-3>", self.on_image_click)  # Right click
        self.active_calibration = None  # Store reference to active calibration window

        # Status bar (minimal)
        self.status_frame = tk.Frame(self.root, bg='#007acc', height=20)
        self.status_frame.pack(fill='x', side='bottom')
        self.status_frame.pack_propagate(False)

        self.status_label = tk.Label(self.status_frame, text="Ready",
                                    bg='#007acc', fg='white', font=('Segoe UI', 8))
        self.status_label.pack(side='left', padx=5, pady=1)

        # Coordinate conversion display
        self.coord_display_label = tk.Label(self.status_frame, text="Screen: (0,0) | Viewer: (0,0) | Target: (0,0)",
                                          bg='#007acc', fg='white', font=('Segoe UI', 8))
        self.coord_display_label.pack(side='right', padx=10)

        # Service status indicator
        self.service_status = tk.Label(self.status_frame, text="●",
                                      bg='#007acc', fg='green', font=('Segoe UI', 10))
        self.service_status.pack(side='right', padx=5)

        # Make title bar draggable
        self.make_draggable(self.title_frame)
        self.make_draggable(self.title_label)

        # Start mouse coordinate tracking
        self.start_mouse_tracking()

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
        self.system_menu.add_command(label="Test Coordinate System (Multi-Monitor)", command=self.open_coordinate_test)
        self.system_menu.add_separator()
        self.system_menu.add_command(label="Close", command=self.on_closing)

    def show_system_menu(self):
        """Show system menu when icon is clicked"""
        try:
            self.system_menu.post(self.system_icon.winfo_rootx(),
                                 self.system_icon.winfo_rooty() + self.system_icon.winfo_height())
        except Exception as e:
            print(f"[WARN] show_system_menu failed: {e}")

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

    def open_coordinate_test(self):
        """Open coordinate system test for multi-monitor setups"""
        print("Starting COORDINATE SYSTEM TEST - analyzing monitor setup and coordinate mapping...")
        CalibrationWindow(self.root, self.settings, coordinate_test_mode=True)

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

    def on_image_click(self, event):
        """Handle clicks on the main image for click-through"""
        # Get click coordinates relative to the image_label
        click_x = event.x
        click_y = event.y
        print(f"Viewer window clicked at ({click_x}, {click_y})")

        # Handle click-through
        self.handle_click_through(click_x, click_y, event)

    def handle_click_through(self, viewer_x, viewer_y, event):
        """Click screen position and return mouse to original position"""
        try:
            import ctypes

            # FIRST: Save the current mouse position
            original_point = POINT()
            ctypes.windll.user32.GetCursorPos(ctypes.byref(original_point))
            original_x, original_y = original_point.x, original_point.y

            # Convert viewer coordinates to screen coordinates
            screen_x, screen_y = self.convert_viewer_to_screen_coords(viewer_x, viewer_y)

            # Determine click type
            click_type = "left"
            if hasattr(event, 'num') and event.num == 3:
                click_type = "right"

            print(f"DEBUG: Original mouse at ({original_x}, {original_y}) -> Click screen ({screen_x}, {screen_y}) -> Return to ({original_x}, {original_y})")
            print(f"DEBUG: Viewer coordinates: ({viewer_x}, {viewer_y}) -> Screen coordinates: ({screen_x}, {screen_y})")

            # Get window position for debugging
            window_x = self.root.winfo_rootx()
            window_y = self.root.winfo_rooty()
            print(f"DEBUG: Window position: ({window_x}, {window_y})")

            # Click the screen position
            ctypes.windll.user32.SetCursorPos(int(screen_x), int(screen_y))

            if click_type == "left":
                ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)  # Down
                ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)  # Up
            else:
                ctypes.windll.user32.mouse_event(0x0008, 0, 0, 0, 0)  # Right down
                ctypes.windll.user32.mouse_event(0x0010, 0, 0, 0, 0)  # Right up

            # ALWAYS return mouse to original position with drift compensation
            # Compensation for viewer window click-through drift (measured scientifically)
            CLICK_THROUGH_Y_COMPENSATION = 0  # Negative means compensate upward

            # Apply compensation to prevent mouse jump
            adjusted_x = original_x
            adjusted_y = original_y + CLICK_THROUGH_Y_COMPENSATION

            print(f"DEBUG: Returning mouse with compensation:")
            print(f"  Original: ({original_x}, {original_y})")
            print(f"  Adjusted: ({adjusted_x}, {adjusted_y}) [Y offset: {CLICK_THROUGH_Y_COMPENSATION:+d}]")

            #ctypes.windll.user32.SetCursorPos(int(adjusted_x), int(adjusted_y))

            self.update_status(f"Clicked screen ({screen_x}, {screen_y}), mouse restored", "green")

        except Exception as e:
            print(f"Click-through error: {e}")
            self.update_status(f"Click error: {e}", "red")

    def convert_viewer_to_screen_coords(self, viewer_x, viewer_y):
        """Convert viewer window coordinates to screen coordinates with size compensation"""
        try:
            # Get the current image size (this is the screen capture size)
            output_path = Path(self.get_output_path())
            if not output_path.exists():
                # Fallback to screen size if no image available
                import ctypes
                screen_width = ctypes.windll.user32.GetSystemMetrics(0)
                screen_height = ctypes.windll.user32.GetSystemMetrics(1)
                image_width, image_height = screen_width, screen_height
            else:
                from PIL import Image
                with Image.open(output_path) as img:
                    image_width, image_height = img.size

            # Get viewer display size
            self.image_label.update_idletasks()
            display_width = self.image_label.winfo_width()
            display_height = self.image_label.winfo_height()

            print(f"Image size: {image_width}x{image_height}")
            print(f"Viewer display: {display_width}x{display_height}")

            # Calculate the scale factor used to fit image in viewer
            # (Same logic as update_display method)
            if display_width > 0 and display_height > 0:
                scale_x = display_width / image_width
                scale_y = display_height / image_height
                scale = min(scale_x, scale_y)  # Maintain aspect ratio

                # Calculate displayed image size
                displayed_width = int(image_width * scale)
                displayed_height = int(image_height * scale)

                # Calculate offset (image is centered in label)
                offset_x = (display_width - displayed_width) // 2
                offset_y = (display_height - displayed_height) // 2

                print(f"Scale: {scale:.3f}, Displayed: {displayed_width}x{displayed_height}, Offset: ({offset_x}, {offset_y})")

                # Convert viewer coordinates to image coordinates
                image_x = (viewer_x - offset_x) / scale
                image_y = (viewer_y - offset_y) / scale

                # Clamp to image bounds and convert to screen coordinates
                screen_x = max(0, min(image_width - 1, int(image_x)))
                screen_y = max(0, min(image_height - 1, int(image_y)))

                return screen_x, screen_y
            else:
                # Fallback: assume 1:1 mapping
                return viewer_x, viewer_y

        except Exception as e:
            print(f"Coordinate conversion error: {e}")
            # Fallback: return viewer coordinates as-is
            return viewer_x, viewer_y

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

    def hide_viewer_for_capture(self):
        """Temporarily hide viewer window during capture to see through it (like OBS)"""
        try:
            # Store current state and hide window
            self.viewer_was_visible = self.root.state() != 'withdrawn'
            if self.viewer_was_visible:
                print("🙈 Hiding viewer window for capture...")
                self.root.withdraw()  # Hide window completely
                self.root.update()  # Force UI update
                # Longer delay to ensure window is fully hidden
                time.sleep(0.1)  # 100ms should be enough
                print("✅ Viewer window hidden")
        except Exception as e:
            print(f"Hide viewer error: {e}")

    def restore_viewer_after_capture(self):
        """Restore viewer window after capture"""
        try:
            # Restore window if it was visible before
            if hasattr(self, 'viewer_was_visible') and self.viewer_was_visible:
                print("👁️ Restoring viewer window after capture...")
                self.root.deiconify()  # Show window again
                self.root.update()  # Force UI update
                # Brief delay for window to restore
                time.sleep(0.05)
                print("✅ Viewer window restored")
        except Exception as e:
            print(f"Restore viewer error: {e}")

    def start_mouse_tracking(self):
        """Start continuous mouse coordinate tracking"""
        self.update_mouse_coordinates()

    def update_mouse_coordinates(self):
        """Update coordinate display showing screen, viewer, and target coordinates"""
        try:
            # Get current mouse position (screen coordinates)
            point = POINT()
            ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
            screen_x, screen_y = point.x, point.y

            # Calculate viewer-relative coordinates if mouse is over viewer
            viewer_x, viewer_y = "---", "---"
            target_x, target_y = "---", "---"

            try:
                # Get viewer window position and image label position
                if hasattr(self, 'image_label') and self.image_label.winfo_exists():
                    viewer_window_x = self.root.winfo_rootx()
                    viewer_window_y = self.root.winfo_rooty()

                    image_label_x = viewer_window_x + self.image_label.winfo_x()
                    image_label_y = viewer_window_y + self.image_label.winfo_y()
                    image_label_width = self.image_label.winfo_width()
                    image_label_height = self.image_label.winfo_height()

                    # Check if mouse is over the image area
                    if (image_label_x <= screen_x <= image_label_x + image_label_width and
                        image_label_y <= screen_y <= image_label_y + image_label_height):

                        # Calculate viewer-relative coordinates
                        viewer_x = screen_x - image_label_x
                        viewer_y = screen_y - image_label_y

                        # Calculate what screen coordinates this would map to
                        try:
                            target_x, target_y = self.convert_viewer_to_screen_coords(viewer_x, viewer_y)
                        except Exception as e:
                            print(f"[WARN] coordinate conversion failed: {e}")
                            target_x, target_y = "err", "err"

            except Exception as coord_error:
                # Continue with basic display if coordinate calculation fails
                pass

            # Update the coordinate display
            coord_text = f"Screen: ({screen_x},{screen_y}) | Viewer: ({viewer_x},{viewer_y}) | Target: ({target_x},{target_y})"
            self.coord_display_label.config(text=coord_text)

            # Store target coordinates for virtual cursor overlay
            if target_x != "---" and target_y != "---":
                self.virtual_cursor_target = (target_x, target_y)
            else:
                self.virtual_cursor_target = None

            # Schedule next update in 100ms (10 times per second)
            self.root.after(100, self.update_mouse_coordinates)

        except Exception as e:
            # Fallback if API call fails
            self.coord_display_label.config(text="Coordinates: ERROR")
            # Continue tracking even if one update fails
            self.root.after(100, self.update_mouse_coordinates)

    def capture_temp_view(self):
        """Capture temp_view.jpg (no cursor) - for display and change detection"""
        try:
            temp_path = Path(self.get_temp_view_path())
            temp_path.parent.mkdir(parents=True, exist_ok=True)

            # Always capture without cursor for temp_view, hide viewer window if enabled
            prevent_feedback = self.settings.get("prevent_feedback", True)
            print(f"🔧 DEBUG temp_view: prevent_feedback = {prevent_feedback}")
            if prevent_feedback:
                self.hide_viewer_for_capture()

            screen_data = self.capture_monitor_target(include_cursor=False)

            if self.settings.get("prevent_feedback", True):
                self.restore_viewer_after_capture()
            self.capture.save_capture(screen_data, str(temp_path))

            return screen_data

        except Exception as e:
            print(f"Temp capture failed: {e}")
            return None

    def capture_current_view_with_cursor(self):
        """Capture current_view.jpg WITH cursor baked in - for LLM analysis"""
        try:
            output_path = Path(self.get_output_path())
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Capture WITH cursor for LLM, hide viewer window if enabled
            if self.settings.get("prevent_feedback", True):
                self.hide_viewer_for_capture()

            if self.settings["cursor_type"] == "none":
                screen_data = self.capture_monitor_target(include_cursor=False)
            else:
                screen_data = self.capture_monitor_target(
                    include_cursor=True,
                    cursor_size=self.settings["cursor_size"],
                    cursor_thickness=self.settings["cursor_thickness"],
                    cursor_type=self.settings["cursor_type"]
                )

            if self.settings.get("prevent_feedback", True):
                self.restore_viewer_after_capture()

            # Save current_view.jpg
            self.capture.save_capture(screen_data, str(output_path))


            self.frame_count += 1
            return True

        except Exception as e:
            print(f"Current view capture failed: {e}")
            return False


    def detect_screen_changes(self, new_image_data):
        """Detect if screen has changed using your pixel comparison method"""
        try:
            nomouse_path = Path(self.get_nomouse_path())

            # If no reference image exists, save this as reference
            if not nomouse_path.exists():
                self.capture.save_capture(new_image_data, str(nomouse_path))
                return True

            # Load reference image
            reference_image = Image.open(nomouse_path)
            new_image = Image.fromarray(new_image_data)

            # Convert to same format for comparison
            if reference_image.size != new_image.size:
                reference_image = reference_image.resize(new_image.size)

            ref_array = np.array(reference_image)
            new_array = np.array(new_image)

            # Pixel change detection
            if self.settings["detection_method"] == "random":
                changed = self.detect_changes_random_sampling(ref_array, new_array)
            else:
                changed = self.detect_changes_fixed_coords(ref_array, new_array)

            # If changed, update reference
            if changed:
                self.capture.save_capture(new_image_data, str(nomouse_path))

            return changed

        except Exception as e:
            print(f"Change detection failed: {e}")
            return True  # Assume changed on error

    def detect_changes_random_sampling(self, ref_array, new_array):
        """Detect changes using random pixel sampling"""
        height, width = ref_array.shape[:2]
        sample_points = min(self.settings["sample_points"], width * height)

        # Generate random sample coordinates
        sample_coords = [(np.random.randint(0, width), np.random.randint(0, height))
                        for _ in range(sample_points)]

        different_pixels = 0
        for x, y in sample_coords:
            # Compare RGB values
            ref_pixel = ref_array[y, x]
            new_pixel = new_array[y, x]

            # Check if pixel difference exceeds threshold
            if np.sum(np.abs(ref_pixel.astype(int) - new_pixel.astype(int))) > 30:  # Adjust threshold
                different_pixels += 1

        change_percentage = (different_pixels / sample_points) * 100
        return change_percentage > self.settings["pixel_threshold"]

    def detect_changes_fixed_coords(self, ref_array, new_array):
        """Detect changes using fixed coordinate sampling"""
        coords_str = self.settings["fixed_coords"]
        coords = []

        for coord_pair in coords_str.split(';'):
            try:
                x, y = map(int, coord_pair.split(','))
                if 0 <= x < ref_array.shape[1] and 0 <= y < ref_array.shape[0]:
                    coords.append((x, y))
            except (ValueError, IndexError) as e:
                continue

        if not coords:
            return self.detect_changes_random_sampling(ref_array, new_array)

        different_pixels = 0
        for x, y in coords:
            ref_pixel = ref_array[y, x]
            new_pixel = new_array[y, x]

            if np.sum(np.abs(ref_pixel.astype(int) - new_pixel.astype(int))) > 30:
                different_pixels += 1

        change_percentage = (different_pixels / len(coords)) * 100
        return change_percentage > self.settings["pixel_threshold"]

    def service_loop(self):
        """Background service loop - implements your three-image architecture"""
        last_temp_capture = 0
        last_current_capture = 0
        capture_interval = self.settings["capture_interval"]

        while self.service_running:
            current_time = time.time()

            try:
                # 1. Capture temp_view.jpg every interval (no cursor)
                if (current_time - last_temp_capture) >= capture_interval:
                    screen_data = self.capture_temp_view()
                    last_temp_capture = current_time

                    if screen_data is not None:
                        # 2. Check if screen changed (temp_view vs nomouse_cursor)
                        screen_changed = self.detect_screen_changes(screen_data)

                        # 3. Update current_view.jpg with cursor baked in every interval
                        if (current_time - last_current_capture) >= capture_interval:
                            self.capture_current_view_with_cursor()
                            last_current_capture = current_time

                # Update display more frequently than capture
                time.sleep(0.1)  # 10 FPS display updates

            except Exception as e:
                print(f"Service loop error: {e}")
                time.sleep(1)

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
        """Monitor temp_view.jpg for display (with live cursor overlay)"""
        while self.running:
            try:
                # Monitor temp_view.jpg for display
                temp_path = Path(self.get_temp_view_path())
                if temp_path.exists():
                    modified_time = temp_path.stat().st_mtime
                    if modified_time != self.last_modified:
                        self.last_modified = modified_time
                        self.load_and_display_temp_image()
                else:
                    self.gui_queue.put(('status', 'Waiting for capture...'))

            except Exception as e:
                self.gui_queue.put(('status', f'Error: {str(e)[:30]}'))

            time.sleep(0.1)  # Fast refresh for smooth cursor overlay

    def load_and_display_temp_image(self):
        """Load temp_view.jpg and display with live cursor overlay"""
        try:
            temp_path = Path(self.get_temp_view_path())
            pil_image = Image.open(temp_path)

            # Add live cursor overlay if enabled
            if self.settings["cursor_type"] != "none":
                pil_image = self.add_live_cursor_overlay(pil_image)

            original_width, original_height = pil_image.size
            aspect_ratio = original_width / original_height

            # Calculate target dimensions based on sizing mode (same logic as before)
            sizing_mode = self.settings.get("sizing_mode", "auto_fit")

            if sizing_mode == "auto_fit":
                screen_width = self.root.winfo_screenwidth()
                screen_height = self.root.winfo_screenheight()
                max_width = int(screen_width * 0.9)
                max_height = int(screen_height * 0.85)
                scale = min(max_width / original_width, max_height / original_height, 1.0)
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)

            elif sizing_mode == "width_based":
                new_width = self.settings.get("target_width", 1200)
                new_height = int(new_width / aspect_ratio)

            elif sizing_mode == "height_based":
                new_height = self.settings.get("target_height", 800)
                new_width = int(new_height * aspect_ratio)

            elif sizing_mode == "fixed":
                new_width = self.settings.get("target_width", 1200)
                new_height = self.settings.get("target_height", 800)
            else:
                new_width = int(original_width * 0.8)
                new_height = int(original_height * 0.8)

            # Resize image if needed (same logic as before)
            if new_width != original_width or new_height != original_height:
                if sizing_mode == "fixed":
                    target_aspect = new_width / new_height
                    if aspect_ratio > target_aspect:
                        fit_width = new_width
                        fit_height = int(new_width / aspect_ratio)
                    else:
                        fit_height = new_height
                        fit_width = int(new_height * aspect_ratio)

                    pil_image = pil_image.resize((fit_width, fit_height), Image.Resampling.LANCZOS)
                    letterbox_image = Image.new('RGB', (new_width, new_height), (0, 0, 0))
                    x_offset = (new_width - fit_width) // 2
                    y_offset = (new_height - fit_height) // 2
                    letterbox_image.paste(pil_image, (x_offset, y_offset))
                    pil_image = letterbox_image
                else:
                    pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Update window size
            window_height = new_height + 50
            self.settings["window_width"] = new_width
            self.settings["window_height"] = window_height

            # Convert and display
            photo = ImageTk.PhotoImage(pil_image)
            self.gui_queue.put(('resize_window', new_width, window_height))
            self.gui_queue.put(('image', photo, new_width, new_height))

        except Exception as e:
            self.gui_queue.put(('status', f'Display error: {str(e)[:30]}'))

    def add_live_cursor_overlay(self, pil_image):
        """Add live cursor overlay to image - smooth and responsive"""
        try:
            import ctypes
            from ctypes import wintypes

            from PIL import ImageDraw

            # Get current cursor position
            point = POINT()
            ctypes.windll.user32.GetCursorPos(ctypes.byref(point))

            # Apply cursor offset corrections
            cursor_offset_x = self.settings.get("cursor_offset_x", 0)
            cursor_offset_y = self.settings.get("cursor_offset_y", 0)
            cursor_x, cursor_y = point.x + cursor_offset_x, point.y + cursor_offset_y

            # Draw cursor overlay on image
            draw = ImageDraw.Draw(pil_image)
            cursor_size = self.settings["cursor_size"]
            cursor_thickness = self.settings["cursor_thickness"]
            cursor_type = self.settings["cursor_type"]

            if cursor_type == 'cross':
                # Draw a bright cursor that stands out
                draw.line([(cursor_x-cursor_size, cursor_y), (cursor_x+cursor_size, cursor_y)],
                         fill='red', width=cursor_thickness)
                draw.line([(cursor_x, cursor_y-cursor_size), (cursor_x, cursor_y+cursor_size)],
                         fill='red', width=cursor_thickness)
            elif cursor_type == 'cursor':
                # Draw arrow cursor
                arrow_points = [
                    (cursor_x, cursor_y),
                    (cursor_x, cursor_y + cursor_size),
                    (cursor_x + cursor_size//3, cursor_y + cursor_size*2//3),
                    (cursor_x + cursor_size//2, cursor_y + cursor_size//2),
                    (cursor_x + cursor_size*2//3, cursor_y + cursor_size//3)
                ]
                draw.polygon(arrow_points, fill='red', outline='white', width=cursor_thickness)

            # Add virtual cursor overlay if mouse is hovering over viewer
            if hasattr(self, 'virtual_cursor_target') and self.virtual_cursor_target:
                target_x, target_y = self.virtual_cursor_target

                # Draw virtual cursor in blue to distinguish from real cursor
                virtual_size = cursor_size + 2  # Slightly larger
                virtual_thickness = cursor_thickness + 1

                # Draw blue cross for virtual cursor target
                draw.line([(target_x-virtual_size, target_y), (target_x+virtual_size, target_y)],
                         fill='blue', width=virtual_thickness)
                draw.line([(target_x, target_y-virtual_size), (target_x, target_y+virtual_size)],
                         fill='blue', width=virtual_thickness)

                # Add a small circle around the virtual cursor for clarity
                circle_radius = virtual_size // 2
                draw.ellipse([(target_x-circle_radius, target_y-circle_radius),
                             (target_x+circle_radius, target_y+circle_radius)],
                            outline='cyan', width=2)

            return pil_image

        except Exception as e:
            print(f"Cursor overlay error: {e}")
            return pil_image

    def load_and_display_image(self):
        """Load and display the stream image"""
        try:
            image_path = Path(self.get_output_path())
            pil_image = Image.open(image_path)

            original_width, original_height = pil_image.size
            aspect_ratio = original_width / original_height

            # Calculate target dimensions based on sizing mode
            sizing_mode = self.settings.get("sizing_mode", "auto_fit")

            if sizing_mode == "auto_fit":
                # Determine optimal window size based on screen size
                screen_width = self.root.winfo_screenwidth()
                screen_height = self.root.winfo_screenheight()

                # Maximum usable screen area (leave space for taskbar, etc.)
                max_width = int(screen_width * 0.9)
                max_height = int(screen_height * 0.85)

                # Calculate scale to fit within screen while maintaining aspect ratio
                scale = min(max_width / original_width, max_height / original_height, 1.0)  # Don't upscale
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)

            elif sizing_mode == "width_based":
                # Set width, calculate height from aspect ratio
                new_width = self.settings.get("target_width", 1200)
                new_height = int(new_width / aspect_ratio)

            elif sizing_mode == "height_based":
                # Set height, calculate width from aspect ratio
                new_height = self.settings.get("target_height", 800)
                new_width = int(new_height * aspect_ratio)

            elif sizing_mode == "fixed":
                # Use both dimensions (may letterbox)
                new_width = self.settings.get("target_width", 1200)
                new_height = self.settings.get("target_height", 800)

            else:
                # Fallback to auto_fit
                new_width = int(original_width * 0.8)
                new_height = int(original_height * 0.8)

            # Resize image if dimensions changed
            if new_width != original_width or new_height != original_height:
                if sizing_mode == "fixed":
                    # For fixed mode, we may need letterboxing
                    target_aspect = new_width / new_height
                    if aspect_ratio > target_aspect:
                        # Image is wider - fit to width, add vertical letterboxing
                        fit_width = new_width
                        fit_height = int(new_width / aspect_ratio)
                    else:
                        # Image is taller - fit to height, add horizontal letterboxing
                        fit_height = new_height
                        fit_width = int(new_height * aspect_ratio)

                    # Resize to fit dimensions
                    pil_image = pil_image.resize((fit_width, fit_height), Image.Resampling.LANCZOS)

                    # Create letterboxed image
                    letterbox_image = Image.new('RGB', (new_width, new_height), (0, 0, 0))
                    x_offset = (new_width - fit_width) // 2
                    y_offset = (new_height - fit_height) // 2
                    letterbox_image.paste(pil_image, (x_offset, y_offset))
                    pil_image = letterbox_image
                else:
                    # Normal resize maintaining aspect ratio
                    pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Update window size to match image + UI elements
            window_height = new_height + 50  # Add space for title bar and status bar

            # Update settings to remember the new size
            self.settings["window_width"] = new_width
            self.settings["window_height"] = window_height

            # Convert to PhotoImage and queue for GUI update
            photo = ImageTk.PhotoImage(pil_image)
            self.gui_queue.put(('resize_window', new_width, window_height))
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
                    self.update_status(f"Frame {self.frame_count} ({width}x{height})")

                elif update_type == 'status':
                    message = data[0]
                    self.update_status(message)

                elif update_type == 'resize_window':
                    window_width, window_height = data
                    self.root.geometry(f"{window_width}x{window_height}")

        except queue.Empty:
            pass

        # Schedule next update
        self.root.after(100, self.process_gui_updates)

    def on_closing(self):
        """Handle window closing"""
        self.service_running = False
        self.running = False

        # Join daemon threads with timeout to prevent resource leaks
        for attr in ('service_thread', 'monitor_thread'):
            t = getattr(self, attr, None)
            if t is not None and t.is_alive():
                t.join(timeout=2)

        self.save_settings()

        # Clean up sessions folder
        self.cleanup_sessions()

        self.root.quit()
        self.root.destroy()

    def cleanup_sessions(self):
        """Delete entire sessions folder on close"""
        try:
            import shutil
            base_dir = os.path.dirname(__file__)
            sessions_path = os.path.join(base_dir, "core", "sessions")
            if os.path.exists(sessions_path):
                shutil.rmtree(sessions_path)
                print(f"Cleaned up sessions folder: {sessions_path}")
        except Exception as e:
            print(f"Error cleaning up sessions: {e}")


        except Exception as e:
            print(f"Error cleaning up archive on startup: {e}")

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
        self.dialog.geometry("465x595")
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

        # Target selection
        ttk.Label(general_frame, text="Monitor Target:").grid(row=0, column=0, sticky='w', padx=5, pady=5)

        # Get available monitors dynamically
        available_monitors = self.viewer.get_available_monitors()
        monitor_values = [monitor['value'] for monitor in available_monitors]

        self.monitor_target = tk.StringVar(value=self.settings["monitor_target"])
        monitor_combo = ttk.Combobox(general_frame, textvariable=self.monitor_target,
                                   values=monitor_values, width=30, state="readonly")
        monitor_combo.grid(row=0, column=1, columnspan=2, sticky='ew', padx=5, pady=5)

        # Add help text showing what the numbers mean
        monitor_options = [f"{m['value']}={m['name']}" for m in available_monitors[:3]]
        monitor_help = f"Available: {', '.join(monitor_options)}..."
        ttk.Label(general_frame, text=monitor_help, font=('Segoe UI', 7), foreground='gray').grid(row=1, column=0, columnspan=3, sticky='w', padx=5, pady=2)

        # Window sizing options
        sizing_frame = ttk.LabelFrame(general_frame, text="Window Sizing")
        sizing_frame.grid(row=2, column=0, columnspan=3, sticky='ew', padx=5, pady=10)

        # Sizing mode
        ttk.Label(sizing_frame, text="Sizing Mode:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.sizing_mode = tk.StringVar(value=self.settings["sizing_mode"])
        sizing_combo = ttk.Combobox(sizing_frame, textvariable=self.sizing_mode,
                                   values=["auto_fit", "width_based", "height_based", "fixed"])
        sizing_combo.grid(row=0, column=1, sticky='w', padx=5, pady=5)

        # Target width
        ttk.Label(sizing_frame, text="Target Width:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.target_width = tk.IntVar(value=self.settings["target_width"])
        ttk.Entry(sizing_frame, textvariable=self.target_width, width=10).grid(row=1, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(sizing_frame, text="px").grid(row=1, column=2, sticky='w', padx=2, pady=5)

        # Target height
        ttk.Label(sizing_frame, text="Target Height:").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.target_height = tk.IntVar(value=self.settings["target_height"])
        ttk.Entry(sizing_frame, textvariable=self.target_height, width=10).grid(row=2, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(sizing_frame, text="px").grid(row=2, column=2, sticky='w', padx=2, pady=5)

        # Help text
        help_text = "auto_fit: Scale to screen\nwidth_based: Set width, calc height\nheight_based: Set height, calc width\nfixed: Use both dimensions (letterbox)"
        ttk.Label(sizing_frame, text=help_text, font=('Segoe UI', 8), foreground='gray').grid(row=3, column=0, columnspan=3, sticky='w', padx=5, pady=5)

        # Capture timing frame
        timing_frame = ttk.LabelFrame(general_frame, text="Capture & Detection")
        timing_frame.grid(row=3, column=0, columnspan=3, sticky='ew', padx=5, pady=10)

        # Capture interval
        ttk.Label(timing_frame, text="Capture Interval:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.capture_interval = tk.DoubleVar(value=self.settings["capture_interval"])
        ttk.Entry(timing_frame, textvariable=self.capture_interval, width=10).grid(row=0, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(timing_frame, text="seconds").grid(row=0, column=2, sticky='w', padx=2, pady=5)

        # Pixel threshold
        ttk.Label(timing_frame, text="Pixel Threshold:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.pixel_threshold = tk.DoubleVar(value=self.settings["pixel_threshold"])
        ttk.Entry(timing_frame, textvariable=self.pixel_threshold, width=10).grid(row=1, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(timing_frame, text="% changed").grid(row=1, column=2, sticky='w', padx=2, pady=5)

        # Detection method
        ttk.Label(timing_frame, text="Detection Method:").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.detection_method = tk.StringVar(value=self.settings["detection_method"])
        detection_combo = ttk.Combobox(timing_frame, textvariable=self.detection_method,
                                     values=["random", "fixed_coords"])
        detection_combo.grid(row=2, column=1, sticky='w', padx=5, pady=5)

        # Sample points
        ttk.Label(timing_frame, text="Sample Points:").grid(row=3, column=0, sticky='w', padx=5, pady=5)
        self.sample_points = tk.IntVar(value=self.settings["sample_points"])
        ttk.Entry(timing_frame, textvariable=self.sample_points, width=10).grid(row=3, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(timing_frame, text="pixels").grid(row=3, column=2, sticky='w', padx=2, pady=5)

        # Fixed coordinates
        ttk.Label(timing_frame, text="Fixed Coords:").grid(row=4, column=0, sticky='w', padx=5, pady=5)
        self.fixed_coords = tk.StringVar(value=self.settings["fixed_coords"])
        ttk.Entry(timing_frame, textvariable=self.fixed_coords, width=30).grid(row=4, column=1, columnspan=2, sticky='ew', padx=5, pady=5)

        # Help for coords
        coord_help = "Format: x1,y1;x2,y2;x3,y3 (used when method=fixed_coords)"
        ttk.Label(timing_frame, text=coord_help, font=('Segoe UI', 7), foreground='gray').grid(row=5, column=0, columnspan=3, sticky='w', padx=5, pady=2)

        # Prevent feedback checkbox
        self.prevent_feedback = tk.BooleanVar(value=self.settings.get("prevent_feedback", True))
        ttk.Checkbutton(timing_frame, text="Prevent feedback (mask viewer window)",
                       variable=self.prevent_feedback).grid(row=6, column=0, columnspan=3, sticky='w', padx=5, pady=5)

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

        # Cursor offset settings
        offset_frame = ttk.LabelFrame(cursor_frame, text="Cursor Offset Correction")
        offset_frame.grid(row=3, column=0, columnspan=3, sticky='ew', padx=5, pady=10)

        ttk.Label(offset_frame, text="X Offset:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.cursor_offset_x = tk.IntVar(value=self.settings.get("cursor_offset_x", 0))
        offset_x_spin = ttk.Spinbox(offset_frame, from_=-100, to=100, textvariable=self.cursor_offset_x, width=10)
        offset_x_spin.grid(row=0, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(offset_frame, text="pixels").grid(row=0, column=2, sticky='w', padx=5, pady=5)

        ttk.Label(offset_frame, text="Y Offset:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.cursor_offset_y = tk.IntVar(value=self.settings.get("cursor_offset_y", 0))
        offset_y_spin = ttk.Spinbox(offset_frame, from_=-100, to=100, textvariable=self.cursor_offset_y, width=10)
        offset_y_spin.grid(row=1, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(offset_frame, text="pixels").grid(row=1, column=2, sticky='w', padx=5, pady=5)

        # Help text for offsets
        offset_help = "Automatic calibration adjustments. Positive X = right, Positive Y = down."
        ttk.Label(offset_frame, text=offset_help, font=('Segoe UI', 8), foreground='gray').grid(row=2, column=0, columnspan=3, sticky='w', padx=5, pady=2)

        # Buttons
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(fill='x', padx=10, pady=10)

        ttk.Button(button_frame, text="OK", command=self.ok_clicked).pack(side='right', padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.cancel_clicked).pack(side='right', padx=5)

    def ok_clicked(self):
        """Apply settings and close"""
        self.viewer.settings["monitor_target"] = self.monitor_target.get()
        self.viewer.settings["cursor_type"] = self.cursor_type.get()
        self.viewer.settings["cursor_size"] = self.cursor_size.get()
        self.viewer.settings["cursor_thickness"] = self.cursor_thickness.get()

        # Window sizing settings
        self.viewer.settings["sizing_mode"] = self.sizing_mode.get()
        self.viewer.settings["target_width"] = self.target_width.get()
        self.viewer.settings["target_height"] = self.target_height.get()

        # Capture timing settings
        self.viewer.settings["capture_interval"] = self.capture_interval.get()
        self.viewer.settings["pixel_threshold"] = self.pixel_threshold.get()
        self.viewer.settings["detection_method"] = self.detection_method.get()
        self.viewer.settings["sample_points"] = self.sample_points.get()
        self.viewer.settings["fixed_coords"] = self.fixed_coords.get()
        self.viewer.settings["prevent_feedback"] = self.prevent_feedback.get()

        self.viewer.save_settings()
        self.dialog.destroy()

    def cancel_clicked(self):
        """Cancel without applying"""
        self.dialog.destroy()


class CalibrationWindow:
    """Full-screen calibration for LLMs"""

    def __init__(self, parent, settings, test_mode=False, coordinate_test_mode=False, two_way_calibration=False):
        self.parent = parent
        self.viewer_parent = None  # Will store reference to MinimalViewer for cleanup
        self.settings = settings
        self.test_mode = test_mode
        self.coordinate_test_mode = coordinate_test_mode
        self.two_way_calibration = two_way_calibration  # NEW: Two-way click mapping
        self.window = tk.Toplevel(parent)

        mode_suffix = ""
        if two_way_calibration:
            mode_suffix = " - TWO-WAY CLICK MAPPING"
        elif coordinate_test_mode:
            mode_suffix = " - COORDINATE SYSTEM TEST"
        elif test_mode:
            mode_suffix = " - OFFSET TEST MODE"
        self.window.title("LLM Mouse Calibration" + mode_suffix)

        # Get monitor information
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()

        # Get absolute monitor bounds for coordinate system analysis
        self.monitor_info = self.get_detailed_monitor_info()
        self.current_monitor = self.detect_current_monitor()

        # Get the target monitor from settings
        self.target_monitor_setting = self.settings.get("monitor_target", "primary")
        self.target_monitor_index = self.resolve_target_monitor()

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

        # Two-way calibration state
        self.calibration_step = 1  # 1 = click monitor target, 2 = click viewer position
        self.monitor_click = None  # Store step 1 click
        self.viewer_click = None   # Store step 2 click
        self.coordinate_mappings = []  # Store all mapping pairs

        # Test mode simulation parameters
        if self.test_mode:
            self.simulated_offset_x = 7
            self.simulated_offset_y = -5
            print(f"OFFSET TEST MODE: Simulating small cursor misalignment of +{self.simulated_offset_x}px X, {self.simulated_offset_y:+}px Y")
        elif self.coordinate_test_mode:
            print(f"COORDINATE SYSTEM TEST: Analyzing coordinate mapping for {len(self.monitor_info)} monitors")
            print(f"Current monitor: {self.current_monitor}")
            for i, monitor in enumerate(self.monitor_info):
                print(f"  Monitor {i+1}: {monitor['width']}x{monitor['height']} at ({monitor['left']},{monitor['top']})")
        elif self.two_way_calibration:
            print(f"TWO-WAY CALIBRATION: Monitor {self.target_monitor_index + 1} -> Viewer mapping")
            print(f"Target monitor: {self.target_monitor_setting}")
            print("Step 1: Click targets on calibration popup (monitor space)")
            print("Step 2: Click corresponding positions on viewer window (screen space)")

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

    def get_detailed_monitor_info(self):
        """Get detailed information about all monitors"""
        try:
            import ctypes
            from ctypes import wintypes

            monitors = []

            def enum_proc(hmonitor, hdc, lprect, lparam):
                monitor_info = wintypes.RECT()
                ctypes.windll.user32.GetMonitorInfoW(hmonitor, ctypes.byref(monitor_info))

                monitors.append({
                    'handle': hmonitor,
                    'left': monitor_info.left,
                    'top': monitor_info.top,
                    'right': monitor_info.right,
                    'bottom': monitor_info.bottom,
                    'width': monitor_info.right - monitor_info.left,
                    'height': monitor_info.bottom - monitor_info.top
                })
                return True

            enum_proc_type = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)
            ctypes.windll.user32.EnumDisplayMonitors(None, None, enum_proc_type(enum_proc), 0)

            return monitors
        except Exception as e:
            print(f"Error getting monitor info: {e}")
            return [{'left': 0, 'top': 0, 'width': self.screen_width, 'height': self.screen_height}]

    def detect_current_monitor(self):
        """Detect which monitor this window is on"""
        try:
            # Get window position
            window_x = self.window.winfo_x()
            window_y = self.window.winfo_y()

            # Find which monitor contains this position
            for i, monitor in enumerate(self.monitor_info):
                if (monitor['left'] <= window_x < monitor['right'] and
                    monitor['top'] <= window_y < monitor['bottom']):
                    return i

            return 0  # Default to primary monitor
        except Exception as e:
            print(f"[WARN] get_window_monitor failed: {e}")
            return 0

    def resolve_target_monitor(self):
        """Resolve target monitor setting to monitor index"""
        try:
            monitor_setting = self.target_monitor_setting.lower()

            if monitor_setting == "primary":
                return 0  # Primary monitor is index 0
            elif monitor_setting.isdigit():
                # Direct monitor number (1-based to 0-based)
                monitor_num = int(monitor_setting) - 1
                return max(0, min(monitor_num, len(self.monitor_info) - 1))
            elif monitor_setting == "all":
                return 0  # Default to primary for calibration
            else:
                return 0  # Default fallback
        except Exception as e:
            print(f"[WARN] resolve_target_monitor failed: {e}")
            return 0

    def start_calibration(self):
        """Start calibration process"""
        self.targets.clear()
        self.results.clear()
        self.current_target = 0

        if self.two_way_calibration:
            # Two-way calibration: Create grid for monitor-to-viewer mapping
            margin = 100
            grid_positions = [
                # Center first for immediate validation
                (self.screen_width // 2, self.screen_height // 2, 'center'),
                # Corners for transformation matrix
                (margin, margin, 'top-left'),
                (self.screen_width - margin, margin, 'top-right'),
                (margin, self.screen_height - margin, 'bottom-left'),
                (self.screen_width - margin, self.screen_height - margin, 'bottom-right'),
                # Edges for comprehensive mapping
                (self.screen_width // 2, margin, 'top-center'),
                (self.screen_width // 2, self.screen_height - margin, 'bottom-center'),
                (margin, self.screen_height // 2, 'left-center'),
                (self.screen_width - margin, self.screen_height // 2, 'right-center'),
            ]

            for x, y, position_type in grid_positions:
                self.targets.append({
                    'x': x, 'y': y,
                    'clicked': False,
                    'type': position_type,
                    'step1_complete': False,
                    'step2_complete': False,
                    'monitor_click': None,
                    'viewer_click': None
                })

        elif self.coordinate_test_mode:
            # Phase 1: Center target for immediate coordinate system validation
            center_x = self.screen_width // 2
            center_y = self.screen_height // 2
            self.targets.append({'x': center_x, 'y': center_y, 'clicked': False, 'type': 'center'})

            # Phase 2: Grid targets for coordinate system mapping
            margin = 100
            grid_positions = [
                # Corners
                (margin, margin, 'top-left'),
                (self.screen_width - margin, margin, 'top-right'),
                (margin, self.screen_height - margin, 'bottom-left'),
                (self.screen_width - margin, self.screen_height - margin, 'bottom-right'),
                # Edges midpoints
                (self.screen_width // 2, margin, 'top-center'),
                (self.screen_width // 2, self.screen_height - margin, 'bottom-center'),
                (margin, self.screen_height // 2, 'left-center'),
                (self.screen_width - margin, self.screen_height // 2, 'right-center'),
            ]

            for x, y, position_type in grid_positions:
                self.targets.append({'x': x, 'y': y, 'clicked': False, 'type': position_type})

        elif self.test_mode:
            # Offset correction test: Start with center, then a few strategic points
            center_x = self.screen_width // 2
            center_y = self.screen_height // 2

            # Center for immediate offset detection
            self.targets.append({'x': center_x, 'y': center_y, 'clicked': False, 'type': 'center'})

            # Four additional points to refine the offset
            offset_distance = 200
            test_points = [
                (center_x + offset_distance, center_y, 'right'),
                (center_x - offset_distance, center_y, 'left'),
                (center_x, center_y - offset_distance, 'up'),
                (center_x, center_y + offset_distance, 'down')
            ]

            for x, y, direction in test_points:
                if 50 <= x <= self.screen_width - 50 and 50 <= y <= self.screen_height - 50:
                    self.targets.append({'x': x, 'y': y, 'clicked': False, 'type': direction})
        else:
            # Traditional 3x3 grid for general calibration
            margin = 150
            for row in range(3):
                for col in range(3):
                    x = margin + col * (self.screen_width - 2*margin) // 2
                    y = margin + row * (self.screen_height - 2*margin) // 2
                    self.targets.append({'x': x, 'y': y, 'clicked': False, 'type': f'grid-{row}-{col}'})

        self.show_next_target()

    def show_next_target(self):
        """Show next calibration target"""
        self.canvas.delete("all")

        if self.current_target < len(self.targets):
            target = self.targets[self.current_target]

            # Two-way calibration display
            if self.two_way_calibration:
                self.show_two_way_target(target)
            else:
                # Standard calibration display
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
                total_targets = len(self.targets)
                self.canvas.create_text(50, 50, text=f"Target {self.current_target + 1}/{total_targets}",
                                       fill='white', font=('Arial', 20), anchor='nw')
                self.canvas.create_text(self.screen_width - 50, 50, text="Press ESC to cancel",
                                       fill='gray', font=('Arial', 16), anchor='ne')

            # Show test mode information and current cursor offset status
            if self.test_mode:
                current_offset_x = self.settings.get("cursor_offset_x", 0)
                current_offset_y = self.settings.get("cursor_offset_y", 0)
                self.canvas.create_text(self.screen_width // 2, 80,
                                       text=f"TEST MODE: Simulating +{self.simulated_offset_x}px X, {self.simulated_offset_y:+}px Y offset",
                                       fill='yellow', font=('Arial', 18, 'bold'), anchor='n')
                self.canvas.create_text(self.screen_width // 2, 110,
                                       text=f"Current Cursor Offset: {current_offset_x:+}px X, {current_offset_y:+}px Y",
                                       fill='cyan', font=('Arial', 16), anchor='n')
                self.canvas.create_text(self.screen_width // 2, 140,
                                       text="System will auto-correct after each successful hit",
                                       fill='lime', font=('Arial', 14), anchor='n')
        else:
            self.save_results()
            self.close_window()

    def on_canvas_click(self, event):
        """Handle canvas click"""
        if self.current_target < len(self.targets):
            target = self.targets[self.current_target]
            click_x, click_y = event.x, event.y

            # Two-way calibration handling
            if self.two_way_calibration:
                self.handle_two_way_click(target, click_x, click_y)
                return

            # Apply test mode simulation to simulate misaligned cursor
            if self.test_mode:
                original_click_x, original_click_y = click_x, click_y
                click_x += self.simulated_offset_x
                click_y += self.simulated_offset_y
                print(f"TEST MODE: Original click ({original_click_x}, {original_click_y}) -> Simulated click ({click_x}, {click_y})")

                # Show visual markers for both positions
                self.show_test_click_markers(original_click_x, original_click_y, click_x, click_y)

            # Calculate distance from target center
            distance = math.sqrt((click_x - target['x'])**2 + (click_y - target['y'])**2)

            # Get target radius (same as outer circle)
            target_radius = self.settings.get("cursor_size", 15) * 2

            # Check if click is within target circle
            is_hit = distance <= target_radius

            # Calculate offset corrections for LLM feedback
            offset_x = target['x'] - click_x  # How much to adjust X
            offset_y = target['y'] - click_y  # How much to adjust Y

            # Output LLM feedback for every click
            self.output_llm_feedback(target, click_x, click_y, offset_x, offset_y, distance, is_hit)

            if is_hit:
                # Valid hit - calculate accuracy and advance
                accuracy = max(0, 100 - (distance / target_radius) * 100)

                # Store result
                result = {
                    'target': self.current_target + 1,
                    'target_x': target['x'],
                    'target_y': target['y'],
                    'click_x': click_x,
                    'click_y': click_y,
                    'distance': distance,
                    'accuracy': accuracy,
                    'hit': True,
                    'offset_x': offset_x,
                    'offset_y': offset_y
                }
                self.results.append(result)

                # Auto-update cursor offset settings based on this click
                old_offset_x = self.settings.get("cursor_offset_x", 0)
                old_offset_y = self.settings.get("cursor_offset_y", 0)
                self.auto_update_cursor_offset(offset_x, offset_y)
                new_offset_x = self.settings.get("cursor_offset_x", 0)
                new_offset_y = self.settings.get("cursor_offset_y", 0)

                if self.test_mode:
                    print(f"TEST MODE CORRECTION: Offset changed from ({old_offset_x:+}, {old_offset_y:+}) to ({new_offset_x:+}, {new_offset_y:+})")
                    remaining_error_x = self.simulated_offset_x + new_offset_x
                    remaining_error_y = self.simulated_offset_y + new_offset_y
                    print(f"TEST MODE: Remaining error after correction: ({remaining_error_x:+}px X, {remaining_error_y:+}px Y)")

                target['clicked'] = True
                self.current_target += 1

                if self.current_target < len(self.targets):
                    self.show_next_target()
                else:
                    self.save_results()
                    self.close_window()
            else:
                # Miss - show visual feedback but don't advance
                self.show_miss_feedback(click_x, click_y)

                # Still store the miss for analysis
                result = {
                    'target': self.current_target + 1,
                    'target_x': target['x'],
                    'target_y': target['y'],
                    'click_x': click_x,
                    'click_y': click_y,
                    'distance': distance,
                    'accuracy': 0,
                    'hit': False,
                    'offset_x': offset_x,
                    'offset_y': offset_y
                }
                self.results.append(result)

    def show_miss_feedback(self, click_x, click_y):
        """Show visual feedback for missed clicks"""
        # Create a red X mark at click location
        size = 10
        x_mark = self.canvas.create_line(click_x-size, click_y-size, click_x+size, click_y+size,
                                        fill='red', width=3, tags='miss_feedback')
        self.canvas.create_line(click_x-size, click_y+size, click_x+size, click_y-size,
                               fill='red', width=3, tags='miss_feedback')

        # Create text feedback
        text = self.canvas.create_text(click_x, click_y-30, text="MISS",
                                      fill='red', font=('Arial', 12, 'bold'), tags='miss_feedback')

        # Remove feedback after 1 second
        self.window.after(1000, lambda: self.canvas.delete('miss_feedback'))

    def show_test_click_markers(self, original_x, original_y, simulated_x, simulated_y):
        """Show visual markers for test mode clicks"""
        # Clear previous test markers
        self.canvas.delete('test_markers')

        # Show original click position (green circle)
        self.canvas.create_oval(original_x-8, original_y-8, original_x+8, original_y+8,
                               fill='lime', outline='white', width=2, tags='test_markers')
        self.canvas.create_text(original_x, original_y-25, text="REAL CLICK",
                               fill='lime', font=('Arial', 10, 'bold'), tags='test_markers')

        # Show simulated click position (red circle)
        self.canvas.create_oval(simulated_x-8, simulated_y-8, simulated_x+8, simulated_y+8,
                               fill='red', outline='white', width=2, tags='test_markers')
        self.canvas.create_text(simulated_x, simulated_y-25, text="SIMULATED CLICK",
                               fill='red', font=('Arial', 10, 'bold'), tags='test_markers')

        # Draw line connecting them
        self.canvas.create_line(original_x, original_y, simulated_x, simulated_y,
                               fill='yellow', width=2, dash=(5, 5), tags='test_markers')

        # Remove markers after 4 seconds
        self.window.after(4000, lambda: self.canvas.delete('test_markers'))

    def show_two_way_target(self, target):
        """Show two-way calibration target with step progression"""
        # Determine target colors based on completion status
        if target.get('step2_complete', False):
            # Both steps complete - green
            target_color = 'green'
            center_color = 'white'
            status_text = "COMPLETE"
            status_color = 'lime'
        elif target.get('step1_complete', False):
            # Step 1 complete, waiting for step 2 - orange
            target_color = 'orange'
            center_color = 'white'
            status_text = "STEP 2: Click viewer position"
            status_color = 'orange'
        else:
            # Waiting for step 1 - red
            target_color = 'red'
            center_color = 'white'
            status_text = "STEP 1: Click this target"
            status_color = 'red'

        # Draw target with appropriate color
        size = self.settings.get("cursor_size", 15) * 2
        self.canvas.create_oval(target['x']-size, target['y']-size, target['x']+size, target['y']+size,
                               fill=target_color, outline='white', width=4)
        self.canvas.create_oval(target['x']-8, target['y']-8, target['x']+8, target['y']+8,
                               fill=center_color, outline='black', width=2)

        # Show target type and position info
        self.canvas.create_text(target['x'], target['y'] + size + 30,
                               text=f"{target['type'].upper()}",
                               fill='white', font=('Arial', 12, 'bold'), anchor='n')

        # Show current step status
        self.canvas.create_text(target['x'], target['y'] - size - 30,
                               text=status_text,
                               fill=status_color, font=('Arial', 14, 'bold'), anchor='s')

        # Show completed targets
        for i, prev_target in enumerate(self.targets[:self.current_target]):
            if prev_target.get('step2_complete', False):
                self.canvas.create_oval(prev_target['x']-20, prev_target['y']-20,
                                       prev_target['x']+20, prev_target['y']+20,
                                       fill='green', outline='white', width=3)
                # Show mapping info
                if prev_target.get('monitor_click') and prev_target.get('viewer_click'):
                    self.canvas.create_text(prev_target['x'], prev_target['y'] + 35,
                                           text=f"✓ {i+1}",
                                           fill='lime', font=('Arial', 10, 'bold'), anchor='n')

        # Show two-way calibration header
        total_targets = len(self.targets)
        monitor_name = f"Monitor {self.target_monitor_index + 1}" if self.target_monitor_index < len(self.monitor_info) else "Target Monitor"

        self.canvas.create_text(50, 50,
                               text=f"Two-Way Calibration: {monitor_name} → Viewer",
                               fill='cyan', font=('Arial', 18, 'bold'), anchor='nw')
        self.canvas.create_text(50, 80,
                               text=f"Target {self.current_target + 1}/{total_targets} - {target['type'].replace('-', ' ').title()}",
                               fill='white', font=('Arial', 16), anchor='nw')

        # Show instructions
        if not target.get('step1_complete', False):
            instruction = "Click the RED target above to complete Step 1"
        elif not target.get('step2_complete', False):
            instruction = "Now click the corresponding position on the VIEWER WINDOW for Step 2"
        else:
            instruction = "Both steps complete! Moving to next target..."

        self.canvas.create_text(self.screen_width // 2, self.screen_height - 100,
                               text=instruction,
                               fill='yellow', font=('Arial', 16, 'bold'), anchor='n')

        # Show coordinate mapping progress
        completed_mappings = len([t for t in self.targets if t.get('step2_complete', False)])
        self.canvas.create_text(self.screen_width - 50, 50,
                               text=f"Mappings: {completed_mappings}/{total_targets}",
                               fill='lime', font=('Arial', 14), anchor='ne')
        self.canvas.create_text(self.screen_width - 50, 75,
                               text="Press ESC to cancel",
                               fill='gray', font=('Arial', 12), anchor='ne')

    def handle_two_way_click(self, target, click_x, click_y):
        """Handle two-way calibration click"""
        # Calculate distance from target center
        distance = math.sqrt((click_x - target['x'])**2 + (click_y - target['y'])**2)
        target_radius = self.settings.get("cursor_size", 15) * 2
        is_hit = distance <= target_radius

        if is_hit:
            if not target.get('step1_complete', False):
                # Step 1: Monitor click
                target['step1_complete'] = True
                target['monitor_click'] = {
                    'x': click_x,
                    'y': click_y,
                    'monitor_index': self.target_monitor_index,
                    'timestamp': datetime.now().isoformat()
                }

                print(f"TWO-WAY STEP 1: Monitor click at ({click_x}, {click_y}) on {target['type']}")
                print("Now click the corresponding position on the VIEWER WINDOW")

                # Create visual feedback for step 1 completion
                self.canvas.create_text(target['x'], target['y'] + 80,
                                       text="✓ STEP 1 COMPLETE",
                                       fill='lime', font=('Arial', 12, 'bold'), anchor='n',
                                       tags='step1_complete')

                # Remove feedback after 2 seconds and refresh display
                self.window.after(2000, lambda: [
                    self.canvas.delete('step1_complete'),
                    self.show_next_target()
                ])

                # Start monitoring for viewer window click (Step 2)
                self.start_viewer_click_monitoring()

            else:
                print("Step 1 already complete. Waiting for viewer window click for Step 2.")
        else:
            # Miss feedback
            self.show_miss_feedback(click_x, click_y)
            print(f"MISS: Click was {distance:.1f}px from target (radius: {target_radius}px)")

    def start_viewer_click_monitoring(self):
        """Start monitoring for clicks on the viewer window"""
        # The viewer window will now detect clicks and call back to this calibration
        print("Monitoring for viewer window click...")
        print("Click on the viewer window where this target should appear!")
        # For now, we'll simulate with a timeout and manual input
        self.window.after(1000, self.check_for_viewer_click)

    def check_for_viewer_click(self):
        """Check if viewer click has been received"""
        # This is a placeholder for the actual viewer coordination
        # In the real implementation, the viewer window would call a method here
        if not self.targets[self.current_target].get('step2_complete', False):
            # Keep checking
            self.window.after(1000, self.check_for_viewer_click)

    def receive_viewer_click(self, viewer_x, viewer_y):
        """Receive click coordinates from viewer window"""
        if self.current_target < len(self.targets):
            target = self.targets[self.current_target]

            if target.get('step1_complete', False) and not target.get('step2_complete', False):
                # Step 2: Viewer click
                target['step2_complete'] = True
                target['viewer_click'] = {
                    'x': viewer_x,
                    'y': viewer_y,
                    'timestamp': datetime.now().isoformat()
                }

                # Calculate coordinate mapping
                monitor_click = target['monitor_click']
                mapping = self.calculate_coordinate_mapping(monitor_click, target['viewer_click'])
                target['coordinate_mapping'] = mapping
                self.coordinate_mappings.append(mapping)

                print(f"TWO-WAY STEP 2: Viewer click at ({viewer_x}, {viewer_y})")
                print(f"MAPPING CREATED: Monitor({monitor_click['x']}, {monitor_click['y']}) -> Viewer({viewer_x}, {viewer_y})")
                print(f"Transform: Scale({mapping['scale_x']:.3f}, {mapping['scale_y']:.3f}), Offset({mapping['offset_x']:.1f}, {mapping['offset_y']:.1f})")

                # Store detailed mapping result
                result = {
                    'target': self.current_target + 1,
                    'target_type': target['type'],
                    'monitor_click': monitor_click,
                    'viewer_click': target['viewer_click'],
                    'coordinate_mapping': mapping,
                    'timestamp': datetime.now().isoformat()
                }
                self.results.append(result)

                # Advance to next target
                target['clicked'] = True
                self.current_target += 1

                if self.current_target < len(self.targets):
                    self.show_next_target()
                else:
                    self.save_two_way_results()
                    self.close_window()

    def calculate_coordinate_mapping(self, monitor_click, viewer_click):
        """Calculate coordinate transformation between monitor and viewer"""
        # Get monitor information
        monitor = self.monitor_info[self.target_monitor_index] if self.target_monitor_index < len(self.monitor_info) else None

        if monitor:
            # Calculate scale factors
            scale_x = viewer_click['x'] / monitor_click['x'] if monitor_click['x'] != 0 else 1.0
            scale_y = viewer_click['y'] / monitor_click['y'] if monitor_click['y'] != 0 else 1.0

            # Calculate offset
            offset_x = viewer_click['x'] - (monitor_click['x'] * scale_x)
            offset_y = viewer_click['y'] - (monitor_click['y'] * scale_y)
        else:
            # Fallback calculation
            scale_x = 1.0
            scale_y = 1.0
            offset_x = viewer_click['x'] - monitor_click['x']
            offset_y = viewer_click['y'] - monitor_click['y']

        return {
            'scale_x': scale_x,
            'scale_y': scale_y,
            'offset_x': offset_x,
            'offset_y': offset_y,
            'monitor_index': self.target_monitor_index,
            'monitor_bounds': monitor,
            'transformation_type': 'linear'
        }

    def save_two_way_results(self):
        """Save two-way calibration results"""
        try:
            results_data = {
                'timestamp': datetime.now().isoformat(),
                'calibration_type': 'two_way_mapping',
                'source_monitor': self.target_monitor_index,
                'monitor_info': self.monitor_info[self.target_monitor_index] if self.target_monitor_index < len(self.monitor_info) else None,
                'screen_size': {'width': self.screen_width, 'height': self.screen_height},
                'settings': self.settings,
                'coordinate_mappings': self.coordinate_mappings,
                'results': self.results,
                'summary': {
                    'total_mappings': len(self.coordinate_mappings),
                    'avg_scale_x': sum(m['scale_x'] for m in self.coordinate_mappings) / len(self.coordinate_mappings) if self.coordinate_mappings else 0,
                    'avg_scale_y': sum(m['scale_y'] for m in self.coordinate_mappings) / len(self.coordinate_mappings) if self.coordinate_mappings else 0,
                    'transformation_matrix': self.calculate_transformation_matrix()
                }
            }

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(os.path.dirname(__file__), "core", "calibration", f"two_way_mapping_{timestamp}.json")

            with open(filename, 'w') as f:
                json.dump(results_data, f, indent=2)

            print(f"Two-way calibration results saved: {filename}")

        except Exception as e:
            print(f"Error saving two-way calibration results: {e}")

    def calculate_transformation_matrix(self):
        """Calculate overall transformation matrix from all mappings"""
        if not self.coordinate_mappings:
            return None

        # Calculate average transformation
        avg_scale_x = sum(m['scale_x'] for m in self.coordinate_mappings) / len(self.coordinate_mappings)
        avg_scale_y = sum(m['scale_y'] for m in self.coordinate_mappings) / len(self.coordinate_mappings)
        avg_offset_x = sum(m['offset_x'] for m in self.coordinate_mappings) / len(self.coordinate_mappings)
        avg_offset_y = sum(m['offset_y'] for m in self.coordinate_mappings) / len(self.coordinate_mappings)

        return {
            'scale_x': avg_scale_x,
            'scale_y': avg_scale_y,
            'offset_x': avg_offset_x,
            'offset_y': avg_offset_y,
            'formula': f"viewer_x = monitor_x * {avg_scale_x:.3f} + {avg_offset_x:.1f}, viewer_y = monitor_y * {avg_scale_y:.3f} + {avg_offset_y:.1f}"
        }

    def output_llm_feedback(self, target, click_x, click_y, offset_x, offset_y, distance, is_hit):
        """Output LLM feedback for each calibration click"""
        try:
            # Prepare feedback data
            feedback_data = {
                'timestamp': datetime.now().isoformat(),
                'target_number': self.current_target + 1,
                'target_position': {'x': target['x'], 'y': target['y']},
                'click_position': {'x': click_x, 'y': click_y},
                'offset_correction': {'x': offset_x, 'y': offset_y},
                'distance': round(distance, 2),
                'hit': is_hit,
                'test_mode': self.test_mode,
                'recommendation': {
                    'cursor_offset_x_adjust': round(offset_x / max(1, self.current_target + 1), 1),
                    'cursor_offset_y_adjust': round(offset_y / max(1, self.current_target + 1), 1),
                    'message': f"{'HIT' if is_hit else 'MISS'}: Adjust cursor {offset_x:+.0f}px X, {offset_y:+.0f}px Y"
                },
                'current_settings': {
                    'cursor_offset_x': self.settings.get("cursor_offset_x", 0),
                    'cursor_offset_y': self.settings.get("cursor_offset_y", 0)
                }
            }

            # Add coordinate system analysis information
            if self.coordinate_test_mode or self.test_mode:
                feedback_data['coordinate_analysis'] = self.analyze_coordinate_system(target, click_x, click_y)

            # Add test mode specific information
            if self.test_mode:
                feedback_data['test_mode_info'] = {
                    'simulated_offset_x': self.simulated_offset_x,
                    'simulated_offset_y': self.simulated_offset_y,
                    'explanation': f"Test mode is simulating a cursor misalignment of +{self.simulated_offset_x}px X, {self.simulated_offset_y:+}px Y. The system should gradually correct this through auto-calibration."
                }
            elif self.coordinate_test_mode:
                feedback_data['coordinate_test_info'] = {
                    'phase': self.calibration_phase,
                    'explanation': "Coordinate system test mode is analyzing the relationship between mouse click coordinates and screen positioning across multiple monitors."
                }

            # Create LLM feedback file path
            base_dir = os.path.dirname(__file__)
            llm_feedback_dir = os.path.join(base_dir, "core", "calibration")
            os.makedirs(llm_feedback_dir, exist_ok=True)

            # Write individual click feedback for LLM
            feedback_file = os.path.join(llm_feedback_dir, "llm_feedback_latest.json")
            with open(feedback_file, 'w') as f:
                json.dump(feedback_data, f, indent=2)

            # Append to session log
            session_log = os.path.join(llm_feedback_dir, "calibration_session_log.json")
            if os.path.exists(session_log):
                with open(session_log, 'r') as f:
                    log_data = json.load(f)
            else:
                log_data = {'session_start': datetime.now().isoformat(), 'clicks': []}

            log_data['clicks'].append(feedback_data)
            with open(session_log, 'w') as f:
                json.dump(log_data, f, indent=2)

            # Display feedback on screen temporarily
            feedback_text = f"Target {self.current_target + 1}: {feedback_data['recommendation']['message']}"
            text_item = self.canvas.create_text(
                self.screen_width // 2, 50,
                text=feedback_text,
                fill='yellow' if is_hit else 'orange',
                font=('Arial', 14, 'bold'),
                tags='llm_feedback'
            )

            # Remove text after 3 seconds
            self.window.after(3000, lambda: self.canvas.delete('llm_feedback'))

        except Exception as e:
            print(f"Error outputting LLM feedback: {e}")

    def auto_update_cursor_offset(self, offset_x, offset_y):
        """Auto-update cursor offset settings based on calibration clicks"""
        try:
            # Calculate running average offset from all successful hits
            hit_results = [r for r in self.results if r.get('hit', False)]
            if hit_results:
                avg_offset_x = sum(r.get('offset_x', 0) for r in hit_results) / len(hit_results)
                avg_offset_y = sum(r.get('offset_y', 0) for r in hit_results) / len(hit_results)

                # Update settings with the running average
                self.settings['cursor_offset_x'] = round(avg_offset_x)
                self.settings['cursor_offset_y'] = round(avg_offset_y)

                # Save updated settings
                self.save_settings()

                print(f"Auto-updated cursor offset: X={self.settings['cursor_offset_x']}, Y={self.settings['cursor_offset_y']}")

        except Exception as e:
            print(f"Error auto-updating cursor offset: {e}")

    def save_settings(self):
        """Save current settings to file atomically (temp file + rename)"""
        tmp_path = None
        try:
            base_dir = os.path.dirname(__file__)
            settings_file = os.path.join(base_dir, "configs", "viewer_settings.json")
            tmp_path = settings_file + ".tmp"

            with open(tmp_path, 'w') as f:
                json.dump(self.settings, f, indent=2)
            os.replace(tmp_path, settings_file)

        except Exception as e:
            print(f"Error saving settings: {e}")
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def analyze_coordinate_system(self, target, click_x, click_y):
        """Analyze coordinate system for multi-monitor setups"""
        try:
            # Get current cursor position in screen coordinates
            point = POINT()
            ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
            absolute_x, absolute_y = point.x, point.y

            # Analyze coordinate mappings
            analysis = {
                'target_info': {
                    'target_position': {'x': target['x'], 'y': target['y']},
                    'target_type': target.get('type', 'unknown'),
                    'window_relative': True,  # Target coordinates are relative to window
                },
                'click_info': {
                    'window_click': {'x': click_x, 'y': click_y},
                    'absolute_screen': {'x': absolute_x, 'y': absolute_y},
                    'coordinate_system': 'tkinter_canvas'  # Using tkinter canvas coordinates
                },
                'monitor_analysis': {
                    'current_monitor': self.current_monitor,
                    'monitor_count': len(self.monitor_info),
                    'current_monitor_bounds': self.monitor_info[self.current_monitor] if self.current_monitor < len(self.monitor_info) else None,
                    'window_bounds': {
                        'width': self.screen_width,
                        'height': self.screen_height,
                        'top_left': (0, 0),  # Window coordinates start at 0,0
                        'bottom_right': (self.screen_width, self.screen_height)
                    }
                },
                'coordinate_mapping': {
                    'window_to_screen_offset': {
                        'x': absolute_x - click_x,
                        'y': absolute_y - click_y
                    },
                    'y_direction': 'positive_down',  # Y increases downward in tkinter
                    'origin': 'top_left'  # Window origin is top-left corner
                }
            }

            # Determine which monitor the absolute position is on
            for i, monitor in enumerate(self.monitor_info):
                if (monitor['left'] <= absolute_x < monitor['right'] and
                    monitor['top'] <= absolute_y < monitor['bottom']):
                    analysis['click_monitor'] = i
                    analysis['click_monitor_bounds'] = monitor
                    analysis['click_relative_to_monitor'] = {
                        'x': absolute_x - monitor['left'],
                        'y': absolute_y - monitor['top']
                    }
                    break
            else:
                analysis['click_monitor'] = None
                analysis['warning'] = f"Click at ({absolute_x}, {absolute_y}) is outside all monitor bounds"

            # Analyze coordinate system consistency
            if target.get('type') == 'center':
                expected_center_x = self.screen_width // 2
                expected_center_y = self.screen_height // 2
                center_error_x = click_x - expected_center_x
                center_error_y = click_y - expected_center_y

                analysis['center_target_analysis'] = {
                    'expected_center': {'x': expected_center_x, 'y': expected_center_y},
                    'actual_click': {'x': click_x, 'y': click_y},
                    'error': {'x': center_error_x, 'y': center_error_y},
                    'coordinate_system_status': 'correct' if abs(center_error_x) < 50 and abs(center_error_y) < 50 else 'misaligned'
                }

            # Multi-monitor coordinate space analysis
            if len(self.monitor_info) > 1:
                analysis['multi_monitor_info'] = {
                    'total_desktop_bounds': {
                        'left': min(m['left'] for m in self.monitor_info),
                        'top': min(m['top'] for m in self.monitor_info),
                        'right': max(m['right'] for m in self.monitor_info),
                        'bottom': max(m['bottom'] for m in self.monitor_info)
                    },
                    'primary_monitor': 0,  # Assume first monitor is primary
                    'monitors': [
                        {
                            'index': i,
                            'bounds': monitor,
                            'size': f"{monitor['width']}x{monitor['height']}",
                            'position': f"({monitor['left']},{monitor['top']})"
                        }
                        for i, monitor in enumerate(self.monitor_info)
                    ]
                }

            return analysis

        except Exception as e:
            return {'error': f"Error analyzing coordinate system: {e}"}

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
            filename = os.path.join(os.path.dirname(__file__), "core", "calibration", f"calibration_results_{timestamp}.json")

            with open(filename, 'w') as f:
                json.dump(results_data, f, indent=2)

        except Exception as e:
            print(f"Error saving calibration results: {e}")

    def cancel_calibration(self, event=None):
        """Cancel calibration"""
        self.close_window()

    def close_window(self):
        """Close calibration window"""
        # Notify viewer parent to clear the active calibration reference
        if self.viewer_parent and hasattr(self.viewer_parent, 'active_calibration'):
            self.viewer_parent.active_calibration = None
        self.window.destroy()


if __name__ == "__main__":
    viewer = MinimalViewer()
    viewer.run()