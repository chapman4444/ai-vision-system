"""
AI Vision System Control Panel
Main GUI interface for controlling screen capture and vision processing.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
from typing import Optional, Dict, Any
import json
from PIL import Image, ImageTk
import numpy as np

# Import our modules (adjust paths as needed)
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from capture.gdi_screen_capture import GDIScreenCapture
    from monitors.monitor_manager import MonitorManager
    from regions.region_selector import RegionSelector, Region
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all modules are in the correct directories")


class VisionControlPanel:
    """Main control panel for AI vision system."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI Vision System Control Panel")
        self.root.geometry("800x600")
        
        # Initialize components
        self.screen_capture = None
        self.monitor_manager = None
        self.region_selector = None
        self.capture_active = False
        self.capture_thread = None
        
        # GUI variables
        self.selected_monitor = tk.StringVar(value="0")
        self.capture_interval = tk.DoubleVar(value=0.1)  # 100ms default
        self.save_captures = tk.BooleanVar(value=False)
        self.current_region = tk.StringVar(value="Full Screen")
        
        self._init_components()
        self._create_gui()
        self._refresh_monitors()
    
    def _init_components(self):
        """Initialize the core components."""
        try:
            self.screen_capture = GDIScreenCapture()
            self.monitor_manager = MonitorManager()
            self.region_selector = RegionSelector(self.screen_capture, self.monitor_manager)
            self.status_text = "Components initialized successfully"
        except Exception as e:
            self.status_text = f"Error initializing components: {e}"
    
    def _create_gui(self):
        """Create the main GUI interface."""
        # Main notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Monitor Selection Tab
        self._create_monitor_tab(notebook)
        
        # Region Selection Tab
        self._create_region_tab(notebook)
        
        # Capture Control Tab
        self._create_capture_tab(notebook)
        
        # Status bar
        self.status_bar = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self._update_status("System ready")
    
    def _create_monitor_tab(self, parent):
        """Create monitor selection and info tab."""
        monitor_frame = ttk.Frame(parent)
        parent.add(monitor_frame, text="Monitors")
        
        # Monitor selection
        ttk.Label(monitor_frame, text="Select Monitor:").pack(pady=5)
        
        self.monitor_combo = ttk.Combobox(monitor_frame, textvariable=self.selected_monitor)
        self.monitor_combo.pack(pady=5)
        self.monitor_combo.bind('<<ComboboxSelected>>', self._on_monitor_selected)
        
        ttk.Button(monitor_frame, text="Refresh Monitors", 
                  command=self._refresh_monitors).pack(pady=5)
        
        # Monitor info display
        self.monitor_info_text = tk.Text(monitor_frame, height=15, width=70)
        scrollbar = ttk.Scrollbar(monitor_frame, orient=tk.VERTICAL, command=self.monitor_info_text.yview)
        self.monitor_info_text.configure(yscrollcommand=scrollbar.set)
        
        info_frame = tk.Frame(monitor_frame)
        info_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.monitor_info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Test capture button
        ttk.Button(monitor_frame, text="Test Monitor Capture", 
                  command=self._test_monitor_capture).pack(pady=5)
    
    def _create_region_tab(self, parent):
        """Create region selection and management tab."""
        region_frame = ttk.Frame(parent)
        parent.add(region_frame, text="Regions")
        
        # Region selection tools
        tools_frame = ttk.Frame(region_frame)
        tools_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(tools_frame, text="Select New Region", 
                  command=self._select_new_region).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(tools_frame, text="Load Regions", 
                  command=self._load_regions).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(tools_frame, text="Save Regions", 
                  command=self._save_regions).pack(side=tk.LEFT, padx=5)
        
        # Region list
        ttk.Label(region_frame, text="Defined Regions:").pack(pady=(10,5))
        
        list_frame = tk.Frame(region_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.region_listbox = tk.Listbox(list_frame, height=10)
        region_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, 
                                        command=self.region_listbox.yview)
        self.region_listbox.configure(yscrollcommand=region_scrollbar.set)
        
        self.region_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        region_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Region controls
        region_controls = ttk.Frame(region_frame)
        region_controls.pack(fill=tk.X, pady=5)
        
        ttk.Button(region_controls, text="Preview Region", 
                  command=self._preview_selected_region).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(region_controls, text="Delete Region", 
                  command=self._delete_selected_region).pack(side=tk.LEFT, padx=5)
    
    def _create_capture_tab(self, parent):
        """Create capture control and monitoring tab."""
        capture_frame = ttk.Frame(parent)
        parent.add(capture_frame, text="Capture Control")
        
        # Capture settings
        settings_frame = ttk.LabelFrame(capture_frame, text="Capture Settings")
        settings_frame.pack(fill=tk.X, pady=5)
        
        # Interval setting
        ttk.Label(settings_frame, text="Capture Interval (seconds):").grid(row=0, column=0, padx=5, pady=5)
        interval_spin = ttk.Spinbox(settings_frame, from_=0.01, to=10.0, increment=0.01, 
                                   textvariable=self.capture_interval, width=10)
        interval_spin.grid(row=0, column=1, padx=5, pady=5)
        
        # Save captures checkbox
        ttk.Checkbutton(settings_frame, text="Save Captures to Disk", 
                       variable=self.save_captures).grid(row=1, column=0, columnspan=2, pady=5)
        
        # Capture controls
        controls_frame = ttk.LabelFrame(capture_frame, text="Controls")
        controls_frame.pack(fill=tk.X, pady=5)
        
        self.start_button = ttk.Button(controls_frame, text="Start Continuous Capture", 
                                      command=self._start_capture)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.stop_button = ttk.Button(controls_frame, text="Stop Capture", 
                                     command=self._stop_capture, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        ttk.Button(controls_frame, text="Single Capture", 
                  command=self._single_capture).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Capture info
        info_frame = ttk.LabelFrame(capture_frame, text="Capture Information")
        info_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.capture_info_text = tk.Text(info_frame, height=10)
        capture_scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, 
                                         command=self.capture_info_text.yview)
        self.capture_info_text.configure(yscrollcommand=capture_scrollbar.set)
        
        self.capture_info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        capture_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _refresh_monitors(self):
        """Refresh the list of available monitors."""
        if not self.monitor_manager:
            return
        
        try:
            monitors = self.monitor_manager.get_monitors()
            monitor_options = [f"{i}: {m.name} ({m.width}x{m.height})" 
                             for i, m in enumerate(monitors)]
            
            self.monitor_combo['values'] = monitor_options
            if monitor_options:
                self.monitor_combo.set(monitor_options[0])
            
            # Update monitor info display
            self._update_monitor_info()
            
        except Exception as e:
            self._update_status(f"Error refreshing monitors: {e}")
    
    def _update_monitor_info(self):
        """Update the monitor information display."""
        if not self.monitor_manager:
            return
        
        self.monitor_info_text.delete(1.0, tk.END)
        
        monitors = self.monitor_manager.get_monitors()
        info_text = "Monitor Configuration:\n" + "="*50 + "\n\n"
        
        for monitor in monitors:
            primary_text = " (PRIMARY)" if monitor.is_primary else ""
            info_text += f"Monitor {monitor.index}{primary_text}:\n"
            info_text += f"  Name: {monitor.name}\n"
            info_text += f"  Resolution: {monitor.width}x{monitor.height}\n"
            info_text += f"  Position: ({monitor.left}, {monitor.top})\n"
            info_text += f"  DPI: {monitor.dpi_x}x{monitor.dpi_y}\n\n"
        
        vx, vy, vw, vh = self.monitor_manager.get_virtual_screen_bounds()
        info_text += f"Virtual Screen: {vw}x{vh} at ({vx}, {vy})\n"
        
        self.monitor_info_text.insert(1.0, info_text)
    
    def _on_monitor_selected(self, event):
        """Handle monitor selection change."""
        selected = self.selected_monitor.get()
        monitor_index = int(selected.split(':')[0])
        self._update_status(f"Selected monitor {monitor_index}")
    
    def _test_monitor_capture(self):
        """Test capture of selected monitor."""
        if not self.screen_capture:
            self._update_status("Screen capture not initialized")
            return
        
        try:
            selected = self.selected_monitor.get()
            monitor_index = int(selected.split(':')[0])
            
            start_time = time.time()
            image_data = self.screen_capture.capture_monitor(monitor_index)
            capture_time = time.time() - start_time
            
            # Save test image
            filename = f"test_capture_monitor_{monitor_index}.png"
            self.screen_capture.save_capture(image_data, filename)
            
            self._update_status(f"Test capture saved: {filename} ({capture_time:.3f}s)")
            
        except Exception as e:
            self._update_status(f"Error during test capture: {e}")
    
    def _select_new_region(self):
        """Start interactive region selection."""
        if not self.region_selector:
            self._update_status("Region selector not initialized")
            return
        
        selected = self.selected_monitor.get()
        monitor_index = int(selected.split(':')[0])
        
        def on_region_selected(region_info):
            """Handle region selection completion."""
            # Prompt for region name
            name = tk.simpledialog.askstring("Region Name", "Enter name for this region:")
            if name:
                self.region_selector.add_region(
                    name=name,
                    x=region_info['x'],
                    y=region_info['y'],
                    width=region_info['width'],
                    height=region_info['height'],
                    monitor_index=monitor_index,
                    description=f"Selected on monitor {monitor_index}"
                )
                self._refresh_region_list()
                self._update_status(f"Region '{name}' added")
        
        self.region_selector.select_region_interactive(monitor_index, on_region_selected)
    
    def _refresh_region_list(self):
        """Refresh the region list display."""
        self.region_listbox.delete(0, tk.END)
        
        if self.region_selector:
            for region in self.region_selector.regions:
                display_text = f"{region.name} - {region.width}x{region.height} at ({region.x}, {region.y})"
                self.region_listbox.insert(tk.END, display_text)
    
    def _load_regions(self):
        """Load regions from file."""
        filename = filedialog.askopenfilename(
            title="Load Regions",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename and self.region_selector:
            try:
                self.region_selector.load_regions(filename)
                self._refresh_region_list()
                self._update_status(f"Regions loaded from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load regions: {e}")
    
    def _save_regions(self):
        """Save regions to file."""
        filename = filedialog.asksaveasfilename(
            title="Save Regions",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename and self.region_selector:
            try:
                self.region_selector.save_regions(filename)
                self._update_status(f"Regions saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save regions: {e}")
    
    def _preview_selected_region(self):
        """Preview the selected region."""
        selection = self.region_listbox.curselection()
        if not selection or not self.region_selector:
            return
        
        region_index = selection[0]
        if region_index < len(self.region_selector.regions):
            region = self.region_selector.regions[region_index]
            self.region_selector.create_preview_window(region.name)
    
    def _delete_selected_region(self):
        """Delete the selected region."""
        selection = self.region_listbox.curselection()
        if not selection or not self.region_selector:
            return
        
        region_index = selection[0]
        if region_index < len(self.region_selector.regions):
            region = self.region_selector.regions[region_index]
            
            if messagebox.askyesno("Confirm Delete", f"Delete region '{region.name}'?"):
                self.region_selector.remove_region(region.name)
                self._refresh_region_list()
                self._update_status(f"Region '{region.name}' deleted")
    
    def _start_capture(self):
        """Start continuous capture."""
        if self.capture_active:
            return
        
        self.capture_active = True
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        self._update_status("Continuous capture started")
    
    def _stop_capture(self):
        """Stop continuous capture."""
        self.capture_active = False
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        
        self._update_status("Continuous capture stopped")
    
    def _single_capture(self):
        """Perform single capture."""
        if not self.screen_capture:
            return
        
        try:
            selected = self.selected_monitor.get()
            monitor_index = int(selected.split(':')[0])
            
            start_time = time.time()
            image_data = self.screen_capture.capture_monitor(monitor_index)
            capture_time = time.time() - start_time
            
            if self.save_captures.get():
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.png"
                self.screen_capture.save_capture(image_data, filename)
                status_msg = f"Capture saved: {filename}"
            else:
                status_msg = "Capture completed (not saved)"
            
            self._log_capture_info(f"Single capture - {image_data.shape} - {capture_time:.3f}s")
            self._update_status(status_msg)
            
        except Exception as e:
            self._update_status(f"Capture error: {e}")
    
    def _capture_loop(self):
        """Continuous capture loop (runs in thread)."""
        capture_count = 0
        
        while self.capture_active:
            try:
                selected = self.selected_monitor.get()
                monitor_index = int(selected.split(':')[0])
                
                start_time = time.time()
                image_data = self.screen_capture.capture_monitor(monitor_index)
                capture_time = time.time() - start_time
                
                capture_count += 1
                
                if self.save_captures.get():
                    timestamp = time.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
                    filename = f"capture_{timestamp}_{capture_count:06d}.png"
                    self.screen_capture.save_capture(image_data, filename)
                
                # Log capture info
                self.root.after(0, self._log_capture_info, 
                              f"Capture {capture_count} - {image_data.shape} - {capture_time:.3f}s")
                
                # Wait for next capture
                time.sleep(self.capture_interval.get())
                
            except Exception as e:
                self.root.after(0, self._update_status, f"Capture loop error: {e}")
                break
    
    def _log_capture_info(self, message):
        """Log capture information to the info text widget."""
        timestamp = time.strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        self.capture_info_text.insert(tk.END, log_message)
        self.capture_info_text.see(tk.END)
        
        # Keep only last 100 lines
        lines = self.capture_info_text.get(1.0, tk.END).split('\n')
        if len(lines) > 100:
            self.capture_info_text.delete(1.0, f"{len(lines)-100}.0")
    
    def _update_status(self, message):
        """Update status bar."""
        self.status_bar.configure(text=message)
        self.root.update_idletasks()
    
    def run(self):
        """Run the GUI application."""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self._stop_capture()


# Simple dialog for region naming
import tkinter.simpledialog

if __name__ == "__main__":
    app = VisionControlPanel()
    app.run()