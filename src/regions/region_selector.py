"""
Interactive Region Selection System
Allows users to select specific screen regions for capture and monitoring.
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageTk
from typing import Tuple, Optional, Callable, List
from dataclasses import dataclass
import json


@dataclass
class Region:
    """Data class for screen regions."""
    name: str
    x: int
    y: int
    width: int
    height: int
    monitor_index: int = 0
    description: str = ""


class RegionSelector:
    """Interactive region selection tool with preview."""
    
    def __init__(self, screen_capture, monitor_manager):
        self.screen_capture = screen_capture
        self.monitor_manager = monitor_manager
        self.regions: List[Region] = []
        self.current_region: Optional[Region] = None
        self.selection_callback: Optional[Callable] = None
        
    def create_selection_overlay(self, monitor_index: int = 0) -> None:
        """Create a transparent overlay for region selection."""
        monitor = self.monitor_manager.get_monitor_by_index(monitor_index)
        if not monitor:
            raise ValueError(f"Monitor {monitor_index} not found")
        
        # Create overlay window
        self.overlay = tk.Toplevel()
        self.overlay.title("Region Selection")
        self.overlay.attributes('-alpha', 0.3)
        self.overlay.attributes('-topmost', True)
        self.overlay.configure(bg='red')
        
        # Position overlay on selected monitor
        self.overlay.geometry(f"{monitor.width}x{monitor.height}+{monitor.left}+{monitor.top}")
        
        # Bind mouse events
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        self.canvas = tk.Canvas(self.overlay, highlightthickness=0, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.canvas.bind('<Button-1>', self._start_selection)
        self.canvas.bind('<B1-Motion>', self._update_selection)
        self.canvas.bind('<ButtonRelease-1>', self._end_selection)
        self.canvas.bind('<Escape>', lambda e: self.overlay.destroy())
        
        self.overlay.focus_set()
        
        # Add instructions
        instruction = tk.Label(self.overlay, 
                             text="Click and drag to select region. Press ESC to cancel.",
                             bg='yellow', fg='black')
        instruction.pack(side=tk.TOP)
        
        self.selected_region = None
        self.monitor_offset = (monitor.left, monitor.top)
    
    def _start_selection(self, event) -> None:
        """Start region selection."""
        self.start_x = event.x
        self.start_y = event.y
        
        if self.rect_id:
            self.canvas.delete(self.rect_id)
    
    def _update_selection(self, event) -> None:
        """Update selection rectangle."""
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y,
            outline='red', width=2, fill='blue', stipple='gray50'
        )
    
    def _end_selection(self, event) -> None:
        """Finalize region selection."""
        if self.start_x and self.start_y:
            # Calculate region bounds
            x1, y1 = min(self.start_x, event.x), min(self.start_y, event.y)
            x2, y2 = max(self.start_x, event.x), max(self.start_y, event.y)
            
            # Convert to screen coordinates
            screen_x = x1 + self.monitor_offset[0]
            screen_y = y1 + self.monitor_offset[1]
            width = x2 - x1
            height = y2 - y1
            
            self.selected_region = {
                'x': screen_x,
                'y': screen_y, 
                'width': width,
                'height': height
            }
            
            self.overlay.destroy()
            
            if self.selection_callback:
                self.selection_callback(self.selected_region)
    
    def select_region_interactive(self, monitor_index: int = 0, 
                                callback: Optional[Callable] = None) -> None:
        """Start interactive region selection."""
        self.selection_callback = callback
        self.create_selection_overlay(monitor_index)
    
    def add_region(self, name: str, x: int, y: int, width: int, height: int, 
                   monitor_index: int = 0, description: str = "") -> None:
        """Add a predefined region."""
        region = Region(name, x, y, width, height, monitor_index, description)
        self.regions.append(region)
    
    def get_region_by_name(self, name: str) -> Optional[Region]:
        """Get region by name."""
        for region in self.regions:
            if region.name == name:
                return region
        return None
    
    def remove_region(self, name: str) -> bool:
        """Remove region by name."""
        for i, region in enumerate(self.regions):
            if region.name == name:
                del self.regions[i]
                return True
        return False
    
    def save_regions(self, filename: str) -> None:
        """Save regions to JSON file."""
        data = {
            'regions': [
                {
                    'name': r.name,
                    'x': r.x,
                    'y': r.y,
                    'width': r.width,
                    'height': r.height,
                    'monitor_index': r.monitor_index,
                    'description': r.description
                }
                for r in self.regions
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_regions(self, filename: str) -> None:
        """Load regions from JSON file."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.regions.clear()
            for r_data in data.get('regions', []):
                region = Region(
                    name=r_data['name'],
                    x=r_data['x'],
                    y=r_data['y'], 
                    width=r_data['width'],
                    height=r_data['height'],
                    monitor_index=r_data.get('monitor_index', 0),
                    description=r_data.get('description', '')
                )
                self.regions.append(region)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading regions: {e}")
    
    def capture_region(self, region_name: str) -> Optional[np.ndarray]:
        """Capture a specific region by name."""
        region = self.get_region_by_name(region_name)
        if not region:
            return None
        
        return self.screen_capture.capture_screen_region(
            region.x, region.y, region.width, region.height
        )
    
    def create_preview_window(self, region_name: str) -> None:
        """Create a preview window showing the region capture."""
        region = self.get_region_by_name(region_name)
        if not region:
            print(f"Region '{region_name}' not found")
            return
        
        # Create preview window
        preview = tk.Toplevel()
        preview.title(f"Preview: {region.name}")
        
        def update_preview():
            """Update preview image."""
            try:
                # Capture region
                img_data = self.capture_region(region_name)
                if img_data is not None:
                    # Convert to PIL Image
                    pil_img = Image.fromarray(img_data)
                    
                    # Resize for preview if too large
                    max_size = 800
                    if pil_img.width > max_size or pil_img.height > max_size:
                        pil_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    
                    # Convert to Tkinter PhotoImage
                    photo = ImageTk.PhotoImage(pil_img)
                    
                    # Update label
                    if hasattr(update_preview, 'label'):
                        update_preview.label.configure(image=photo)
                        update_preview.label.image = photo  # Keep reference
                    else:
                        update_preview.label = tk.Label(preview, image=photo)
                        update_preview.label.image = photo
                        update_preview.label.pack()
                        
                        # Add info label
                        info_text = f"{region.name} - {region.width}x{region.height} at ({region.x}, {region.y})"
                        info_label = tk.Label(preview, text=info_text)
                        info_label.pack()
                
                # Schedule next update
                preview.after(100, update_preview)  # Update every 100ms
                
            except Exception as e:
                print(f"Preview update error: {e}")
                preview.after(1000, update_preview)  # Retry after 1 second
        
        update_preview()
    
    def list_regions(self) -> None:
        """Print all defined regions."""
        if not self.regions:
            print("No regions defined.")
            return
        
        print("Defined Regions:")
        print("-" * 50)
        for region in self.regions:
            print(f"Name: {region.name}")
            print(f"  Position: ({region.x}, {region.y})")
            print(f"  Size: {region.width}x{region.height}")
            print(f"  Monitor: {region.monitor_index}")
            if region.description:
                print(f"  Description: {region.description}")
            print()


# Example usage and testing
if __name__ == "__main__":
    # This would typically be imported and used with the screen capture system
    print("Region Selector module - use with screen capture system")
    print("Example usage:")
    print("  selector = RegionSelector(screen_capture, monitor_manager)")
    print("  selector.select_region_interactive(0, lambda r: print(f'Selected: {r}'))")