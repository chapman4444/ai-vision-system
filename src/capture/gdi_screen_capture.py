"""
GDI+ Screen Capture System
Provides high-performance screen capture capabilities for AI vision systems.
"""

import ctypes
import ctypes.wintypes
from ctypes import wintypes, windll
import numpy as np
from PIL import Image
import time
from typing import Tuple, Optional, List

# GDI32 and User32 constants
SRCCOPY = 0x00CC0020
CAPTUREBLT = 0x40000000
DIB_RGB_COLORS = 0


class GDIScreenCapture:
    """High-performance screen capture using GDI+ for AI vision systems."""
    
    def __init__(self):
        self.user32 = windll.user32
        self.gdi32 = windll.gdi32
        self.kernel32 = windll.kernel32
        
    def get_monitor_info(self) -> List[dict]:
        """Get information about all available monitors."""
        monitors = []
        
        def monitor_enum_proc(hmonitor, hdc, lprect, lparam):
            """Callback function for EnumDisplayMonitors."""
            monitor_info = wintypes.RECT()
            ctypes.memmove(ctypes.byref(monitor_info), lprect, ctypes.sizeof(wintypes.RECT))
            
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
        
        # Define the callback function type
        MonitorEnumProc = ctypes.WINFUNCTYPE(ctypes.c_int, 
                                           wintypes.HMONITOR, 
                                           wintypes.HDC, 
                                           ctypes.POINTER(wintypes.RECT), 
                                           wintypes.LPARAM)
        
        enum_proc = MonitorEnumProc(monitor_enum_proc)
        self.user32.EnumDisplayMonitors(None, None, enum_proc, 0)
        
        return monitors
    
    def capture_screen_region(self, x: int, y: int, width: int, height: int, include_cursor: bool = True, previous_image: np.ndarray = None, cursor_size: int = 15, cursor_thickness: int = 3, cursor_type: str = 'cross') -> np.ndarray:
        """
        Capture a specific region of the screen using GDI+.
        
        Args:
            x: Left coordinate
            y: Top coordinate  
            width: Width of capture area
            height: Height of capture area
            include_cursor: Whether to include mouse cursor in capture
            cursor_size: Size of cursor overlay
            cursor_thickness: Thickness of cursor lines
            cursor_type: Type of cursor ('cross', 'cursor', 'none')
            
        Returns:
            numpy.ndarray: RGB image data
        """
        # Get desktop window and device context
        hwnd = self.user32.GetDesktopWindow()
        desktop_dc = self.user32.GetWindowDC(hwnd)
        
        # Create compatible device context and bitmap
        mem_dc = self.gdi32.CreateCompatibleDC(desktop_dc)
        bitmap = self.gdi32.CreateCompatibleBitmap(desktop_dc, width, height)
        
        # Select bitmap into memory DC
        old_bitmap = self.gdi32.SelectObject(mem_dc, bitmap)
        
        # Copy screen region to memory DC
        self.gdi32.BitBlt(mem_dc, 0, 0, width, height, desktop_dc, x, y, SRCCOPY)
        
        # Create a second back buffer for double buffering to prevent flicker
        back_buffer_dc = self.gdi32.CreateCompatibleDC(desktop_dc)
        back_buffer_bitmap = self.gdi32.CreateCompatibleBitmap(desktop_dc, width, height)
        old_back_buffer = self.gdi32.SelectObject(back_buffer_dc, back_buffer_bitmap)
        
        # Copy to back buffer first
        self.gdi32.BitBlt(back_buffer_dc, 0, 0, width, height, mem_dc, 0, 0, SRCCOPY)
        
        # Define bitmap info structures
        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ('biSize', wintypes.DWORD),
                ('biWidth', wintypes.LONG),
                ('biHeight', wintypes.LONG),
                ('biPlanes', wintypes.WORD),
                ('biBitCount', wintypes.WORD),
                ('biCompression', wintypes.DWORD),
                ('biSizeImage', wintypes.DWORD),
                ('biXPelsPerMeter', wintypes.LONG),
                ('biYPelsPerMeter', wintypes.LONG),
                ('biClrUsed', wintypes.DWORD),
                ('biClrImportant', wintypes.DWORD)
            ]
        
        class BITMAPINFO(ctypes.Structure):
            _fields_ = [
                ('bmiHeader', BITMAPINFOHEADER),
                ('bmiColors', wintypes.DWORD * 3)
            ]
        
        bitmap_info = BITMAPINFO()
        
        bitmap_info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bitmap_info.bmiHeader.biWidth = width
        bitmap_info.bmiHeader.biHeight = -height  # Negative for top-down DIB
        bitmap_info.bmiHeader.biPlanes = 1
        bitmap_info.bmiHeader.biBitCount = 32
        bitmap_info.bmiHeader.biCompression = 0  # BI_RGB
        
        # Allocate buffer for bitmap data
        buffer_size = width * height * 4  # 4 bytes per pixel (BGRA)
        buffer = (ctypes.c_ubyte * buffer_size)()
        
        # Get bitmap bits from back buffer (ensures complete image)
        self.gdi32.GetDIBits(back_buffer_dc, back_buffer_bitmap, 0, height, buffer, 
                            ctypes.byref(bitmap_info), DIB_RGB_COLORS)
        
        # Cleanup GDI objects (proper order for double buffering)
        self.gdi32.SelectObject(back_buffer_dc, old_back_buffer)
        self.gdi32.DeleteObject(back_buffer_bitmap)
        self.gdi32.DeleteDC(back_buffer_dc)
        
        self.gdi32.SelectObject(mem_dc, old_bitmap)
        self.gdi32.DeleteObject(bitmap)
        self.gdi32.DeleteDC(mem_dc)
        self.user32.ReleaseDC(hwnd, desktop_dc)
        
        # Convert buffer to numpy array and reshape
        img_array = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 4))
        
        # Convert BGRA to RGB
        rgb_array = img_array[:, :, [2, 1, 0]]  # Swap B and R channels
        
        # Composite cursor if requested - now with double-buffered rendering
        if include_cursor:
            rgb_array = self._composite_cursor_on_image(rgb_array, x, y, cursor_size, cursor_thickness, cursor_type)
        
        return rgb_array
    
    def capture_monitor(self, monitor_index: int = 0, include_cursor: bool = True, cursor_size: int = 15, cursor_thickness: int = 3, cursor_type: str = 'cross') -> np.ndarray:
        """
        Capture entire monitor by index.
        
        Args:
            monitor_index: Index of monitor to capture (0 = primary)
            include_cursor: Whether to include mouse cursor in capture
            cursor_size: Size of cursor overlay
            cursor_thickness: Thickness of cursor lines
            cursor_type: Type of cursor ('cross', 'cursor', 'none')
            
        Returns:
            numpy.ndarray: RGB image data
        """
        monitors = self.get_monitor_info()
        if monitor_index >= len(monitors):
            raise ValueError(f"Monitor index {monitor_index} not found. Available: {len(monitors)}")
        
        monitor = monitors[monitor_index]
        return self.capture_screen_region(
            monitor['left'], monitor['top'], 
            monitor['width'], monitor['height'],
            include_cursor, None, cursor_size, cursor_thickness, cursor_type
        )
    
    def capture_primary_monitor(self, include_cursor: bool = True, cursor_size: int = 15, cursor_thickness: int = 3, cursor_type: str = 'cross') -> np.ndarray:
        """Capture the primary monitor."""
        return self.capture_monitor(0, include_cursor, cursor_size, cursor_thickness, cursor_type)
    
    def save_capture(self, image_data: np.ndarray, filename: str) -> None:
        """Save captured image data to file."""
        pil_image = Image.fromarray(image_data)
        pil_image.save(filename)
    
    def get_screen_dimensions(self) -> Tuple[int, int]:
        """Get primary screen dimensions."""
        width = self.user32.GetSystemMetrics(0)  # SM_CXSCREEN
        height = self.user32.GetSystemMetrics(1)  # SM_CYSCREEN
        return width, height
    
    def _get_cursor_info(self) -> Optional[dict]:
        """Get current cursor information."""
        # Define CURSORINFO structure
        class CURSORINFO(ctypes.Structure):
            _fields_ = [
                ('cbSize', wintypes.DWORD),
                ('flags', wintypes.DWORD),
                ('hCursor', wintypes.HANDLE),
                ('ptScreenPos', wintypes.POINT)
            ]
        
        # Get cursor info
        cursor_info = CURSORINFO()
        cursor_info.cbSize = ctypes.sizeof(CURSORINFO)
        
        if self.user32.GetCursorInfo(ctypes.byref(cursor_info)) and cursor_info.flags & 0x00000001:
            return {
                'x': cursor_info.ptScreenPos.x,
                'y': cursor_info.ptScreenPos.y,
                'handle': cursor_info.hCursor,
                'visible': True
            }
        return None
    
    def _composite_cursor_on_image(self, image_array: np.ndarray, offset_x: int, offset_y: int, cursor_size: int = 15, cursor_thickness: int = 3, cursor_type: str = 'cross') -> np.ndarray:
        """Composite cursor onto captured image using PIL."""
        cursor_info = self._get_cursor_info()
        if not cursor_info:
            return image_array
        
        # Convert numpy array to PIL image using optimal format (32bppPARGB equivalent)
        pil_image = Image.fromarray(image_array, mode='RGB')
        
        # Calculate cursor position relative to capture region
        cursor_x = cursor_info['x'] - offset_x
        cursor_y = cursor_info['y'] - offset_y
        
        # Check if cursor is within the captured region
        if 0 <= cursor_x < image_array.shape[1] and 0 <= cursor_y < image_array.shape[0]:
            try:
                from PIL import ImageDraw
                draw = ImageDraw.Draw(pil_image)
                
                if cursor_type == 'cross':
                    # Draw a red cross with configurable size and thickness
                    draw.line([(cursor_x-cursor_size, cursor_y), (cursor_x+cursor_size, cursor_y)], fill='red', width=cursor_thickness)
                    draw.line([(cursor_x, cursor_y-cursor_size), (cursor_x, cursor_y+cursor_size)], fill='red', width=cursor_thickness)
                elif cursor_type == 'cursor':
                    # Draw an arrow-like cursor shape
                    arrow_points = [
                        (cursor_x, cursor_y),
                        (cursor_x, cursor_y + cursor_size),
                        (cursor_x + cursor_size//3, cursor_y + cursor_size*2//3),
                        (cursor_x + cursor_size//2, cursor_y + cursor_size//2),
                        (cursor_x + cursor_size*2//3, cursor_y + cursor_size//3)
                    ]
                    draw.polygon(arrow_points, fill='red', outline='white', width=cursor_thickness)
                # cursor_type == 'none' is handled by not calling this function
                
            except Exception as e:
                pass  # Silently handle cursor drawing errors
        
        return np.array(pil_image)
    
    def _draw_cursor_on_dc(self, dc, offset_x: int, offset_y: int) -> None:
        """Draw the current mouse cursor on the device context - DEPRECATED."""
        # This method doesn't work reliably, using PIL compositing instead
        pass


# Example usage and testing
if __name__ == "__main__":
    capture = GDIScreenCapture()
    
    print("Available monitors:")
    monitors = capture.get_monitor_info()
    for i, monitor in enumerate(monitors):
        print(f"Monitor {i}: {monitor['width']}x{monitor['height']} at ({monitor['left']}, {monitor['top']})")
    
    # Capture primary monitor
    print("\nCapturing primary monitor...")
    start_time = time.time()
    screen_data = capture.capture_primary_monitor()
    capture_time = time.time() - start_time
    
    print(f"Capture completed in {capture_time:.3f}s")
    print(f"Captured image shape: {screen_data.shape}")
    
    # Save capture
    capture.save_capture(screen_data, "screen_capture_test.png")
    print("Saved as 'screen_capture_test.png'")
    
    # Capture region around cursor
    cursor_info = capture._get_cursor_info()
    if cursor_info:
        print(f"\nCapturing 800x600 region around cursor at ({cursor_info['x']}, {cursor_info['y']})...")
        
        # Center capture region around cursor
        region_x = max(0, cursor_info['x'] - 400)
        region_y = max(0, cursor_info['y'] - 300)
        region_width = min(800, 3840 - region_x)  # Assuming dual monitor max width
        region_height = min(600, 1080 - region_y)
        
        print(f"Capturing region: ({region_x}, {region_y}) {region_width}x{region_height}")
        cursor_region_data = capture.capture_screen_region(region_x, region_y, region_width, region_height)
        capture.save_capture(cursor_region_data, "cursor_region_test.png")
        print("Saved as 'cursor_region_test.png'")
    else:
        # Fallback to center region
        width, height = capture.get_screen_dimensions()
        center_x = (width - 800) // 2
        center_y = (height - 600) // 2
        
        print(f"\nCapturing center region 800x600...")
        region_data = capture.capture_screen_region(center_x, center_y, 800, 600)
        capture.save_capture(region_data, "region_capture_test.png")
        print("Saved as 'region_capture_test.png'")