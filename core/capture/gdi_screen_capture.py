"""
GDI+ Screen Capture System
Provides high-performance screen capture capabilities for AI vision systems.
"""

import ctypes
import ctypes.wintypes
import time
from ctypes import windll, wintypes
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

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

    def _mem_dc_context(self, desktop_dc):
        """
        Context manager factory for a compatible memory DC tied to 'desktop_dc'.
        Usage:
            with self._mem_dc_context(desktop_dc) as mem_dc:
                ...
        """
        from contextlib import contextmanager

        gdi32 = self.gdi32

        @contextmanager
        def _cm():
            mem_dc = gdi32.CreateCompatibleDC(desktop_dc)
            if not mem_dc:
                raise RuntimeError("CreateCompatibleDC failed")
            try:
                yield mem_dc
            finally:
                try:
                    gdi32.DeleteDC(mem_dc)
                except Exception:
                    pass

        return _cm()

    def _bitmap_context(self, desktop_dc, width: int, height: int):
        """
        Context manager factory for a compatible bitmap of given size.
        Usage:
            with self._bitmap_context(desktop_dc, w, h) as hbmp:
                ...
        """
        from contextlib import contextmanager

        gdi32 = self.gdi32

        @contextmanager
        def _cm():
            hbmp = gdi32.CreateCompatibleBitmap(desktop_dc, width, height)
            if not hbmp:
                raise RuntimeError("CreateCompatibleBitmap failed")
            try:
                yield hbmp
            finally:
                try:
                    gdi32.DeleteObject(hbmp)
                except Exception:
                    pass

        return _cm()

    def get_monitor_info(self) -> List[dict]:
        """Get information about all available monitors."""
        monitors = []

        def monitor_enum_proc(hmonitor, hdc, lprect, lparam):
            """Callback function for EnumDisplayMonitors."""
            monitor_info = wintypes.RECT()
            ctypes.memmove(
                ctypes.byref(monitor_info), lprect, ctypes.sizeof(wintypes.RECT)
            )

            monitors.append(
                {
                    "handle": hmonitor,
                    "left": monitor_info.left,
                    "top": monitor_info.top,
                    "right": monitor_info.right,
                    "bottom": monitor_info.bottom,
                    "width": monitor_info.right - monitor_info.left,
                    "height": monitor_info.bottom - monitor_info.top,
                }
            )
            return True

        # Define the callback function type
        MonitorEnumProc = ctypes.WINFUNCTYPE(
            ctypes.c_int,
            wintypes.HMONITOR,
            wintypes.HDC,
            ctypes.POINTER(wintypes.RECT),
            wintypes.LPARAM,
        )

        enum_proc = MonitorEnumProc(monitor_enum_proc)
        self.user32.EnumDisplayMonitors(None, None, enum_proc, 0)

        return monitors

    def capture_screen_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        include_cursor: bool = True,
        previous_image: np.ndarray = None,
        cursor_size: int = 15,
        cursor_thickness: int = 3,
        cursor_type: str = "cross",
        exclude_windows: List[str] = None,
        cursor_offset_x: int = 0,
        cursor_offset_y: int = 0,
        timeout: float = 5.0,
    ) -> np.ndarray:
        """
        Capture a specific region of the screen using composable pipeline functions.

        This is now a thin orchestrator that composes pure functions.
        """
        from .capture_pipeline import (acquire_frame_dc, apply_window_masks,
                                       overlay_cursor)

        # Step 1: Acquire frame data with timeout protection
        rgb_array, metadata = acquire_frame_dc(x, y, width, height, timeout)

        # Step 2: Apply window masking if needed
        if exclude_windows:
            excluded_hwnds = self._get_excluded_window_hwnds(exclude_windows)
            rgb_array = apply_window_masks(rgb_array, x, y, excluded_hwnds)

        # Step 3: Apply cursor overlay if requested
        if include_cursor:
            cursor_info = {
                "enabled": True,
                "type": cursor_type,
                "size": cursor_size,
                "thickness": cursor_thickness,
                "offset_x": cursor_offset_x,
                "offset_y": cursor_offset_y,
                "region_x": x,
                "region_y": y,
            }
            rgb_array = overlay_cursor(rgb_array, cursor_info)

        return rgb_array

    def _get_excluded_window_hwnds(self, exclude_windows: List[str]) -> List[int]:
        """Convert window titles to HWNDs for masking pipeline."""
        if not exclude_windows:
            return []

        excluded_hwnds = []
        windows = self._get_window_list()

        for window in windows:
            window_title = window.get("title", "")
            for exclude_pattern in exclude_windows:
                if exclude_pattern.lower() in window_title.lower():
                    excluded_hwnds.append(window["hwnd"])
                    break

        return excluded_hwnds

    def save_capture(self, rgb_array: np.ndarray, filepath: str, metadata: dict = None):
        """Save capture using pipeline save function."""
        from .capture_pipeline import save_frame

        save_frame(rgb_array, filepath, metadata)

    def capture_monitor(
        self,
        monitor_index: int = 0,
        include_cursor: bool = True,
        cursor_size: int = 15,
        cursor_thickness: int = 3,
        cursor_type: str = "cross",
        exclude_windows: List[str] = None,
        cursor_offset_x: int = 0,
        cursor_offset_y: int = 0,
        timeout: float = 5.0,
    ) -> np.ndarray:
        """
        Capture entire monitor by index.

        Args:
            monitor_index: Index of monitor to capture (0 = primary)
            include_cursor: Whether to include mouse cursor in capture
            cursor_size: Size of cursor overlay
            cursor_thickness: Thickness of cursor lines
            cursor_type: Type of cursor ('cross', 'cursor', 'none')
            exclude_windows: List of window titles to mask out

        Returns:
            numpy.ndarray: RGB image data
        """
        monitors = self.get_monitor_info()
        if monitor_index >= len(monitors):
            raise ValueError(
                f"Monitor index {monitor_index} not found. Available: {len(monitors)}"
            )

        monitor = monitors[monitor_index]
        return self.capture_screen_region(
            monitor["left"],
            monitor["top"],
            monitor["width"],
            monitor["height"],
            include_cursor,
            None,
            cursor_size,
            cursor_thickness,
            cursor_type,
            exclude_windows,
            cursor_offset_x,
            cursor_offset_y,
            timeout,
        )

    def capture_primary_monitor(
        self,
        include_cursor: bool = True,
        cursor_size: int = 15,
        cursor_thickness: int = 3,
        cursor_type: str = "cross",
        exclude_windows: List[str] = None,
        cursor_offset_x: int = 0,
        cursor_offset_y: int = 0,
        timeout: float = 5.0,
    ) -> np.ndarray:
        """Capture the primary monitor."""
        return self.capture_monitor(
            0,
            include_cursor,
            cursor_size,
            cursor_thickness,
            cursor_type,
            exclude_windows,
            cursor_offset_x,
            cursor_offset_y,
            timeout,
        )

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
                ("cbSize", wintypes.DWORD),
                ("flags", wintypes.DWORD),
                ("hCursor", wintypes.HANDLE),
                ("ptScreenPos", wintypes.POINT),
            ]

        # Get cursor info
        cursor_info = CURSORINFO()
        cursor_info.cbSize = ctypes.sizeof(CURSORINFO)

        if (
            self.user32.GetCursorInfo(ctypes.byref(cursor_info))
            and cursor_info.flags & 0x00000001
        ):
            return {
                "x": cursor_info.ptScreenPos.x,
                "y": cursor_info.ptScreenPos.y,
                "handle": cursor_info.hCursor,
                "visible": True,
            }
        return None

    def _composite_cursor_on_image(
        self,
        image_array: np.ndarray,
        offset_x: int,
        offset_y: int,
        cursor_size: int = 15,
        cursor_thickness: int = 3,
        cursor_type: str = "cross",
        cursor_offset_x: int = 0,
        cursor_offset_y: int = 0,
    ) -> np.ndarray:
        """Composite cursor onto captured image using PIL."""
        cursor_info = self._get_cursor_info()
        if not cursor_info:
            return image_array

        # Convert numpy array to PIL image using optimal format (32bppPARGB equivalent)
        pil_image = Image.fromarray(image_array, mode="RGB")

        # Calculate cursor position relative to capture region with offset corrections
        cursor_x = cursor_info["x"] - offset_x + cursor_offset_x
        cursor_y = cursor_info["y"] - offset_y + cursor_offset_y

        # Check if cursor is within the captured region
        if (
            0 <= cursor_x < image_array.shape[1]
            and 0 <= cursor_y < image_array.shape[0]
        ):
            try:
                from PIL import ImageDraw

                draw = ImageDraw.Draw(pil_image)

                if cursor_type == "cross":
                    # Draw a red cross with configurable size and thickness
                    draw.line(
                        [
                            (cursor_x - cursor_size, cursor_y),
                            (cursor_x + cursor_size, cursor_y),
                        ],
                        fill="red",
                        width=cursor_thickness,
                    )
                    draw.line(
                        [
                            (cursor_x, cursor_y - cursor_size),
                            (cursor_x, cursor_y + cursor_size),
                        ],
                        fill="red",
                        width=cursor_thickness,
                    )
                elif cursor_type == "cursor":
                    # Draw an arrow-like cursor shape
                    arrow_points = [
                        (cursor_x, cursor_y),
                        (cursor_x, cursor_y + cursor_size),
                        (cursor_x + cursor_size // 3, cursor_y + cursor_size * 2 // 3),
                        (cursor_x + cursor_size // 2, cursor_y + cursor_size // 2),
                        (cursor_x + cursor_size * 2 // 3, cursor_y + cursor_size // 3),
                    ]
                    draw.polygon(
                        arrow_points,
                        fill="red",
                        outline="white",
                        width=cursor_thickness,
                    )
                # cursor_type == 'none' is handled by not calling this function

            except Exception as e:
                pass  # Silently handle cursor drawing errors

        return np.array(pil_image)

    def _draw_cursor_on_dc(self, dc, offset_x: int, offset_y: int) -> None:
        """Draw the current mouse cursor on the device context - DEPRECATED."""
        # This method doesn't work reliably, using PIL compositing instead
        pass

    def _get_window_list(self) -> List[dict]:
        """Get list of all visible windows with their positions and titles."""
        windows = []

        def enum_windows_proc(hwnd, lparam):
            if self.user32.IsWindowVisible(hwnd):
                # Get window title
                title_length = self.user32.GetWindowTextLengthW(hwnd)
                if title_length > 0:
                    title_buffer = ctypes.create_unicode_buffer(title_length + 1)
                    self.user32.GetWindowTextW(hwnd, title_buffer, title_length + 1)
                    title = title_buffer.value

                    # Get window rectangle
                    rect = wintypes.RECT()
                    if self.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                        windows.append(
                            {
                                "hwnd": hwnd,
                                "title": title,
                                "left": rect.left,
                                "top": rect.top,
                                "right": rect.right,
                                "bottom": rect.bottom,
                                "width": rect.right - rect.left,
                                "height": rect.bottom - rect.top,
                            }
                        )
            return True

        # Define the callback function type
        EnumWindowsProc = ctypes.WINFUNCTYPE(
            ctypes.c_bool, wintypes.HWND, wintypes.LPARAM
        )
        enum_proc = EnumWindowsProc(enum_windows_proc)
        self.user32.EnumWindows(enum_proc, 0)

        return windows

    def _mask_excluded_windows(
        self,
        image_array: np.ndarray,
        capture_x: int,
        capture_y: int,
        exclude_windows: List[str],
    ) -> np.ndarray:
        """Mask out (blacken) windows that match the excluded titles to prevent feedback loops."""
        try:
            windows = self._get_window_list()

            for window in windows:
                # Check if this window title matches any of our exclude patterns
                for exclude_pattern in exclude_windows:
                    if exclude_pattern.lower() in window["title"].lower():
                        # Calculate window position relative to capture region
                        win_left = window["left"] - capture_x
                        win_top = window["top"] - capture_y
                        win_right = window["right"] - capture_x
                        win_bottom = window["bottom"] - capture_y

                        # Clip to image boundaries
                        win_left = max(0, win_left)
                        win_top = max(0, win_top)
                        win_right = min(image_array.shape[1], win_right)
                        win_bottom = min(image_array.shape[0], win_bottom)

                        # Only mask if there's actually an intersection
                        if win_left < win_right and win_top < win_bottom:
                            # Mask the window area (make it black)
                            image_array[win_top:win_bottom, win_left:win_right] = [
                                0,
                                0,
                                0,
                            ]
                            print(
                                f"Masked window '{window['title']}' at ({window['left']},{window['top']}) to ({window['right']},{window['bottom']}) [screen coords]"
                            )
                        break  # Found a match, no need to check other patterns

            return image_array

        except Exception as e:
            print(f"Window masking error: {e}")
            return image_array


# Example usage and testing
if __name__ == "__main__":
    capture = GDIScreenCapture()

    print("Available monitors:")
    monitors = capture.get_monitor_info()
    for i, monitor in enumerate(monitors):
        print(
            f"Monitor {i}: {monitor['width']}x{monitor['height']} at ({monitor['left']}, {monitor['top']})"
        )

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
        print(
            f"\nCapturing 800x600 region around cursor at ({cursor_info['x']}, {cursor_info['y']})..."
        )

        # Center capture region around cursor
        region_x = max(0, cursor_info["x"] - 400)
        region_y = max(0, cursor_info["y"] - 300)
        region_width = min(800, 3840 - region_x)  # Assuming dual monitor max width
        region_height = min(600, 1080 - region_y)

        print(
            f"Capturing region: ({region_x}, {region_y}) {region_width}x{region_height}"
        )
        cursor_region_data = capture.capture_screen_region(
            region_x, region_y, region_width, region_height
        )
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
