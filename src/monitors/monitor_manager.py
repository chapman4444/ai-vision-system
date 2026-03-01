"""
Monitor Management System
Handles monitor detection, selection, and configuration for screen capture.
"""

import ctypes
import ctypes.wintypes
from ctypes import wintypes, windll, Structure, c_wchar
from typing import List, Dict, Optional, Tuple
import json
from dataclasses import dataclass, asdict


# Define MONITORINFOEX structure since it's not in wintypes
class MONITORINFOEX(Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT), 
        ("dwFlags", wintypes.DWORD),
        ("szDevice", c_wchar * 32)
    ]


@dataclass
class MonitorInfo:
    """Data class for monitor information."""
    index: int
    handle: int
    name: str
    left: int
    top: int
    right: int
    bottom: int
    width: int
    height: int
    is_primary: bool
    dpi_x: int = 96
    dpi_y: int = 96


class MonitorManager:
    """Advanced monitor management and selection system."""
    
    def __init__(self):
        self.user32 = windll.user32
        self.gdi32 = windll.gdi32
        self.shcore = windll.shcore
        self.monitors: List[MonitorInfo] = []
        self._refresh_monitors()
    
    def _refresh_monitors(self) -> None:
        """Refresh the list of available monitors."""
        self.monitors.clear()
        monitor_data = []
        
        def monitor_enum_proc(hmonitor, hdc, lprect, lparam):
            """Callback for EnumDisplayMonitors."""
            rect = wintypes.RECT()
            ctypes.memmove(ctypes.byref(rect), lprect, ctypes.sizeof(wintypes.RECT))
            
            # Get monitor info
            monitor_info_ex = MONITORINFOEX()
            monitor_info_ex.cbSize = ctypes.sizeof(MONITORINFOEX)
            
            if self.user32.GetMonitorInfoW(hmonitor, ctypes.byref(monitor_info_ex)):
                device_name = monitor_info_ex.szDevice
                is_primary = bool(monitor_info_ex.dwFlags & 1)  # MONITORINFOF_PRIMARY
                
                monitor_data.append({
                    'handle': hmonitor,
                    'hdc': hdc,
                    'rect': rect,
                    'device_name': device_name,
                    'is_primary': is_primary
                })
            
            return True
        
        # Enumerate monitors
        MonitorEnumProc = ctypes.WINFUNCTYPE(ctypes.c_int, wintypes.HMONITOR, 
                                           wintypes.HDC, ctypes.POINTER(wintypes.RECT), 
                                           wintypes.LPARAM)
        enum_proc = MonitorEnumProc(monitor_enum_proc)
        self.user32.EnumDisplayMonitors(None, None, enum_proc, 0)
        
        # Process monitor data
        for i, data in enumerate(monitor_data):
            rect = data['rect']
            width = rect.right - rect.left
            height = rect.bottom - rect.top
            
            # Get DPI information
            dpi_x, dpi_y = self._get_monitor_dpi(data['handle'])
            
            monitor = MonitorInfo(
                index=i,
                handle=data['handle'],
                name=data['device_name'],
                left=rect.left,
                top=rect.top,
                right=rect.right,
                bottom=rect.bottom,
                width=width,
                height=height,
                is_primary=data['is_primary'],
                dpi_x=dpi_x,
                dpi_y=dpi_y
            )
            
            self.monitors.append(monitor)
    
    def _get_monitor_dpi(self, hmonitor: int) -> Tuple[int, int]:
        """Get DPI information for a monitor."""
        try:
            dpi_x = ctypes.c_uint()
            dpi_y = ctypes.c_uint()
            
            # Try to get DPI awareness (Windows 8.1+)
            result = self.shcore.GetDpiForMonitor(hmonitor, 0, 
                                                ctypes.byref(dpi_x), 
                                                ctypes.byref(dpi_y))
            
            if result == 0:  # S_OK
                return int(dpi_x.value), int(dpi_y.value)
        except (AttributeError, OSError):
            pass
        
        # Fallback to system DPI
        return 96, 96
    
    def get_monitors(self) -> List[MonitorInfo]:
        """Get list of all available monitors."""
        return self.monitors.copy()
    
    def get_primary_monitor(self) -> Optional[MonitorInfo]:
        """Get the primary monitor."""
        for monitor in self.monitors:
            if monitor.is_primary:
                return monitor
        return None
    
    def get_monitor_by_index(self, index: int) -> Optional[MonitorInfo]:
        """Get monitor by index."""
        if 0 <= index < len(self.monitors):
            return self.monitors[index]
        return None
    
    def get_monitor_by_name(self, name: str) -> Optional[MonitorInfo]:
        """Get monitor by device name."""
        for monitor in self.monitors:
            if monitor.name == name:
                return monitor
        return None
    
    def find_monitor_at_point(self, x: int, y: int) -> Optional[MonitorInfo]:
        """Find which monitor contains the given point."""
        point = wintypes.POINT(x, y)
        hmonitor = self.user32.MonitorFromPoint(point, 2)  # MONITOR_DEFAULTTONEAREST
        
        for monitor in self.monitors:
            if monitor.handle == hmonitor:
                return monitor
        return None
    
    def get_monitor_bounds(self, monitor_index: int) -> Optional[Tuple[int, int, int, int]]:
        """Get monitor bounds as (left, top, width, height)."""
        monitor = self.get_monitor_by_index(monitor_index)
        if monitor:
            return (monitor.left, monitor.top, monitor.width, monitor.height)
        return None
    
    def get_virtual_screen_bounds(self) -> Tuple[int, int, int, int]:
        """Get bounds of the entire virtual screen."""
        left = self.user32.GetSystemMetrics(76)  # SM_XVIRTUALSCREEN
        top = self.user32.GetSystemMetrics(77)   # SM_YVIRTUALSCREEN
        width = self.user32.GetSystemMetrics(78) # SM_CXVIRTUALSCREEN
        height = self.user32.GetSystemMetrics(79) # SM_CYVIRTUALSCREEN
        return (left, top, width, height)
    
    def save_monitor_config(self, filename: str) -> None:
        """Save current monitor configuration to JSON file."""
        config_data = {
            'monitors': [asdict(monitor) for monitor in self.monitors],
            'virtual_screen': self.get_virtual_screen_bounds()
        }
        
        with open(filename, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def print_monitor_info(self) -> None:
        """Print detailed information about all monitors."""
        print("Monitor Configuration:")
        print("-" * 60)
        
        for monitor in self.monitors:
            primary_text = " (PRIMARY)" if monitor.is_primary else ""
            print(f"Monitor {monitor.index}{primary_text}:")
            print(f"  Name: {monitor.name}")
            print(f"  Resolution: {monitor.width}x{monitor.height}")
            print(f"  Position: ({monitor.left}, {monitor.top})")
            print(f"  DPI: {monitor.dpi_x}x{monitor.dpi_y}")
            print()
        
        vx, vy, vw, vh = self.get_virtual_screen_bounds()
        print(f"Virtual Screen: {vw}x{vh} at ({vx}, {vy})")


# Example usage
if __name__ == "__main__":
    manager = MonitorManager()
    manager.print_monitor_info()
    
    # Save configuration
    manager.save_monitor_config("monitor_config.json")
    print("Monitor configuration saved to 'monitor_config.json'")
    
    # Test point detection
    print("\nTesting point detection:")
    test_points = [(0, 0), (500, 300), (-100, 200)]
    
    for x, y in test_points:
        monitor = manager.find_monitor_at_point(x, y)
        if monitor:
            print(f"Point ({x}, {y}) is on Monitor {monitor.index}: {monitor.name}")
        else:
            print(f"Point ({x}, {y}) not found on any monitor")