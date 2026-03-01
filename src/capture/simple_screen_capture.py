"""
Simple Screen Capture System
Simplified version for immediate compatibility.
"""

import ctypes
import ctypes.wintypes
import numpy as np
from PIL import Image
import time
from typing import Tuple, Optional, List, Dict, Union
from enum import Enum


class ImageFormat(Enum):
    """Supported image formats."""
    PNG = "PNG"
    JPEG = "JPEG"
    WEBP = "WEBP"
    BMP = "BMP"


class CompressionLevel(Enum):
    """Compression quality levels."""
    LOW = 30      # High compression, lower quality
    MEDIUM = 60   # Balanced
    HIGH = 85     # Low compression, high quality
    LOSSLESS = 100  # No compression (PNG only)


class SimpleScreenCapture:
    """Simplified screen capture using basic Windows API with format/compression support."""
    
    def __init__(self, default_format: ImageFormat = ImageFormat.PNG, 
                 default_quality: CompressionLevel = CompressionLevel.HIGH):
        self.user32 = ctypes.windll.user32
        self.gdi32 = ctypes.windll.gdi32
        self.kernel32 = ctypes.windll.kernel32
        
        # Format and compression settings
        self.default_format = default_format
        self.default_quality = default_quality
        
        # Format-specific settings
        self.format_settings = {
            ImageFormat.PNG: {
                "optimize": True,
                "compress_level": 6  # 0-9, 6 is good balance
            },
            ImageFormat.JPEG: {
                "optimize": True,
                "progressive": True
            },
            ImageFormat.WEBP: {
                "method": 4,  # 0-6, 4 is balanced
                "lossless": False
            },
            ImageFormat.BMP: {}
        }
    
    def get_screen_dimensions(self) -> Tuple[int, int]:
        """Get primary screen dimensions."""
        width = self.user32.GetSystemMetrics(0)   # SM_CXSCREEN  
        height = self.user32.GetSystemMetrics(1)  # SM_CYSCREEN
        return width, height
    
    def capture_screen_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """
        Capture a specific region of the screen.
        
        Args:
            x: Left coordinate
            y: Top coordinate
            width: Width of capture area
            height: Height of capture area
            
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
        SRCCOPY = 0x00CC0020
        self.gdi32.BitBlt(mem_dc, 0, 0, width, height, desktop_dc, x, y, SRCCOPY)
        
        # Get bitmap info
        class BITMAPINFO(ctypes.Structure):
            class BITMAPINFOHEADER(ctypes.Structure):
                _fields_ = [
                    ('biSize', ctypes.wintypes.DWORD),
                    ('biWidth', ctypes.c_long),
                    ('biHeight', ctypes.c_long),
                    ('biPlanes', ctypes.wintypes.WORD),
                    ('biBitCount', ctypes.wintypes.WORD),
                    ('biCompression', ctypes.wintypes.DWORD),
                    ('biSizeImage', ctypes.wintypes.DWORD),
                    ('biXPelsPerMeter', ctypes.c_long),
                    ('biYPelsPerMeter', ctypes.c_long),
                    ('biClrUsed', ctypes.wintypes.DWORD),
                    ('biClrImportant', ctypes.wintypes.DWORD)
                ]
            
            _fields_ = [('bmiHeader', BITMAPINFOHEADER)]
        
        bitmap_info = BITMAPINFO()
        bitmap_info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFO.BITMAPINFOHEADER)
        bitmap_info.bmiHeader.biWidth = width
        bitmap_info.bmiHeader.biHeight = -height  # Negative for top-down DIB
        bitmap_info.bmiHeader.biPlanes = 1
        bitmap_info.bmiHeader.biBitCount = 32
        bitmap_info.bmiHeader.biCompression = 0  # BI_RGB
        
        # Allocate buffer for bitmap data
        buffer_size = width * height * 4  # 4 bytes per pixel (BGRA)
        buffer = (ctypes.c_ubyte * buffer_size)()
        
        # Get bitmap bits
        DIB_RGB_COLORS = 0
        self.gdi32.GetDIBits(mem_dc, bitmap, 0, height, buffer, 
                            ctypes.byref(bitmap_info), DIB_RGB_COLORS)
        
        # Cleanup GDI objects
        self.gdi32.SelectObject(mem_dc, old_bitmap)
        self.gdi32.DeleteObject(bitmap)
        self.gdi32.DeleteDC(mem_dc)
        self.user32.ReleaseDC(hwnd, desktop_dc)
        
        # Convert buffer to numpy array and reshape
        img_array = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 4))
        
        # Convert BGRA to RGB
        rgb_array = img_array[:, :, [2, 1, 0]]  # Swap B and R channels
        
        return rgb_array
    
    def capture_primary_monitor(self) -> np.ndarray:
        """Capture the primary monitor."""
        width, height = self.get_screen_dimensions()
        return self.capture_screen_region(0, 0, width, height)
    
    def save_capture(self, image_data: np.ndarray, filename: str, 
                    format_type: Optional[ImageFormat] = None,
                    quality: Optional[Union[CompressionLevel, int]] = None,
                    **kwargs) -> Dict[str, any]:
        """Save captured image data to file with format and compression options.
        
        Args:
            image_data: Image array data
            filename: Output filename
            format_type: Image format (PNG, JPEG, WEBP, BMP)
            quality: Compression quality level
            **kwargs: Additional format-specific options
            
        Returns:
            Dict with save information (file_size, format, quality, etc.)
        """
        pil_image = Image.fromarray(image_data)
        
        # Determine format from filename or use default
        if format_type is None:
            # Try to infer from extension
            ext = filename.lower().split('.')[-1]
            format_map = {
                'png': ImageFormat.PNG,
                'jpg': ImageFormat.JPEG,
                'jpeg': ImageFormat.JPEG,
                'webp': ImageFormat.WEBP,
                'bmp': ImageFormat.BMP
            }
            format_type = format_map.get(ext, self.default_format)
        
        # Determine quality
        if quality is None:
            quality = self.default_quality
        elif isinstance(quality, int):
            quality = CompressionLevel(quality)
        
        # Get base format settings
        save_kwargs = self.format_settings[format_type].copy()
        save_kwargs.update(kwargs)  # Override with user kwargs
        
        # Apply quality settings based on format
        if format_type == ImageFormat.JPEG:
            save_kwargs['quality'] = quality.value
        elif format_type == ImageFormat.WEBP:
            if quality == CompressionLevel.LOSSLESS:
                save_kwargs['lossless'] = True
                save_kwargs['quality'] = 100
            else:
                save_kwargs['lossless'] = False
                save_kwargs['quality'] = quality.value
        elif format_type == ImageFormat.PNG:
            # PNG compression level (0-9, lower = less compression but faster)
            if quality == CompressionLevel.LOW:
                save_kwargs['compress_level'] = 1
            elif quality == CompressionLevel.MEDIUM:
                save_kwargs['compress_level'] = 6
            else:  # HIGH or LOSSLESS
                save_kwargs['compress_level'] = 9
        
        # Convert RGB to appropriate color mode if needed
        if format_type == ImageFormat.JPEG and pil_image.mode != 'RGB':
            # JPEG doesn't support transparency, convert RGBA to RGB with white background
            if pil_image.mode == 'RGBA':
                background = Image.new('RGB', pil_image.size, (255, 255, 255))
                background.paste(pil_image, mask=pil_image.split()[-1])  # Use alpha as mask
                pil_image = background
        
        # Save the image
        pil_image.save(filename, format=format_type.value, **save_kwargs)
        
        # Get file info
        import os
        file_size = os.path.getsize(filename)
        
        return {
            'filename': filename,
            'format': format_type.value,
            'quality': quality.value if hasattr(quality, 'value') else quality,
            'file_size': file_size,
            'dimensions': pil_image.size,
            'save_kwargs': save_kwargs
        }
    
    def get_monitor_count(self) -> int:
        """Get number of monitors (simplified)."""
        return self.user32.GetSystemMetrics(80)  # SM_CMONITORS
    
    def estimate_file_size(self, image_data: np.ndarray, 
                          format_type: ImageFormat = ImageFormat.PNG,
                          quality: CompressionLevel = CompressionLevel.HIGH) -> int:
        """Estimate output file size without saving.
        
        Args:
            image_data: Image array data
            format_type: Target format
            quality: Target quality
            
        Returns:
            Estimated file size in bytes
        """
        height, width = image_data.shape[:2]
        pixels = width * height
        
        # Rough estimates based on format and quality
        if format_type == ImageFormat.PNG:
            # PNG: roughly 2-4 bytes per pixel depending on content
            return int(pixels * 2.5)
        elif format_type == ImageFormat.JPEG:
            # JPEG: varies greatly with quality
            if quality.value >= 90:
                return int(pixels * 0.8)
            elif quality.value >= 70:
                return int(pixels * 0.4)
            else:
                return int(pixels * 0.2)
        elif format_type == ImageFormat.WEBP:
            # WebP: generally 25-35% smaller than JPEG
            jpeg_estimate = int(pixels * (0.8 if quality.value >= 90 else 
                                        0.4 if quality.value >= 70 else 0.2))
            return int(jpeg_estimate * 0.7)
        elif format_type == ImageFormat.BMP:
            # BMP: uncompressed, 3 bytes per pixel
            return pixels * 3
        
        return pixels * 3  # Fallback
    
    def get_optimal_format(self, image_data: np.ndarray, 
                          max_file_size: Optional[int] = None,
                          prefer_quality: bool = True) -> Tuple[ImageFormat, CompressionLevel]:
        """Suggest optimal format and quality for given constraints.
        
        Args:
            image_data: Image to analyze
            max_file_size: Maximum desired file size in bytes
            prefer_quality: Whether to prefer quality over file size
            
        Returns:
            Tuple of (recommended_format, recommended_quality)
        """
        height, width = image_data.shape[:2]
        
        # Check for transparency (if RGBA)
        has_transparency = len(image_data.shape) == 3 and image_data.shape[2] == 4
        
        # If transparency is present, PNG or WebP are best options
        if has_transparency:
            if max_file_size:
                webp_size = self.estimate_file_size(image_data, ImageFormat.WEBP, CompressionLevel.HIGH)
                if webp_size <= max_file_size:
                    return ImageFormat.WEBP, CompressionLevel.HIGH
                else:
                    return ImageFormat.WEBP, CompressionLevel.MEDIUM
            else:
                return ImageFormat.PNG, CompressionLevel.HIGH
        
        # For screenshots, check content complexity
        # Simple heuristic: calculate color variance
        color_variance = np.var(image_data)
        is_simple_content = color_variance < 1000  # Threshold for "simple" screenshots
        
        if max_file_size:
            # Try different formats and find the best fit
            formats_to_try = [
                (ImageFormat.WEBP, CompressionLevel.HIGH),
                (ImageFormat.JPEG, CompressionLevel.HIGH),
                (ImageFormat.WEBP, CompressionLevel.MEDIUM),
                (ImageFormat.JPEG, CompressionLevel.MEDIUM),
                (ImageFormat.WEBP, CompressionLevel.LOW),
                (ImageFormat.JPEG, CompressionLevel.LOW),
            ]
            
            if is_simple_content:
                # PNG might be good for simple content
                formats_to_try.insert(0, (ImageFormat.PNG, CompressionLevel.HIGH))
            
            for fmt, qual in formats_to_try:
                estimated_size = self.estimate_file_size(image_data, fmt, qual)
                if estimated_size <= max_file_size:
                    return fmt, qual
            
            # If nothing fits, use lowest quality JPEG
            return ImageFormat.JPEG, CompressionLevel.LOW
        
        else:
            # No size constraint, optimize for content type
            if is_simple_content:
                return ImageFormat.PNG, CompressionLevel.HIGH
            elif prefer_quality:
                return ImageFormat.WEBP, CompressionLevel.HIGH
            else:
                return ImageFormat.JPEG, CompressionLevel.HIGH


# Example usage and testing
if __name__ == "__main__":
    capture = SimpleScreenCapture()
    
    width, height = capture.get_screen_dimensions()
    print(f"Primary screen: {width}x{height}")
    
    monitor_count = capture.get_monitor_count()
    print(f"Monitor count: {monitor_count}")
    
    # Test capture
    print("Capturing screen...")
    start_time = time.time()
    
    # Capture small test region
    test_data = capture.capture_screen_region(0, 0, 400, 300)
    capture_time = time.time() - start_time
    
    print(f"Captured {test_data.shape} in {capture_time:.3f}s")
    
    # Test different formats
    formats_to_test = [
        (ImageFormat.PNG, CompressionLevel.HIGH, "simple_test_capture.png"),
        (ImageFormat.JPEG, CompressionLevel.HIGH, "simple_test_capture.jpg"),
        (ImageFormat.WEBP, CompressionLevel.HIGH, "simple_test_capture.webp"),
    ]
    
    print("\nTesting different formats:")
    for fmt, quality, filename in formats_to_test:
        try:
            save_info = capture.save_capture(test_data, filename, fmt, quality)
            print(f"{fmt.value} ({quality.value}): {save_info['file_size']:,} bytes -> {filename}")
        except Exception as e:
            print(f"{fmt.value}: Error - {e}")
    
    # Test optimal format suggestion
    optimal_fmt, optimal_quality = capture.get_optimal_format(test_data, max_file_size=50000)
    print(f"\nOptimal format for <50KB: {optimal_fmt.value} at {optimal_quality.value} quality")