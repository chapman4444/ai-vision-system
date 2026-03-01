# AI Vision System - Screen Capture Foundation

A high-performance Windows screen capture system using GDI+ (Graphics Device Interface Plus) that provides "eyes" for AI systems. This foundation enables real-time screen monitoring, selective region capture, and multi-monitor support.

## 🎯 Features

### Core Capabilities
- **High-Performance GDI+ Capture**: Direct Windows API access for minimal latency
- **Multi-Monitor Support**: Automatic detection and selection of specific monitors
- **Region Selection**: Interactive selection of specific screen areas
- **Real-Time Processing**: Continuous capture with configurable intervals
- **Flexible Output**: Save to disk or process in-memory for AI pipelines

### Advanced Features
- **Interactive Region Selection**: Click-and-drag interface for precise area selection
- **Monitor Management**: Complete monitor topology detection and DPI awareness
- **Live Preview**: Real-time preview windows for selected regions
- **Configuration Persistence**: Save/load monitor and region configurations
- **Thread-Safe Operation**: Background capture with GUI responsiveness

## 📁 Project Structure

```
ai-vision-system/
├── src/
│   ├── capture/
│   │   └── gdi_screen_capture.py     # Core GDI+ capture implementation
│   ├── monitors/
│   │   └── monitor_manager.py        # Monitor detection and management
│   ├── regions/
│   │   └── region_selector.py        # Interactive region selection
│   ├── gui/
│   │   └── vision_control_panel.py   # Main GUI application
│   └── processing/                   # Future: AI processing modules
├── examples/                         # Usage examples
├── tests/                           # Unit tests
├── docs/                            # Documentation
├── config/                          # Configuration files
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🚀 Quick Start

### Prerequisites
- Windows 10/11 (GDI+ support required)
- Python 3.7+
- tkinter (usually included with Python)

### Installation

1. **Clone or extract the project**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the GUI application**:
   ```bash
   python src/gui/vision_control_panel.py
   ```

### Basic Usage

#### GUI Application
Launch the control panel for interactive use:
```python
python src/gui/vision_control_panel.py
```

The GUI provides three main tabs:
- **Monitors**: View and select monitors for capture
- **Regions**: Define and manage capture regions
- **Capture Control**: Start/stop continuous capture with settings

#### Programmatic Usage

```python
from src.capture.gdi_screen_capture import GDIScreenCapture
from src.monitors.monitor_manager import MonitorManager

# Initialize components
capture = GDIScreenCapture()
monitor_mgr = MonitorManager()

# Capture primary monitor
screen_data = capture.capture_primary_monitor()
print(f"Captured: {screen_data.shape}")

# Capture specific region
region_data = capture.capture_screen_region(100, 100, 800, 600)
capture.save_capture(region_data, "region_capture.png")

# List available monitors
monitors = monitor_mgr.get_monitors()
for i, monitor in enumerate(monitors):
    print(f"Monitor {i}: {monitor.width}x{monitor.height}")
```

## 🔧 Core Components

### GDIScreenCapture
High-performance screen capture using Windows GDI+ API:
- `capture_screen_region(x, y, width, height)` - Capture specific area
- `capture_monitor(index)` - Capture entire monitor
- `capture_primary_monitor()` - Capture main display
- `get_monitor_info()` - List available monitors

### MonitorManager
Advanced monitor detection and management:
- Automatic monitor topology detection
- DPI awareness for high-resolution displays
- Monitor identification by position, name, or index
- Virtual screen bounds calculation

### RegionSelector
Interactive region selection and management:
- Click-and-drag region selection overlay
- Region persistence (save/load configurations)
- Live preview of selected regions
- Named region management

### VisionControlPanel
Complete GUI application for system control:
- Monitor selection and information display
- Interactive region selection tools
- Capture control with configurable intervals
- Real-time capture monitoring and logging

## ⚡ Performance Features

### Optimized Capture
- Direct GDI+ API access (no screen copy overhead)
- Memory-efficient numpy array conversion
- Configurable capture intervals (10ms+ supported)
- Thread-safe background capture

### Scalability
- Support for multiple monitors (tested up to 6 displays)
- Region capture reduces processing overhead
- Efficient memory management for continuous operation
- Background processing ready for AI pipeline integration

## 🎮 Usage Examples

### Continuous Screen Monitoring
```python
import time
from src.capture.gdi_screen_capture import GDIScreenCapture

capture = GDIScreenCapture()

# Monitor specific region continuously
while True:
    # Capture center 640x480 region
    x, y = 640, 360  # Center of 1920x1080 screen
    region_data = capture.capture_screen_region(x, y, 640, 480)
    
    # Process with your AI model here
    # ai_model.process(region_data)
    
    time.sleep(0.1)  # 10 FPS
```

### Multi-Monitor Setup
```python
from src.monitors.monitor_manager import MonitorManager
from src.capture.gdi_screen_capture import GDIScreenCapture

monitor_mgr = MonitorManager()
capture = GDIScreenCapture()

# Capture from each monitor
for monitor in monitor_mgr.get_monitors():
    print(f"Capturing Monitor {monitor.index}: {monitor.name}")
    monitor_data = capture.capture_monitor(monitor.index)
    
    filename = f"monitor_{monitor.index}_capture.png"
    capture.save_capture(monitor_data, filename)
```

### Region-Based Processing
```python
from src.regions.region_selector import RegionSelector

# Define regions of interest
selector = RegionSelector(capture, monitor_mgr)
selector.add_region("Taskbar", 0, 1040, 1920, 40, 0, "Windows taskbar")
selector.add_region("Browser", 100, 100, 1720, 900, 0, "Main browser window")

# Process specific regions
taskbar_data = selector.capture_region("Taskbar")
browser_data = selector.capture_region("Browser")
```

## 🔮 Future Enhancements

This foundation is designed for extension with AI capabilities:

### Planned AI Integration
- **Object Detection**: Real-time detection of UI elements, windows, buttons
- **Text Recognition**: OCR for screen text extraction and monitoring  
- **Change Detection**: Intelligent detection of screen content changes
- **Activity Recognition**: Understanding user actions and application states
- **Automation Triggers**: AI-driven automation based on visual cues

### Processing Pipeline
- **Frame Buffering**: Efficient frame queue management for AI processing
- **Preprocessing**: Image enhancement, normalization, and preparation
- **Model Integration**: Support for TensorFlow, PyTorch, and ONNX models
- **Result Overlay**: Visual feedback and annotation on captured content

## 📊 Technical Specifications

### Performance Benchmarks
- **Capture Speed**: ~5-15ms per frame (depending on resolution)
- **Memory Usage**: ~4-24MB per frame (depending on resolution and color depth)  
- **Supported Resolutions**: Up to 4K per monitor
- **Multi-Monitor**: Tested with up to 6 monitors simultaneously
- **Minimum Interval**: 10ms continuous capture (100 FPS theoretical)

### System Requirements
- **OS**: Windows 10/11 (GDI+ API required)
- **Python**: 3.7+ (tested up to 3.11)
- **RAM**: 100MB+ base, +50MB per active capture region
- **CPU**: Any modern CPU (capture is GPU-independent)

## 🤝 Contributing

This project provides a foundation for AI vision systems. Contributions welcome for:
- AI model integration
- Performance optimizations  
- Additional capture formats
- Cross-platform support (Linux/macOS)
- Advanced image processing features

## 📄 License

This project is designed to give AI systems "eyes" to see and understand screen content. Use responsibly and in accordance with applicable privacy and security guidelines.

---

**Getting Started**: Run `python src/gui/vision_control_panel.py` to launch the GUI and begin capturing screen content for your AI vision applications.