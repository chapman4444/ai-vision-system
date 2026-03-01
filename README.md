# AI Vision System

A high-performance Windows screen capture system designed for AI integration. Built with CLI-first architecture and optional GUI interface.

## 🚀 Quick Start

```bash
# First time setup
run setup

# Start background vision service  
run start

# Take a quick screenshot
run capture "analyze this screen"

# Show screen to Claude
run show "what do you see here?"

# Stop service
run stop
```

## 📁 Project Structure

```
ai-vision-system/
├── src/                          # Source code
│   ├── capture/                  # Screen capture implementations
│   ├── monitors/                 # Monitor detection and management
│   ├── regions/                  # Interactive region selection
│   ├── gui/                      # GUI applications
│   ├── interface/                # Claude integration layer
│   └── session/                  # Session management
├── scripts/                      # Batch scripts for easy usage
│   ├── RUN_ME_FIRST.bat         # Initial setup
│   ├── quick_commands.bat        # Quick screenshot commands
│   ├── session_commands.bat      # Claude session integration
│   └── START_VISION.bat          # Background service control
├── docs/                         # Documentation
├── tests/                        # Unit tests
├── claude_session/               # Images for Claude to view
├── claude_workspace/             # Local image storage
└── ai_vision_interface/          # File-based communication
```

## ✨ Features

### Core Capabilities
- **High-Performance Screen Capture**: Direct Windows GDI+ API access
- **Multi-Monitor Support**: Automatic detection and selection
- **Region Selection**: Interactive selection of specific screen areas
- **Claude Code Integration**: File-based communication system

### Advanced Features
- **Background Service**: Continuous capture with configurable intervals
- **Session Management**: Persistent session state and history
- **Real-Time Processing**: Continuous capture for AI pipelines
- **Interactive GUI**: Vision control panel for advanced usage

## 🎯 Usage Scenarios

### 1. "Claude, look at this!"
```bash
scripts\quick_commands.bat show "I'm having trouble with this error"
```
- Instantly captures your screen
- Claude receives image automatically
- Perfect for debugging help

### 2. Code Review Mode
```bash
scripts\session_commands.bat setup
scripts\session_commands.bat show "Please review this function"
```
- Files automatically sync to Claude Code session
- Seamless integration workflow

### 3. Continuous Monitoring
```bash
scripts\START_VISION.bat
# Background service captures every few seconds
# Claude sees updates automatically in claude_session/current_view.png
scripts\STOP_VISION.bat
```

## 🛠️ Development

### Running Tests
```bash
python tests/test_capture.py
```

### GUI Application
```bash
python src/gui/vision_control_panel.py
```

### Python API
```python
from src.capture.gdi_screen_capture import GDIScreenCapture

capture = GDIScreenCapture()
screen_data = capture.capture_primary_monitor()
capture.save_capture(screen_data, "screenshot.png")
```

## 📖 Documentation

- [Claude Integration Guide](docs/claude-integration.md)
- [Session Sync Setup](docs/session-sync.md)
- [Claude Instructions](docs/claude-instructions.md)

## 🔧 System Requirements

- **OS**: Windows 10/11 (GDI+ API required)
- **Python**: 3.7+ (tested up to 3.11)
- **RAM**: 100MB+ base, +50MB per active capture region

## 📄 License

This project provides AI systems with "eyes" to see and understand screen content. Use responsibly and in accordance with applicable privacy and security guidelines.

---

**Getting Started**: Run `scripts\RUN_ME_FIRST.bat` to set up the system, then use `scripts\quick_commands.bat help` to see available commands.