# AI Vision System - Complete Project Reconstruction Prompt

## Project Overview
Create a comprehensive AI Vision System for universal UI automation with LLM integration. This system captures screen content with perfect cursor timing synchronization and provides professional workflow recording capabilities for automation sequences.

## Core Architecture Requirements

### Three-Image System (CRITICAL - This is the breakthrough)
The system must implement a sophisticated three-image architecture to solve cursor timing synchronization:

1. **temp_view.jpg** - Live display image (no cursor baked in)
   - Used for viewer display with real-time cursor overlay
   - Updated every X seconds (configurable)
   - Provides smooth, non-jerky cursor movement in viewer

2. **nomouse_cursor.jpg** - Clean reference image for change detection
   - No cursor baked in
   - Used ONLY for pixel comparison to detect screen changes
   - Prevents false positives from cursor movement
   - Updated only when significant screen changes detected

3. **current_view.jpg** - LLM analysis image with cursor baked in
   - Contains cursor at exact moment of capture
   - Used by LLMs for automation analysis
   - Shows LLM precisely where cursor was when screen was captured
   - Automatically archived with timestamp

### Pixel Change Detection System
Implement two detection methods:

**Random Sampling Method:**
- Generate N random pixel coordinates (configurable: default 100)
- Compare RGB values at each coordinate between temp_view and nomouse_cursor
- Calculate percentage of changed pixels
- Trigger update if change percentage exceeds threshold (configurable: default 5%)

**Fixed Coordinates Method:**
- User-defined pixel coordinates in format: "x1,y1;x2,y2;x3,y3"
- Compare only specified pixels for change detection
- Useful for monitoring specific screen areas
- Fallback to random sampling if coordinates invalid

### Screen Capture Implementation
Use Windows GDI+ APIs for high-performance capture:

**Core Capture Features:**
- Primary monitor capture with double buffering to prevent flicker
- Cursor compositing using PIL (not GDI DrawIcon - doesn't work with memory contexts)
- Support for different cursor types: cross, cursor arrow, none
- Configurable cursor size (5-50px) and thickness (1-10px)
- Cursor colors: red for baked-in cursor, cyan for live overlay

**Cursor Implementation Details:**
- Get cursor position using Windows GetCursorInfo API
- Composite cursor onto image using PIL ImageDraw
- Cross type: Two perpendicular lines with configurable size/thickness
- Arrow type: Polygon arrow shape with outline
- Live overlay: Uses GetCursorPos for real-time position

## Window Management System

### Advanced Window Sizing Modes
Implement four distinct sizing modes:

1. **auto_fit**: Scale image to fit 90% of screen size, maintain aspect ratio
2. **width_based**: Set target width, calculate height from aspect ratio
3. **height_based**: Set target height, calculate width from aspect ratio
4. **fixed**: Use both dimensions, add letterboxing with black bars if needed

### Minimal Command-Window Interface
Create professional minimal viewer:
- Dark title bar (#2d2d30) with gear icon (⚙) for system menu
- Draggable title bar functionality
- Minimize (─) and close (✕) buttons
- Blue status bar (#007acc) with frame counter and service status indicator
- System menu with: Start/Stop/Restart Service, Properties, Calibration, Close

## Service Architecture

### Background Service Loop
Implement timing-controlled capture loop:

```python
def service_loop(self):
    last_temp_capture = 0
    last_current_capture = 0
    capture_interval = settings["capture_interval"]
    
    while service_running:
        current_time = time.time()
        
        # 1. Capture temp_view.jpg (no cursor) every interval
        if (current_time - last_temp_capture) >= capture_interval:
            screen_data = capture_temp_view()
            last_temp_capture = current_time
            
            # 2. Check if screen changed (temp_view vs nomouse_cursor)
            screen_changed = detect_screen_changes(screen_data)
            
            # 3. Update current_view.jpg with cursor every interval
            if (current_time - last_current_capture) >= capture_interval:
                capture_current_view_with_cursor()
                last_current_capture = current_time
        
        time.sleep(0.1)  # 10 FPS display updates
```

### Live Cursor Overlay System
For smooth viewer display:
- Monitor temp_view.jpg for changes every 100ms
- Get real-time cursor position using ctypes and GetCursorPos
- Draw cursor overlay on temp_view image before displaying
- Use bright cyan color to distinguish from baked-in cursor

## Properties Dialog System

### General Tab
- Output Path: Where current_view.jpg is saved
- Archive Folder: Where timestamped frames are stored
- Monitor Target: primary, secondary, all, window, region
- Window Sizing section with mode dropdown and target dimensions
- Help text explaining each sizing mode

### Cursor Tab
- Cursor Type: cross, cursor, none
- Cursor Size: Slider 5-50px
- Cursor Thickness: Slider 1-10px

### Capture & Detection Tab
- Capture Interval: Seconds between captures (default 2.0)
- Pixel Threshold: Percentage of pixels that must change (default 5.0)
- Detection Method: random or fixed_coords
- Sample Points: Number of random pixels to check (default 100)
- Fixed Coords: x1,y1;x2,y2;x3,y3 format for fixed method

## Floating Control Panel (Active Presenter Style)

### Professional Recording Interface
Create always-on-top floating panel:
- 300x80 pixel window, top-right corner positioning
- Dark theme (#2b2b2b background, #404040 title bar)
- Draggable by title bar
- No window decorations (overrideredirect=True)

### Control Buttons
- 🔴 Record: Start workflow recording, changes to "Recording..."
- ⏸️ Pause: Pause/Resume recording
- ⏹️ Stop: End recording and save workflow
- Status label showing current state
- Step counter showing current step number

### Recording Logic
Step-based recording system:
- Create timestamped session folder: automation_sessions/session_YYYYMMDD_HHMMSS
- Capture screen every 2 seconds (or on significant change)
- Save each step as step_001.png, step_002.png, etc.
- Record metadata: timestamp, cursor position, action type
- Generate workflow.json with complete session data

## File Structure and Data Management

### Directory Organization
```
ai-vision-system/
├── src/
│   └── capture/
│       └── gdi_screen_capture.py
├── claude_session/
│   ├── current_view.png      # LLM analysis image (with cursor)
│   ├── temp_view.jpg         # Live display image (no cursor)
│   ├── nomouse_cursor.jpg    # Reference image for change detection
│   └── archive/              # Timestamped frame archive
├── automation_sessions/
│   └── session_YYYYMMDD_HHMMSS/
│       ├── workflow.json
│       ├── step_001.png
│       ├── step_002.png
│       └── ...
├── viewer.pyw               # Main minimal viewer
├── automation_recorder.pyw  # Floating control panel
└── viewer_settings.json    # Persistent configuration
```

### Settings Persistence
JSON configuration with all user preferences:
- Window dimensions and sizing mode
- Cursor appearance settings
- Capture timing and detection parameters
- Output paths and archive locations

## Calibration System

### LLM-Friendly Calibration
Full-screen calibration without button requirements:
- Black background with red targets
- 3x3 grid of 9 calibration points
- Starts immediately when opened
- ESC key to cancel
- Saves accuracy results with timestamp
- Uses cursor size from settings for target sizing

### Calibration Data
Record precision metrics:
- Target coordinates vs actual click coordinates
- Distance accuracy measurements
- Timing data for each click
- Summary statistics (average accuracy, success rate)

## Technical Implementation Details

### Windows Integration
- Use ctypes for Windows API calls (GetCursorPos, GetCursorInfo)
- GDI+ APIs for screen capture (avoid GDI DrawIcon - doesn't work)
- Proper cleanup of GDI objects to prevent memory leaks
- Double buffering for flicker-free rendering

### Threading Architecture
- Main GUI thread for user interface
- Background service thread for screen capture
- Separate monitoring thread for file watching
- Thread-safe communication using queues

### Error Handling
- Graceful fallbacks for capture failures
- Settings validation and defaults
- File system error recovery
- Pixel detection algorithm fallbacks

### Performance Optimization
- Efficient pixel sampling for change detection
- Image resizing with proper aspect ratio maintenance
- Memory management for large screen captures
- Configurable timing to balance responsiveness vs resource usage

## Commercial Considerations

### Future Development Path
This Python prototype proves the concept. For commercial release:
- Port to C#/.NET for professional Windows deployment
- Enhanced obfuscation and IP protection
- Enterprise licensing and deployment tools
- Advanced workflow editing and annotation features

### Key Value Propositions
- Universal UI automation (works with any application)
- LLM integration for intelligent automation
- Visual workflow recording and playback
- No API dependencies or limitations
- Professional screen recording capabilities

## Critical Success Factors

### The Cursor Timing Breakthrough
The three-image system solves the fundamental problem:
- LLMs get accurate cursor position at capture moment
- Users see smooth real-time cursor movement  
- Change detection works without cursor interference
- Perfect synchronization between display and analysis

### Architecture Lessons Learned
- Never use GDI DrawIcon for cursor compositing (use PIL)
- Separate display from analysis images for optimal performance
- Random pixel sampling is more reliable than full image comparison
- Real-time overlay provides smooth user experience
- Step-based recording is more valuable than continuous video

This system represents years of UI automation experience distilled into a working solution. The core logic and architecture are proven - implementation in any framework will be straightforward using these specifications.






