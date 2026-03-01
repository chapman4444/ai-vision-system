# Mouse Calibration Instructions for LLMs

## Overview
This calibration system helps LLMs learn accurate mouse clicking by measuring systematic offset errors across the entire screen.

## Quick Start - Single Command

```bash
cd claude_session
python smart_auto_calibrate.py
```

**That's it!** This single command:
- Launches the fullscreen calibration GUI
- Waits for it to initialize
- Automatically finds and clicks all red targets
- Saves calibration results to JSON file
- No manual steps required

## What You Get

### Results File
After completion, find file: `fullscreen_calibration_[timestamp].json`

### Key Data Points
```json
{
  "summary": {
    "avg_offset_x": -55.1,  // Click 55px RIGHT to compensate
    "avg_offset_y": -6.5,   // Click 6px DOWN to compensate  
    "avg_accuracy": 78.2,   // Overall accuracy percentage
    "hit_rate": 85.0        // Percentage of successful hits
  }
}
```

## Using Calibration Data

### Apply Offset Correction
```python
# Read calibration results
with open('fullscreen_calibration_[timestamp].json', 'r') as f:
    cal_data = json.load(f)

offset_x = cal_data['summary']['avg_offset_x'] 
offset_y = cal_data['summary']['avg_offset_y']

# When clicking at target coordinates:
def calibrated_click(target_x, target_y):
    # Apply offset correction
    corrected_x = target_x - offset_x  # Subtract because offset is error
    corrected_y = target_y - offset_y
    
    # Perform actual click
    pyautogui.click(corrected_x, corrected_y)

# Example: Want to click at (500, 300)
# If offset is (-55, -6), actually click at (555, 306)
calibrated_click(500, 300)  # Clicks at (555, 306)
```

## Files in Session Directory

### Core Files
- `fullscreen_calibration.py` - Main GUI application
- `smart_auto_calibrate.py` - Automated clicking system
- `auto_calibrate.py` - Basic version (less accurate)
- `mouse_calibration_gui.py` - Windowed version

### Generated Files
- `fullscreen_calibration_*.json` - Results data
- `calibration_results_screenshot.png` - Visual results

## Troubleshooting

### GUI Doesn't Start
- Check Python is installed: `python --version`
- Install dependencies: `pip install pyautogui pillow tkinter`

### Automation Doesn't Click
- Install NumPy: `pip install numpy`
- Ensure GUI is running first
- Wait for auto-start (3 seconds)

### Low Accuracy Results
- Normal for first run - use offset data to improve
- Multiple calibration runs can refine accuracy
- Check screen scaling settings (100% recommended)

## Target Pattern

The calibration tests these screen positions:
- **Corners**: (50,50), (1870,50), (50,1030), (1870,1030)  
- **Edges**: Top/bottom/left/right centers
- **Center**: (960, 540) for 1920x1080
- **Grid**: Systematic coverage across display
- **Random**: Additional challenge positions

## Understanding Results

### Accuracy Levels
- **90%+** : Excellent - very close to target center
- **70-89%**: Good - within target radius
- **50-69%**: Fair - near target but outside ideal range
- **<50%**: Poor - significant miss distance

### Offset Interpretation
- **Negative X**: Clicks tend to be LEFT of target
- **Positive X**: Clicks tend to be RIGHT of target  
- **Negative Y**: Clicks tend to be ABOVE target
- **Positive Y**: Clicks tend to be BELOW target

## Advanced Usage

### Multiple Runs
Run calibration multiple times and average the offsets:
```python
offsets_x = [-55.1, -52.3, -58.7]  # Multiple runs
offsets_y = [-6.5, -4.2, -8.1]

avg_offset_x = sum(offsets_x) / len(offsets_x)
avg_offset_y = sum(offsets_y) / len(offsets_y)
```

### Screen-Specific Calibration
Different screens may need different calibrations:
- Run on each display you'll use
- Save results per screen resolution
- Apply appropriate offset per display

## Integration Example

```python
import json
import pyautogui

class CalibratedMouseControl:
    def __init__(self, calibration_file):
        with open(calibration_file, 'r') as f:
            cal_data = json.load(f)
        
        self.offset_x = cal_data['summary']['avg_offset_x']
        self.offset_y = cal_data['summary']['avg_offset_y']
        
    def click(self, x, y):
        corrected_x = x - self.offset_x
        corrected_y = y - self.offset_y
        pyautogui.click(corrected_x, corrected_y)
        
    def get_accuracy_info(self):
        return {
            'offset_x': self.offset_x,
            'offset_y': self.offset_y,
            'correction': f"Adjust clicks by ({-self.offset_x:+.0f}, {-self.offset_y:+.0f}) pixels"
        }

# Usage
mouse = CalibratedMouseControl('fullscreen_calibration_1755007260.json')
mouse.click(500, 300)  # Automatically applies calibration offset
print(mouse.get_accuracy_info())
```

## Tips for LLMs

1. **Run calibration first** before any automation tasks
2. **Save the offset values** for your session
3. **Apply offsets consistently** to all click commands
4. **Re-calibrate periodically** for best accuracy
5. **Test on different screen areas** to validate correction

## Support Files

All calibration files are designed to work together in the `claude_session` directory. Keep all files in this location for proper operation.