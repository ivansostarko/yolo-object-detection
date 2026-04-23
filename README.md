# 🎯 YOLO Object Detection - USB Camera

Real-time object detection using YOLOv8 and your USB camera. Automatically detects objects, saves cropped images organized by type, and logs all detections with timestamps.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![YOLO](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)

---

## 📸 Demo

The application opens a GUI window where you can:
- Select from available cameras via dropdown menu
- Preview cameras before starting
- Start real-time object detection
- Automatically save detected objects to organized folders

---

## ✨ Features

### 🎨 **GUI Camera Selector**
- Beautiful graphical interface
- Dropdown menu showing all detected cameras
- Camera details display (resolution, FPS)
- Live preview before starting

### 🔍 **Real-time Object Detection**
- Powered by YOLOv8 (state-of-the-art object detection)
- GPU acceleration support
- 80+ object classes (people, animals, vehicles, household items, etc.)
- Live bounding boxes with confidence scores

### 💾 **Smart Image Saving**
- Automatically crops and saves each detected object
- Organizes images by object type in separate folders
- Rate limiting to prevent duplicate saves
- Timestamp and confidence in filenames

### 📊 **Detailed Logging**
- Text file log of all detections
- Timestamps and confidence scores
- Summary statistics at the end
- Detection counts per object type

### ⚡ **Performance**
- GPU support (NVIDIA CUDA)
- CPU fallback
- Configurable detection interval
- Adjustable confidence threshold

---

## 📋 Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, Linux, or macOS
- **Camera**: Any USB camera or built-in webcam
- **Optional**: NVIDIA GPU with CUDA for faster processing

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ivansostarko/yolo-object-detection.git
cd yolo-object-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install ultralytics opencv-python torch
```

### 3. Run the Application

```bash
python usb_camera_detection_gui.py
```

---

## 📖 Usage

### Quick Start

1. **Run the script:**
   ```bash
   python usb_camera_detection_gui.py
   ```

2. **Select your camera** from the dropdown menu

3. **Optional:** Click "Preview Camera" to verify

4. **Click "Start Detection"** to begin

5. **Use keyboard controls:**
   - `Q` or `ESC` - Quit
   - `S` - Force save current detections
   - `SPACE` - Pause/Resume
   - `C` - Clear rate limit

### Output Structure

After running, you'll have:

```
project_folder/
├── detected_objects/          # Cropped object images
│   ├── person/
│   │   ├── 2026-04-23_14-30-16_95pct.jpg
│   │   ├── 2026-04-23_14-30-45_87pct.jpg
│   │   └── ...
│   ├── laptop/
│   │   ├── 2026-04-23_14-30-18_89pct.jpg
│   │   └── ...
│   ├── cell_phone/
│   └── cup/
└── detections.txt             # Log file with all detections
```

### Example Log File

```
================================================================================
OBJECT DETECTION LOG
Started: 2026-04-23 14:30:15
================================================================================

[2026-04-23 14:30:16] person (confidence: 95.32%) - Saved: detected_objects/person/2026-04-23_14-30-16_95pct.jpg
[2026-04-23 14:30:16] laptop (confidence: 87.45%) - Saved: detected_objects/laptop/2026-04-23_14-30-16_87pct.jpg
[2026-04-23 14:30:17] cell phone (confidence: 76.21%) - Saved: detected_objects/cell_phone/2026-04-23_14-30-17_76pct.jpg

================================================================================
DETECTION SUMMARY
================================================================================
Camera: 0
Total detections: 156
Images saved: 48
Unique types: 8
Objects detected: cell phone, cup, keyboard, laptop, mouse, person, book, bottle
Ended: 2026-04-23 14:45:30
================================================================================
```

---

## ⚙️ Configuration

Edit these variables at the top of `usb_camera_detection_gui.py`:

```python
# Model selection (speed vs accuracy tradeoff)
MODEL_NAME = 'yolov8n.pt'  # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x

# Detection settings
CONFIDENCE_THRESHOLD = 0.5  # Only save detections above 50% confidence
MIN_DETECTION_INTERVAL = 1.0  # Seconds between saving same object type

# Output settings
SAVE_IMAGES = True  # Save cropped object images
BASE_FOLDER = 'detected_objects'  # Folder for saved images
OUTPUT_FILE = 'detections.txt'  # Log file name

# Optional: Save full frames with bounding boxes
SAVE_FULL_FRAME = False
FULL_FRAME_FOLDER = 'full_frames'
```

### YOLO Model Options

| Model | Speed | Accuracy | Size | Best For |
|-------|-------|----------|------|----------|
| `yolov8n.pt` | ⚡⚡⚡ Fastest | Good | 6 MB | Real-time, low-end hardware |
| `yolov8s.pt` | ⚡⚡ Fast | Better | 22 MB | Balanced performance |
| `yolov8m.pt` | ⚡ Medium | Great | 52 MB | Good GPU |
| `yolov8l.pt` | Slow | Excellent | 87 MB | Powerful GPU |
| `yolov8x.pt` | Slowest | Best | 131 MB | Maximum accuracy |

---

## 🎮 Keyboard Controls

While detection is running:

| Key | Action |
|-----|--------|
| **Q** or **ESC** | Quit and save summary |
| **S** | Force save all current detections (ignore rate limit) |
| **SPACE** | Pause/Resume detection |
| **C** | Clear rate limit (allow immediate saves) |

---

## 🔧 Troubleshooting

### Camera Not Detected

**Problem:** "No cameras detected" error

**Solutions:**
1. Check if camera is physically connected
2. Close other applications using the camera (Zoom, Teams, etc.)
3. Update camera drivers
4. Try running as administrator (Windows)
5. Check camera permissions in system settings

### Low FPS / Laggy Performance

**Problem:** Detection is slow or choppy

**Solutions:**
1. Use a faster model: `MODEL_NAME = 'yolov8n.pt'`
2. Check if GPU is being used (run `gpu_diagnostic.py`)
3. Lower camera resolution in code:
   ```python
   cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
   ```
4. Increase `MIN_DETECTION_INTERVAL` to reduce processing

### Too Many/Few Images Saved

**Problem:** Saving too many or too few images

**Solutions:**

**Too many images:**
```python
MIN_DETECTION_INTERVAL = 3.0  # Save less frequently
CONFIDENCE_THRESHOLD = 0.7    # Only high-confidence detections
```

**Too few images:**
```python
MIN_DETECTION_INTERVAL = 0.5  # Save more frequently
CONFIDENCE_THRESHOLD = 0.3    # Lower confidence threshold
```

### GPU Not Working

**Problem:** "Using CPU" message appears

**Solutions:**
1. Install NVIDIA drivers
2. Install CUDA toolkit
3. Reinstall PyTorch with CUDA:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
4. Run the diagnostic script to verify GPU setup

---

## 📊 Detectable Objects

YOLOv8 can detect 80 common objects including:

**People & Animals:** person, cat, dog, horse, bird, sheep, cow, elephant, bear, zebra, giraffe

**Vehicles:** car, motorcycle, bus, truck, bicycle, airplane, boat, train

**Indoor Objects:** laptop, keyboard, mouse, cell phone, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

**Furniture:** chair, couch, bed, dining table, toilet

**Kitchen:** bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake

**Sports:** sports ball, baseball bat, baseball glove, skateboard, surfboard, tennis racket

**Outdoor:** traffic light, fire hydrant, stop sign, parking meter, bench

**Full list:** [COCO dataset classes](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/)

---

## 💡 Use Cases

### 🏠 Home Automation
Monitor your desk and track what objects appear throughout the day

### 🔒 Security
Detect when people or vehicles enter camera view

### 🐾 Wildlife Monitoring
Automatically capture images of animals in your backyard

### 📦 Inventory Management
Track items on shelves or workspaces

### 🎓 Education & Research
Collect datasets for machine learning projects

### 🎨 Creative Projects
Build interactive installations or art projects

---

## 🖥️ System Requirements

### Minimum Requirements
- **CPU:** Intel Core i3 or equivalent
- **RAM:** 4 GB
- **Storage:** 500 MB free space
- **Camera:** Any USB webcam
- **Python:** 3.8+

### Recommended Requirements
- **CPU:** Intel Core i5 or better
- **RAM:** 8 GB
- **GPU:** NVIDIA GPU with CUDA support
- **Storage:** 2 GB free space
- **Camera:** 720p or higher resolution

---

## 🛠️ Advanced Customization

### Filter Specific Objects

Only save certain types of objects:

```python
# Add this in the detection loop
OBJECTS_TO_SAVE = ['person', 'car', 'dog', 'cat']

if class_name not in OBJECTS_TO_SAVE:
    continue  # Skip this object
```

### Different Intervals Per Object

```python
DETECTION_INTERVALS = {
    'person': 2.0,   # Save people every 2 seconds
    'car': 5.0,      # Save cars every 5 seconds
    'bird': 0.5,     # Save birds more frequently (fast moving)
}

interval = DETECTION_INTERVALS.get(class_name, MIN_DETECTION_INTERVAL)
```

### Add Custom Actions

Execute code when specific objects are detected:

```python
if class_name == 'person':
    # Send notification, trigger alarm, etc.
    print("Person detected!")
```

---

## 📚 Additional Scripts

### GPU Diagnostic Tool

Check if your GPU is properly configured:

```bash
python gpu_diagnostic.py
```

Features:
- Detects NVIDIA GPU
- Verifies CUDA installation
- Tests PyTorch GPU support
- Runs performance benchmarks

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Ideas for Contributions
- Add support for video file input
- Implement object tracking across frames
- Add email/SMS notifications
- Create a web interface
- Add more configuration options
- Improve GUI design
- Add multi-language support

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Ultralytics** - For the amazing YOLOv8 implementation
- **OpenCV** - For computer vision tools
- **PyTorch** - For deep learning framework
- **COCO Dataset** - For object detection training data

---

## 📧 Contact

- **Issues:** [GitHub Issues](https://github.com/ivansostarko/yolo-object-detection/issues)
- **Discussions:** [GitHub Discussions](https://github.com/ivansostarko/yolo-object-detection/discussions)

---

## 🌟 Star History

If you find this project useful, please consider giving it a star! ⭐

---

## 📈 Roadmap

- [ ] Video file input support
- [ ] Real-time object tracking
- [ ] Web interface
- [ ] Email/SMS notifications
- [ ] Cloud storage integration
- [ ] Mobile app
- [ ] Custom model training support
- [ ] Multi-camera support
- [ ] Motion detection zones

---

## 🔐 Privacy & Security

- **Local Processing:** All detection happens on your computer
- **No Internet Required:** Works completely offline
- **No Data Collection:** Nothing is sent to external servers
- **Full Control:** You control what is saved and where

---

## ⚖️ Disclaimer

This software is provided for educational and personal use. When using cameras:
- Respect privacy laws in your jurisdiction
- Obtain consent when recording others
- Follow workplace policies regarding surveillance
- Use responsibly and ethically

---

