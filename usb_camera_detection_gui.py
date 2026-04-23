"""
USB Camera Object Detection with YOLO - GUI Camera Selector
Features a graphical interface with dropdown menu to select cameras
"""

import cv2
import torch
from ultralytics import YOLO
from datetime import datetime
import os
import tkinter as tk
from tkinter import ttk, messagebox
import threading

# Configuration
MODEL_NAME = 'yolov8n.pt'
CONFIDENCE_THRESHOLD = 0.5
OUTPUT_FILE = 'detections.txt'
SAVE_IMAGES = True
BASE_FOLDER = 'detected_objects'
SAVE_FULL_FRAME = False
FULL_FRAME_FOLDER = 'full_frames'
MIN_DETECTION_INTERVAL = 1.0

# Colors for bounding boxes (BGR format)
BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)
TEXT_BG_COLOR = (0, 255, 0)

# Track last detection time for each object type
last_detection_time = {}

class CameraSelector:
    """GUI window for camera selection"""
    
    def __init__(self):
        self.selected_camera = None
        self.cameras = []
        self.root = tk.Tk()
        self.root.title("YOLO Object Detection - Camera Selector")
        self.root.geometry("500x400")
        self.root.resizable(False, False)
        
        # Set window icon and style
        self.root.configure(bg='#f0f0f0')
        
        # Detect cameras
        self.detect_cameras()
        
        # Build GUI
        self.build_gui()
        
    def detect_cameras(self, max_cameras=10):
        """Detect all available cameras"""
        print("🔍 Scanning for available cameras...")
        
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                ret, frame = cap.read()
                if ret:
                    camera_info = {
                        'index': i,
                        'width': width,
                        'height': height,
                        'fps': fps,
                        'name': f"Camera {i} ({width}x{height} @ {fps}fps)"
                    }
                    self.cameras.append(camera_info)
                    print(f"  ✓ Found: {camera_info['name']}")
                cap.release()
        
        if not self.cameras:
            messagebox.showerror("Error", "No cameras detected!\n\nPlease check:\n• Camera is connected\n• Drivers are installed\n• Camera is not in use by another app")
            self.root.destroy()
    
    def build_gui(self):
        """Build the GUI interface"""
        
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill='x')
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="🎯 YOLO Object Detection",
            font=('Arial', 20, 'bold'),
            fg='white',
            bg='#2c3e50'
        )
        title_label.pack(pady=20)
        
        # Main content frame
        content_frame = tk.Frame(self.root, bg='#f0f0f0')
        content_frame.pack(fill='both', expand=True, padx=30, pady=20)
        
        # Camera selection label
        select_label = tk.Label(
            content_frame,
            text="Select Camera:",
            font=('Arial', 12, 'bold'),
            bg='#f0f0f0'
        )
        select_label.pack(anchor='w', pady=(0, 5))
        
        # Dropdown menu
        self.camera_var = tk.StringVar()
        camera_dropdown = ttk.Combobox(
            content_frame,
            textvariable=self.camera_var,
            values=[cam['name'] for cam in self.cameras],
            state='readonly',
            font=('Arial', 11),
            width=45
        )
        camera_dropdown.pack(pady=(0, 20), ipady=5)
        
        # Set default selection
        if self.cameras:
            camera_dropdown.current(0)
        
        # Camera info display
        info_frame = tk.LabelFrame(
            content_frame,
            text="Camera Details",
            font=('Arial', 10, 'bold'),
            bg='#f0f0f0',
            padx=15,
            pady=10
        )
        info_frame.pack(fill='x', pady=(0, 20))
        
        self.info_text = tk.Text(
            info_frame,
            height=4,
            width=50,
            font=('Courier', 9),
            bg='#ffffff',
            relief='flat',
            borderwidth=0
        )
        self.info_text.pack()
        
        # Update info when selection changes
        camera_dropdown.bind('<<ComboboxSelected>>', self.update_camera_info)
        self.update_camera_info()
        
        # Buttons frame
        button_frame = tk.Frame(content_frame, bg='#f0f0f0')
        button_frame.pack(pady=10)
        
        # Preview button
        preview_btn = tk.Button(
            button_frame,
            text="📸 Preview Camera",
            command=self.preview_camera,
            font=('Arial', 11),
            bg='#3498db',
            fg='white',
            activebackground='#2980b9',
            activeforeground='white',
            relief='flat',
            padx=20,
            pady=10,
            cursor='hand2'
        )
        preview_btn.pack(side='left', padx=5)
        
        # Start button
        start_btn = tk.Button(
            button_frame,
            text="▶️ Start Detection",
            command=self.start_detection,
            font=('Arial', 11, 'bold'),
            bg='#27ae60',
            fg='white',
            activebackground='#229954',
            activeforeground='white',
            relief='flat',
            padx=20,
            pady=10,
            cursor='hand2'
        )
        start_btn.pack(side='left', padx=5)
        
        # Status bar
        status_frame = tk.Frame(self.root, bg='#34495e', height=30)
        status_frame.pack(fill='x', side='bottom')
        status_frame.pack_propagate(False)
        
        status_label = tk.Label(
            status_frame,
            text=f"✓ {len(self.cameras)} camera(s) detected | Ready to start",
            font=('Arial', 9),
            fg='white',
            bg='#34495e'
        )
        status_label.pack(pady=5)
    
    def update_camera_info(self, event=None):
        """Update camera info display"""
        selected_name = self.camera_var.get()
        selected_cam = next((cam for cam in self.cameras if cam['name'] == selected_name), None)
        
        if selected_cam:
            info = f"""
Camera Index:  {selected_cam['index']}
Resolution:    {selected_cam['width']} x {selected_cam['height']} pixels
Frame Rate:    {selected_cam['fps']} FPS
Status:        Ready
            """.strip()
            
            self.info_text.delete('1.0', 'end')
            self.info_text.insert('1.0', info)
    
    def preview_camera(self):
        """Show preview of selected camera"""
        selected_name = self.camera_var.get()
        selected_cam = next((cam for cam in self.cameras if cam['name'] == selected_name), None)
        
        if not selected_cam:
            messagebox.showwarning("Warning", "Please select a camera first")
            return
        
        camera_index = selected_cam['index']
        
        # Open camera in new thread to avoid blocking GUI
        def show_preview():
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                messagebox.showerror("Error", f"Could not open Camera {camera_index}")
                return
            
            window_name = f"Camera {camera_index} Preview"
            cv2.namedWindow(window_name)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Add preview text
                cv2.putText(frame, f"Camera {camera_index} Preview", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, "Press ESC to close", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow(window_name, frame)
                
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break
            
            cap.release()
            cv2.destroyAllWindows()
        
        preview_thread = threading.Thread(target=show_preview, daemon=True)
        preview_thread.start()
    
    def start_detection(self):
        """Start object detection with selected camera"""
        selected_name = self.camera_var.get()
        selected_cam = next((cam for cam in self.cameras if cam['name'] == selected_name), None)
        
        if not selected_cam:
            messagebox.showwarning("Warning", "Please select a camera first")
            return
        
        self.selected_camera = selected_cam['index']
        self.root.destroy()
    
    def run(self):
        """Run the GUI"""
        self.root.mainloop()
        return self.selected_camera

def setup_output_files():
    """Create output directory and file"""
    if SAVE_IMAGES:
        if not os.path.exists(BASE_FOLDER):
            os.makedirs(BASE_FOLDER)
        if SAVE_FULL_FRAME and not os.path.exists(FULL_FRAME_FOLDER):
            os.makedirs(FULL_FRAME_FOLDER)
    
    with open(OUTPUT_FILE, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("OBJECT DETECTION LOG\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

def log_detection(obj_name, confidence, timestamp, image_path):
    """Save detection to text file"""
    with open(OUTPUT_FILE, 'a') as f:
        f.write(f"[{timestamp}] {obj_name} (confidence: {confidence:.2%}) - Saved: {image_path}\n")

def save_object_image(frame, bbox, class_name, confidence, timestamp):
    """Save cropped image of detected object"""
    if not SAVE_IMAGES:
        return None
    
    x1, y1, x2, y2 = bbox
    height, width = frame.shape[:2]
    padding_x = int((x2 - x1) * 0.05)
    padding_y = int((y2 - y1) * 0.05)
    
    x1_pad = max(0, x1 - padding_x)
    y1_pad = max(0, y1 - padding_y)
    x2_pad = min(width, x2 + padding_x)
    y2_pad = min(height, y2 + padding_y)
    
    cropped = frame[y1_pad:y2_pad, x1_pad:x2_pad]
    
    if cropped.size == 0:
        return None
    
    object_folder = os.path.join(BASE_FOLDER, class_name)
    if not os.path.exists(object_folder):
        os.makedirs(object_folder)
        print(f"✓ Created folder: {object_folder}")
    
    timestamp_str = timestamp.replace(':', '-').replace(' ', '_')
    confidence_str = f"{confidence:.0%}".replace('%', 'pct')
    filename = f"{timestamp_str}_{confidence_str}.jpg"
    filepath = os.path.join(object_folder, filename)
    
    cv2.imwrite(filepath, cropped)
    return filepath

def draw_detections(frame, results):
    """Draw bounding boxes and labels on frame"""
    detections_this_frame = []
    current_time = datetime.now()
    
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            
            if confidence >= CONFIDENCE_THRESHOLD:
                should_save = True
                if class_name in last_detection_time:
                    time_since_last = (current_time - last_detection_time[class_name]).total_seconds()
                    if time_since_last < MIN_DETECTION_INTERVAL:
                        should_save = False
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
                
                label = f"{class_name} {confidence:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                cv2.rectangle(frame, (x1, y1 - text_height - 10),
                            (x1 + text_width, y1), TEXT_BG_COLOR, -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
                
                detections_this_frame.append({
                    'name': class_name,
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2),
                    'should_save': should_save
                })
    
    return frame, detections_this_frame

def run_detection(camera_index):
    """Main detection loop"""
    print("\n" + "="*60)
    print("STARTING OBJECT DETECTION".center(60))
    print("="*60)
    
    setup_output_files()
    
    # Load YOLO model
    print(f"\nLoading YOLO model: {MODEL_NAME}...")
    try:
        model = YOLO(MODEL_NAME)
        if torch.cuda.is_available():
            model.to('cuda')
            print("✓ Using GPU for inference")
        else:
            print("⚠️  Using CPU")
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Open camera
    print(f"\nOpening Camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera {camera_index}")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("✓ Camera opened successfully")
    print("\n" + "="*60)
    print("CONTROLS:")
    print("  Q or ESC - Quit")
    print("  S - Force save all current detections")
    print("  SPACE - Pause/Resume")
    print("  C - Clear detection rate limit")
    print("="*60)
    print(f"\nUsing Camera {camera_index}")
    print(f"Saving to: {BASE_FOLDER}/<object_type>/")
    print("="*60 + "\n")
    
    total_detections = 0
    total_images_saved = 0
    detection_set = set()
    object_counts = {}
    paused = False
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = model(frame, verbose=False)
                frame_display, detections = draw_detections(frame, results)
                
                if detections:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    current_time = datetime.now()
                    
                    for det in detections:
                        total_detections += 1
                        detection_set.add(det['name'])
                        
                        if det['name'] not in object_counts:
                            object_counts[det['name']] = 0
                        object_counts[det['name']] += 1
                        
                        if det['should_save']:
                            image_path = save_object_image(
                                frame, det['bbox'], det['name'],
                                det['confidence'], timestamp
                            )
                            
                            if image_path:
                                log_detection(det['name'], det['confidence'],
                                            timestamp, image_path)
                                last_detection_time[det['name']] = current_time
                                total_images_saved += 1
                                print(f"💾 Saved: {det['name']} ({det['confidence']:.0%}) -> {image_path}")
            else:
                frame_display = frame.copy()
            
            # Info overlay
            info = f"Camera {camera_index} | Detections: {total_detections} | Saved: {total_images_saved} | Types: {len(detection_set)}"
            cv2.putText(frame_display, info, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if paused:
                cv2.putText(frame_display, "PAUSED", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('YOLO Object Detection', frame_display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                if 'detections' in locals() and detections:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    for det in detections:
                        image_path = save_object_image(
                            frame, det['bbox'], det['name'],
                            det['confidence'], timestamp
                        )
                        if image_path:
                            print(f"💾 Force saved: {det['name']} -> {image_path}")
                            total_images_saved += 1
            elif key == ord(' '):
                paused = not paused
                print(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")
            elif key == ord('c'):
                last_detection_time.clear()
                print("🔄 Rate limit cleared")
    
    except KeyboardInterrupt:
        print("\n👋 Stopped by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Summary
        with open(OUTPUT_FILE, 'a') as f:
            f.write("\n" + "="*80 + "\n")
            f.write("DETECTION SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"Camera: {camera_index}\n")
            f.write(f"Total detections: {total_detections}\n")
            f.write(f"Images saved: {total_images_saved}\n")
            f.write(f"Unique types: {len(detection_set)}\n")
            f.write(f"\nCounts:\n")
            for obj in sorted(object_counts.keys()):
                f.write(f"  {obj}: {object_counts[obj]}\n")
            f.write(f"\nEnded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n")
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Camera: {camera_index}")
        print(f"Detections: {total_detections}")
        print(f"Images saved: {total_images_saved}")
        print(f"Unique objects: {len(detection_set)}")
        
        if object_counts:
            print("\nCounts:")
            for obj in sorted(object_counts.keys()):
                folder = os.path.join(BASE_FOLDER, obj)
                num = len([f for f in os.listdir(folder) if f.endswith('.jpg')]) if os.path.exists(folder) else 0
                print(f"  {obj}: {object_counts[obj]} detected, {num} saved")
        
        print(f"\n✓ Log: {OUTPUT_FILE}")
        print(f"✓ Images: {BASE_FOLDER}/<object_type>/")
        print("="*60 + "\n")

def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("YOLO OBJECT DETECTION - GUI VERSION".center(60))
    print("="*60 + "\n")
    
    # Show GUI camera selector
    selector = CameraSelector()
    camera_index = selector.run()
    
    if camera_index is not None:
        # Start detection with selected camera
        run_detection(camera_index)
    else:
        print("No camera selected. Exiting...")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
