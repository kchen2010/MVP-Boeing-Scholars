"""
Spatial-Only Inpainting Pipeline (YOLO + LaMa)
- Processes a list of input .mp4 files.
- Uses YOLO for segmentation.
- Uses LaMa for single-frame spatial inpainting.
- No temporal refinement or complex blending.
"""

import cv2
import torch
import numpy as np
import time
import os
from ultralytics import YOLO

# Check for LaMa
try:
    from simple_lama_inpainting import SimpleLama
    LAMA_AVAILABLE = True
except ImportError:
    print("⚠ Installing LaMa: pip install simple-lama-inpainting")
    exit(1)

# --- Constants ---
YOLO_MODEL = 'yolo11n-seg.pt'

# -----------------------------------------------------------------
# --- DEFINE YOUR INPUT/OUTPUT FILES HERE ---
# -----------------------------------------------------------------
VIDEO_FILES_IN = [
    'content/TestVideo1.mp4',
    'content/TestVideo2.mp4'
]
OUTPUT_FOLDER = 'output'
OUTPUT_SUFFIX = '_lama_only'
# -----------------------------------------------------------------

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Processing parameters
MODEL_IN_H, MODEL_IN_W = 240, 432
USE_LAMA = True       # Toggle spatial inpainting
DEBUG_MODE = False    # Set to False for clean output

print(f"=== Spatial Inpainting System ===")
print(f"Device: {DEVICE}")
print(f"Pipeline: YOLO -> LaMa")

# 1. Load YOLO
print("Loading YOLO...")
yolo = YOLO(YOLO_MODEL)
yolo.to(DEVICE)

# 2. Load LaMa
print("Loading LaMa...")
lama = SimpleLama()

print("✓ Models loaded\n")

# Check if files exist
valid_files = [f for f in VIDEO_FILES_IN if os.path.exists(f)]
if not valid_files:
    print(f"⚠ Error: No valid video files found in {VIDEO_FILES_IN}")
    exit(1)

print(f"Found {len(valid_files)} video file(s) to process...")

print("\n--- Controls ---")
print("Press 'q' to skip to the next video.")
print("Press 'd' to toggle debug view.")
print("Press '1' to toggle LaMa.")
print("-----------------\n")

# --- MAIN PROCESSING LOOP ---
for video_path in valid_files:
    print(f"\n--- Processing: {video_path} ---")

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠ Error opening video file: {video_path}")
        continue

    # Get video properties
    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create filename
    original_filename = os.path.basename(video_path)
    base_name, ext = os.path.splitext(original_filename)
    new_filename = f"{base_name}{OUTPUT_SUFFIX}{ext}"
    output_filename = os.path.join(OUTPUT_FOLDER, new_filename)

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (original_w, original_h))
    print(f"Saving to: {output_filename}")

    frame_count = 0
    fps_start_time = time.time()
    fps_frame_counter = 0
    fps_to_display = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\nEnd of video file.")
            break
        
        frame_count += 1
        
        # Resize for model processing
        frame_resized = cv2.resize(frame, (MODEL_IN_W, MODEL_IN_H),
                                   interpolation=cv2.INTER_AREA)

        # === STEP 1: YOLO Detection ===
        # Tracking class 2 (usually 'car' in COCO) - change as needed
        results = yolo.track(frame_resized, persist=True, classes=[2],
                             verbose=False, conf=0.3)

        final_frame = frame_resized

        if results[0].masks is not None and len(results[0].masks) > 0:
            # 1. Get Raw Mask
            all_masks = results[0].masks.data
            combined_mask = torch.any(all_masks, dim=0).int()
            mask_np = combined_mask.cpu().numpy().astype(np.uint8)
            
            # 2. Basic Dilate (Required for LaMa to see edges, but removed Gaussian smoothing)
            kernel = np.ones((9, 9), np.uint8) # Slightly smaller kernel since we aren't blurring
            mask_np = cv2.dilate(mask_np, kernel, iterations=2)
            
            # Ensure mask matches model size
            if mask_np.shape != (MODEL_IN_H, MODEL_IN_W):
                mask_np = cv2.resize(mask_np, (MODEL_IN_W, MODEL_IN_H),
                                     interpolation=cv2.INTER_NEAREST)
            
            # === STEP 2: LaMa Spatial Inpainting ===
            if USE_LAMA:
                try:
                    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    frame_rgb_np = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
                    
                    # Simple binary mask (0 or 255)
                    mask_binary = (mask_np * 255).astype(np.uint8)
                    mask_binary_np = np.ascontiguousarray(mask_binary, dtype=np.uint8)
                    
                    # Direct Inference
                    lama_output = lama(frame_rgb_np, mask_binary_np)
                    
                    # Convert back to BGR
                    final_frame = cv2.cvtColor(np.ascontiguousarray(lama_output, dtype=np.uint8),
                                               cv2.COLOR_RGB2BGR)
                except Exception as e:
                    print(f"⚠ LaMa error: {e}")
                    final_frame = frame_resized
            else:
                # Black out the detected area if LaMa is off
                mask_3ch = (mask_np > 0)[..., None]
                final_frame = frame_resized.copy()
                final_frame[mask_3ch.squeeze()] = 0

            # === Debug Visualization ===
            if DEBUG_MODE:
                h_d, w_d = MODEL_IN_H // 2, MODEL_IN_W // 2
                
                # Resize for grid
                d_orig = cv2.resize(frame_resized, (w_d, h_d))
                d_mask = cv2.resize(mask_np * 255, (w_d, h_d))
                d_final = cv2.resize(final_frame, (w_d, h_d))
                
                # Make mask 3 channel for display
                d_mask_bgr = cv2.cvtColor(d_mask, cv2.COLOR_GRAY2BGR)
                
                # Stack
                debug_view = np.hstack([d_orig, d_mask_bgr, d_final])
                
                cv2.putText(debug_view, "Original | Mask | LaMa Output", (5, 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                cv2.imshow("Debug View", debug_view)

        # === Final Display & Save ===
        # Resize final processed frame back to original video dimensions
        final_output_frame = cv2.resize(final_frame, (original_w, original_h))
        
        # Calculate FPS
        fps_frame_counter += 1
        elapsed_time = time.time() - fps_start_time
        if elapsed_time > 1.0:
            fps_to_display = fps_frame_counter / elapsed_time
            fps_frame_counter = 0
            fps_start_time = time.time()

        # UI Text
        cv2.putText(final_output_frame, f"FPS: {fps_to_display:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        status_text = f"LaMa: {'ON' if USE_LAMA else 'OFF'}"
        cv2.putText(final_output_frame, status_text, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Write and Show
        out.write(final_output_frame)
        cv2.imshow("Simple Inpainting", final_output_frame)
        
        # Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Skipping to next video...")
            break
        elif key == ord('d'):
            DEBUG_MODE = not DEBUG_MODE
            if not DEBUG_MODE: cv2.destroyWindow("Debug View")
        elif key == ord('1'):
            USE_LAMA = not USE_LAMA

    # Cleanup per video
    print(f"✓ Finished: {output_filename}")
    cap.release()
    out.release()
    if DEBUG_MODE: cv2.destroyWindow("Debug View")

cv2.destroyAllWindows()
print(f"\n✓ All videos processed.")