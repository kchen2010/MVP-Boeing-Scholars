"""
Spatial Inpainting Pipeline
1. YOLO: Object detection
2. EDGE-TAM: Segmentation from bounding boxes
3. LaMa: Spatial inpainting to fill holes using surrounding context
"""

import cv2
import torch
import numpy as np
import time
import sys
import os
from ultralytics import YOLO

# Add EdgeTAM to path if it exists
if os.path.exists('EdgeTAM'):
    sys.path.insert(0, 'EdgeTAM')

# Check for LaMa
try:
    from simple_lama_inpainting import SimpleLama
    LAMA_AVAILABLE = True
except ImportError:
    print("⚠ Installing LaMa: pip install simple-lama-inpainting")
    LAMA_AVAILABLE = False
    exit(1)

# Check for EDGE-TAM
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    EDGETAM_AVAILABLE = True
except ImportError:
    EDGETAM_AVAILABLE = False
    print("⚠ EDGE-TAM not available. Install EdgeTAM or ensure EdgeTAM folder is in the path.")
    print("   For now, using bounding box fallback for segmentation.")

# --- Constants ---
YOLO_MODEL = 'yolo11s.pt'
#VIDEO_SOURCE = 0  # Use 0 for webcam, or provide path to .mp4 file (e.g., 'content/TestVideo1.mp4')
VIDEO_SOURCE = 'content/TestVideo2.mp4'  # Uncomment to use video file

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_IN_H, MODEL_IN_W = 240, 432
USE_LAMA = True
DEBUG_MODE = False

print(f"=== Spatial Inpainting System ===")
print(f"Device: {DEVICE}")
print(f"Pipeline: YOLO (detection) -> EDGE-TAM (segmentation) -> LaMa (spatial)")

# 1. Load YOLO (detection-only)
print("Loading YOLO (detection-only)...")
yolo = YOLO(YOLO_MODEL)
yolo.to(DEVICE)

# 2. Load EDGE-TAM for segmentation
print("Loading EDGE-TAM...")
edgetam_predictor = None

EDGETAM_ROOT = 'EdgeTAM'
checkpoint_path = os.path.join(EDGETAM_ROOT, "checkpoints", "edgetam.pt")
model_cfg = "edgetam.yaml"

if not os.path.exists(checkpoint_path):
    print(f"\n[ERROR] Model weights not found at: {checkpoint_path}")
    print("Action Required: Download 'edgetam.pt' from the GitHub link and place it in the checkpoints folder.")
    sys.exit(1)

print(f"Loading EdgeTAM from: {checkpoint_path}")

try:
    model = build_sam2(model_cfg, checkpoint_path, device=DEVICE)
    edgetam_predictor = SAM2ImagePredictor(model)
    print("✓ EdgeTAM loaded successfully!")
    EDGETAM_AVAILABLE = True

except FileNotFoundError:
    # Fallback: Try providing the absolute path to the config if the simple name fails
    try:
        print("   'edgetam.yaml' not found in path, trying absolute path...")
        abs_config_path = os.path.join(EDGETAM_ROOT, "sam2", "configs", "edgetam.yaml")
        model = build_sam2(abs_config_path, checkpoint_path, device=DEVICE)
        edgetam_predictor = SAM2ImagePredictor(model)
        print("✓ EdgeTAM loaded successfully (using absolute config path)!")
        EDGETAM_AVAILABLE = True
    except Exception as e:
        print(f"\n[CRITICAL FAILURE] Could not load config: {e}")
        EDGETAM_AVAILABLE = False

# 3. Load LaMa
print("Loading LaMa...")
lama = SimpleLama()

print("✓ All models loaded\n")

# 4. Initialize video capture
is_video_file = isinstance(VIDEO_SOURCE, str) and os.path.exists(VIDEO_SOURCE)
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print(f"Error: Could not open video source: {VIDEO_SOURCE}")
    sys.exit(1)

if is_video_file:
    print(f"Processing video file: {VIDEO_SOURCE}")
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video properties: {total_frames} frames @ {original_fps:.2f} FPS")
else:
    print("Using webcam input")

print("\nPress 'q' to quit, 'd' to toggle debug")
print("Press '1' to toggle LaMa")
print("Press 's' to save current frame\n")

frame_count = 0

# --- 2. INITIALIZE FPS VARIABLES ---
fps_start_time = time.time()
fps_frame_counter = 0
fps_to_display = 0.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        if is_video_file:
            print("\nEnd of video file.")
        break
    
    frame_count += 1

    original_h, original_w = frame.shape[:2]
    frame_resized = cv2.resize(frame, (MODEL_IN_W, MODEL_IN_H), 
                               interpolation=cv2.INTER_AREA)

    # STEP 1: YOLO Detection (no segmentation)
    results = yolo.track(frame_resized, persist=True, classes=[2], 
                         verbose=False, conf=0.3)

    boxes = results[0].boxes
    mask_np = None
    frame_rgb = None
    
    if boxes is not None and len(boxes) > 0:
        # STEP 2: EDGE-TAM Segmentation
        if EDGETAM_AVAILABLE and edgetam_predictor is not None:
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            try:
                edgetam_predictor.set_image(frame_rgb)
                
                combined_mask = np.zeros((MODEL_IN_H, MODEL_IN_W), dtype=np.uint8)
                
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    box_prompt = np.array([x1, y1, x2, y2])
                    
                    masks, scores, _ = edgetam_predictor.predict(
                        box=box_prompt,
                        multimask_output=False
                    )
                    
                    mask = masks[0] > 0
                    combined_mask = np.maximum(combined_mask, mask.astype(np.uint8))
                
            except Exception as e:
                print(f"⚠ EDGE-TAM segmentation error: {e}")
                combined_mask = np.zeros((MODEL_IN_H, MODEL_IN_W), dtype=np.uint8)
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    x1 = max(0, min(x1, MODEL_IN_W - 1))
                    y1 = max(0, min(y1, MODEL_IN_H - 1))
                    x2 = max(0, min(x2, MODEL_IN_W - 1))
                    y2 = max(0, min(y2, MODEL_IN_H - 1))
                    combined_mask[y1:y2, x1:x2] = 1
        else:
            combined_mask = np.zeros((MODEL_IN_H, MODEL_IN_W), dtype=np.uint8)
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                x1 = max(0, min(x1, MODEL_IN_W - 1))
                y1 = max(0, min(y1, MODEL_IN_H - 1))
                x2 = max(0, min(x2, MODEL_IN_W - 1))
                y2 = max(0, min(y2, MODEL_IN_H - 1))
                combined_mask[y1:y2, x1:x2] = 1
        
        mask_np = combined_mask
        
        kernel = np.ones((11, 11), np.uint8)
        mask_np = cv2.dilate(mask_np, kernel, iterations=2)
        
        mask_np_255 = (mask_np * 255).astype(np.uint8)
        mask_smooth = cv2.GaussianBlur(mask_np_255, (15, 15), 0)
        mask_smooth = np.ascontiguousarray(mask_smooth, dtype=np.uint8)
    
    if mask_np is not None and np.any(mask_np > 0):
        
        if frame_rgb is None:
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # STEP 3: LaMa Spatial Inpainting 
        if USE_LAMA:
            try:
                frame_rgb_np = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
                mask_binary = (mask_smooth > 128).astype(np.uint8) * 255
                mask_binary_np = np.ascontiguousarray(mask_binary, dtype=np.uint8)
                
                lama_output = lama(frame_rgb_np, mask_binary_np)
                
                lama_output_np = np.ascontiguousarray(lama_output, dtype=np.uint8)
                stage1_result = cv2.cvtColor(lama_output_np, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"⚠ LaMa error: {e}")
                stage1_result = frame_resized
        else:
            mask_3ch = (mask_np > 0)[..., None]
            masked_frame = frame_rgb.copy()
            masked_frame[mask_3ch.squeeze()] = 0
            stage1_result = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)
        
        final_frame = stage1_result

        # debug mode
        # display the original, mask, LaMa output, and difference
        if DEBUG_MODE:
            h_d, w_d = MODEL_IN_H // 2, MODEL_IN_W // 2
            
            debug_original = cv2.resize(frame_resized, (w_d, h_d))
            
            mask_vis = cv2.applyColorMap(mask_smooth, cv2.COLORMAP_JET) 
            mask_overlay = cv2.addWeighted(debug_original, 0.6, 
                                           cv2.resize(mask_vis, (w_d, h_d)), 0.4, 0)
            
            debug_lama = cv2.resize(stage1_result, (w_d, h_d))
            debug_final = cv2.resize(final_frame, (w_d, h_d))
            
            diff_lama = cv2.absdiff(debug_lama, debug_original)
            diff_lama = cv2.applyColorMap(cv2.cvtColor(diff_lama, cv2.COLOR_BGR2GRAY), 
                                         cv2.COLORMAP_HOT)
            
            top_row = np.hstack([debug_original, mask_overlay, debug_lama])
            bottom_row = np.hstack([diff_lama, debug_final])
            debug_view = np.vstack([top_row, bottom_row])
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            
            cv2.putText(debug_view, "1.Original", (5, 15), font, font_scale, (0, 255, 0), thickness)
            cv2.putText(debug_view, "2.Mask", (w_d+5, 15), font, font_scale, (0, 255, 0), thickness)
            cv2.putText(debug_view, f"3.LaMa {'ON' if USE_LAMA else 'OFF'}", (2*w_d+5, 15), 
                        font, font_scale, (0, 255, 0) if USE_LAMA else (100, 100, 100), thickness)
            cv2.putText(debug_view, "4.Difference", (5, h_d+15), font, font_scale, (0, 255, 0), thickness)
            cv2.putText(debug_view, "5.Final", (w_d+5, h_d+15), font, font_scale, (0, 255, 0), thickness)
            
            cv2.imshow("Debug View", debug_view)
        
    else:
        final_frame = frame_resized
    
    final_display = cv2.resize(final_frame, (original_w, original_h))
    
    fps_frame_counter += 1
    current_time = time.time()
    elapsed_time = current_time - fps_start_time

    if elapsed_time > 1.0:
        fps_to_display = fps_frame_counter / elapsed_time
        fps_frame_counter = 0
        fps_start_time = current_time

    cv2.putText(final_display, f"FPS: {fps_to_display:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    status_text = f"LaMa: {'ON' if USE_LAMA else 'OFF'}"
    cv2.putText(final_display, status_text, (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    detection_status = "Inpainting" if (boxes is not None and len(boxes) > 0) else "No Detection"
    cv2.putText(final_display, detection_status, (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Hybrid Inpainting", final_display)
    
    #  keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        DEBUG_MODE = not DEBUG_MODE
        if not DEBUG_MODE:
            cv2.destroyWindow("Debug View")
        print(f"Debug: {DEBUG_MODE}")
    elif key == ord('1'):
        USE_LAMA = not USE_LAMA
        print(f"LaMa: {USE_LAMA}")
    elif key == ord('s'):
        filename = f'hybrid_inpaint_frame_{frame_count}.jpg'
        cv2.imwrite(filename, final_display)
        print(f"✓ Saved: {filename}")

cap.release()
cv2.destroyAllWindows()
print("\n✓ Video stream ended")