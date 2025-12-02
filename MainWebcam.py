"""
Hybrid Spatial-Temporal Inpainting Pipeline
1. LaMa: Spatial inpainting to fill holes using surrounding context
2. FuseFormer: Temporal refinement for consistency across frames
"""

import cv2
import torch
import numpy as np
import time  # <--- 1. IMPORT TIME
from ultralytics import YOLO
from collections import deque

# Check for LaMa
try:
    from simple_lama_inpainting import SimpleLama
    LAMA_AVAILABLE = True
except ImportError:
    print("⚠ Installing LaMa: pip install simple-lama-inpainting")
    LAMA_AVAILABLE = False
    exit(1)

# Import FuseFormer
from FuseFormer_OM import InpaintGenerator

# --- Constants ---
YOLO_MODEL = 'yolov8s-seg.pt' 
FUSEFORMER_WEIGHTS = 'checkpoints/fuseformer.pth'
VIDEO_SOURCE = 0 #webcam
# VIDEO_SOURCE = ['content/TestVideo1.mp4']

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Processing parameters
MODEL_IN_H, MODEL_IN_W = 240, 432
TEMPORAL_WINDOW = 5  # Frames for FuseFormer temporal consistency
USE_LAMA = True      # Toggle spatial inpainting
USE_FUSEFORMER = False # Toggle temporal refinement

DEBUG_MODE = False   # Set to False for clean output

print(f"=== Hybrid Inpainting System ===")
print(f"Device: {DEVICE}")
print(f"Pipeline: YOLO -> LaMa (spatial) -> FuseFormer (temporal)")

# 1. Load YOLO
print("Loading YOLO...")
yolo = YOLO(YOLO_MODEL)
yolo.to(DEVICE)

# 2. Load LaMa
print("Loading LaMa...")
lama = SimpleLama()

# 3. Load FuseFormer
print("Loading FuseFormer...")
fuseformer = InpaintGenerator(init_weights=False)
checkpoint = torch.load(FUSEFORMER_WEIGHTS, map_location=DEVICE)

if isinstance(checkpoint, dict):
    state_dict = checkpoint.get('netG', checkpoint.get('state_dict', checkpoint))
else:
    state_dict = checkpoint

fuseformer.load_state_dict(state_dict, strict=False)
fuseformer.to(DEVICE)
fuseformer.eval()

print("✓ All models loaded\n")

# 4. Initialize video and buffers
cap = cv2.VideoCapture(VIDEO_SOURCE) 

# Frame buffer for temporal processing
frame_buffer = deque(maxlen=TEMPORAL_WINDOW)
former_attn = torch.Tensor().to(DEVICE)

print("Press 'q' to quit, 'd' to toggle debug")
print("Press '1' to toggle LaMa, '2' to toggle FuseFormer")
print("Press 's' to save current frame\n")

frame_count = 0

# --- 2. INITIALIZE FPS VARIABLES ---
fps_start_time = time.time()
fps_frame_counter = 0
fps_to_display = 0.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1

    original_h, original_w = frame.shape[:2]
    frame_resized = cv2.resize(frame, (MODEL_IN_W, MODEL_IN_H), 
                               interpolation=cv2.INTER_AREA)

    # === STEP 1: YOLO Detection ===
    results = yolo.track(frame_resized, persist=True, classes=[2], 
                         verbose=False, conf=0.3)

    if results[0].masks is not None and len(results[0].masks) > 0:
        # Create mask
        all_masks = results[0].masks.data
        combined_mask = torch.any(all_masks, dim=0).int()
        mask_np = combined_mask.cpu().numpy().astype(np.uint8) # This is 0s and 1s
        
        # Dilate for complete coverage
        kernel = np.ones((11, 11), np.uint8)
        mask_np = cv2.dilate(mask_np, kernel, iterations=2) # Still 0s and 1s
        
        if mask_np.shape != (MODEL_IN_H, MODEL_IN_W):
            mask_np = cv2.resize(mask_np, (MODEL_IN_W, MODEL_IN_H), 
                                 interpolation=cv2.INTER_NEAREST)
        
        # Smooth edges
        mask_float = (mask_np * 255).astype(float) 
        mask_blur = cv2.GaussianBlur(mask_float, (15, 15), 0) 
        mask_smooth = mask_blur.astype(np.uint8)
        
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # === STEP 2: LaMa Spatial Inpainting ===
        if USE_LAMA:
            try:
                frame_rgb_np = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
                mask_binary = (mask_smooth > 128).astype(np.uint8) * 255
                mask_binary_np = np.ascontiguousarray(mask_binary, dtype=np.uint8)
                
                lama_output = lama(frame_rgb_np, mask_binary_np)
                
                stage1_result = cv2.cvtColor(np.ascontiguousarray(lama_output, dtype=np.uint8), 
                                             cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"⚠ LaMa error: {e}")
                stage1_result = frame_resized
        else:
            # Skip LaMa
            mask_3ch = (mask_np > 0)[..., None]
            masked_frame = frame_rgb.copy()
            masked_frame[mask_3ch.squeeze()] = 0
            stage1_result = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)
        
        # === STEP 3: FuseFormer Temporal Refinement ===
        stage2_result = stage1_result.copy()
        
        if USE_FUSEFORMER:
            stage1_rgb = cv2.cvtColor(stage1_result, cv2.COLOR_BGR2RGB)
            frame_buffer.append(stage1_rgb)
            
            if len(frame_buffer) >= TEMPORAL_WINDOW:
                frames_stack = np.stack(list(frame_buffer), axis=0)
                frames_tensor = torch.from_numpy(frames_stack).permute(0, 3, 1, 2).float()
                frames_tensor = (frames_tensor / 127.5) - 1.0
                frames_tensor = frames_tensor.unsqueeze(0).to(DEVICE)
                
                try:
                    with torch.no_grad():
                        output, former_attn = fuseformer(frames_tensor, former_attn)
                    
                    b_t, c, h, w = output.shape
                    output = output.view(1, TEMPORAL_WINDOW, c, h, w)
                    last_frame = output[0, -1]
                    
                    last_frame = torch.clamp(last_frame, -1, 1)
                    last_frame = (last_frame + 1.0) / 2.0
                    
                    refined_np = last_frame.permute(1, 2, 0).cpu().numpy()
                    refined_np = (refined_np * 255).astype(np.uint8)
                    stage2_result = cv2.cvtColor(refined_np, cv2.COLOR_RGB2BGR)
                    
                    mask_blend = cv2.GaussianBlur(mask_blur / 255.0, (21, 21), 0)[..., None]
                    
                    stage2_result = (stage2_result * mask_blend + 
                                     stage1_result * (1 - mask_blend)).astype(np.uint8)
                    
                except Exception as e:
                    print(f"⚠ FuseFormer error: {e}")
                    stage2_result = stage1_result
        
        final_frame = stage2_result
        
        # === Debug Visualization ===
        if DEBUG_MODE:
            h_d, w_d = MODEL_IN_H // 2, MODEL_IN_W // 2
            
            debug_original = cv2.resize(frame_resized, (w_d, h_d))
            
            mask_vis = cv2.applyColorMap(mask_smooth, cv2.COLORMAP_JET) 
            mask_overlay = cv2.addWeighted(debug_original, 0.6, 
                                           cv2.resize(mask_vis, (w_d, h_d)), 0.4, 0)
            
            debug_lama = cv2.resize(stage1_result, (w_d, h_d))
            debug_fuseformer = cv2.resize(stage2_result, (w_d, h_d))
            debug_final = cv2.resize(final_frame, (w_d, h_d))
            
            diff_lama = cv2.absdiff(debug_lama, debug_original)
            diff_lama = cv2.applyColorMap(cv2.cvtColor(diff_lama, cv2.COLOR_BGR2GRAY), 
                                         cv2.COLORMAP_HOT)
            
            top_row = np.hstack([debug_original, mask_overlay, debug_lama])
            bottom_row = np.hstack([diff_lama, debug_fuseformer, debug_final])
            debug_view = np.vstack([top_row, bottom_row])
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            
            cv2.putText(debug_view, "1.Original", (5, 15), font, font_scale, (0, 255, 0), thickness)
            cv2.putText(debug_view, "2.Mask", (w_d+5, 15), font, font_scale, (0, 255, 0), thickness)
            cv2.putText(debug_view, f"3.LaMa {'ON' if USE_LAMA else 'OFF'}", (2*w_d+5, 15), 
                        font, font_scale, (0, 255, 0) if USE_LAMA else (100, 100, 100), thickness)
            cv2.putText(debug_view, "4.Difference", (5, h_d+15), font, font_scale, (0, 255, 0), thickness)
            cv2.putText(debug_view, f"5.FuseFormer {'ON' if USE_FUSEFORMER else 'OFF'}", 
                        (w_d+5, h_d+15), font, font_scale, 
                        (0, 255, 0) if USE_FUSEFORMER else (100, 100, 100), thickness)
            cv2.putText(debug_view, "6.Final", (2*w_d+5, h_d+15), font, font_scale, (0, 255, 0), thickness)
            buffer_status = f"Buffer: {len(frame_buffer)}/{TEMPORAL_WINDOW}"
            cv2.putText(debug_view, buffer_status, (5, h_d*2-10), 
                        font, 0.35, (255, 255, 0), 1)
            
            cv2.imshow("Debug View", debug_view)
        
    else:
        # No detection
        final_frame = frame_resized
        if USE_FUSEFORMER:
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_buffer.append(frame_rgb)
    
    # === Main Display ===
    final_display = cv2.resize(final_frame, (original_w, original_h))
    
    
    # --- 3. CALCULATE AND DISPLAY FPS ---
    
    # Increment frame counter
    fps_frame_counter += 1
    current_time = time.time()
    elapsed_time = current_time - fps_start_time

    # Update FPS calculation every second
    if elapsed_time > 1.0:
        fps_to_display = fps_frame_counter / elapsed_time
        fps_frame_counter = 0  # Reset frame counter
        fps_start_time = current_time  # Reset start time

    # Display FPS
    cv2.putText(final_display, f"FPS: {fps_to_display:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Display pipeline status
    status_text = f"LaMa: {'ON' if USE_LAMA else 'OFF'} | FuseFormer: {'ON' if USE_FUSEFORMER else 'OFF'}"
    cv2.putText(final_display, status_text, (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Display detection status
    detection_status = "Inpainting" if (results[0].masks is not None and len(results[0].masks) > 0) else "No Detection"
    cv2.putText(final_display, detection_status, (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    
    cv2.imshow("Hybrid Inpainting", final_display)
    
    # === Keyboard Controls ===
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
    elif key == ord('2'):
        USE_FUSEFORMER = not USE_FUSEFORMER
        print(f"FuseFormer: {USE_FUSEFORMER}")
    elif key == ord('s'):
        filename = f'hybrid_inpaint_frame_{frame_count}.jpg'
        cv2.imwrite(filename, final_display)
        print(f"✓ Saved: {filename}")

cap.release()
cv2.destroyAllWindows()
print("\n✓ Video stream ended")