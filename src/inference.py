import cv2
from ultralytics import YOLO
import time
import os
import traceback  # For detailed error printing
import numpy as np
import threading  # For dedicated capture thread
from queue import Queue  # For thread-safe frame passing
import sys
import json
from flask import Flask



# --- Global Configuration ---
MODEL_PATH = "F:/JetBrains/PycharmProjects/AWC_CV_QC/src/runs/classify/11s/weights/best.pt" # Path to your trained .pt model
CAMERA_ID = 0  # Change if you have multiple cameras
REQUESTED_WIDTH = 1920  # camera width
REQUESTED_HEIGHT = 1080  # camera height
ZOOM_FACTOR = 3  # 1 is no zoom
AUTO_EXPOSURE = False
MANUAL_EXPOSURE_STOP = -5
MIN_RECT_AREA = 10000  # Minimum pixel area
RECT_DETECT_RETRIES = 100  # Number of times to retry
APPROX_POLY_EPSILON_FACTOR = 0.02  # amount of distortion due to camera
MIN_ASPECT_RATIO = 0.2
MAX_ASPECT_RATIO = 5.0
MORPH_KERNEL_SIZE = (5, 5)  # Morphological kernel size

# --- Flask ---
app = Flask(__name__)
HOST = "0.0.0.0"
PORT = 5001
# ---



def set_model_path(path):
    global MODEL_PATH
    MODEL_PATH = path



class NameTagQualityControl:
    def __init__(self, model_path, camera_id=0, zoom_factor=1):
        """
        Initializes the NameTagQualityControl class. Sets up camera,
        starts a dedicated capture thread, and configures settings.
        """
        self.model = YOLO(model_path)

        # --- Initialize Camera (Try different backends if default fails) ---
        print(f"Attempting to open camera ID {camera_id}...")
        # Option 1: Default backend
        self.camera = cv2.VideoCapture(camera_id)

        # Option 2: Try DirectShow backend (Windows specific) - Uncomment if on Windows and default has issues
        # print("Trying DirectShow backend (cv2.CAP_DSHOW)...")
        # self.camera = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)

        # Option 3: Try Media Foundation backend (Windows specific) - Uncomment if on Windows
        # print("Trying Media Foundation backend (cv2.CAP_MSMF)...")
        # self.camera = cv2.VideoCapture(camera_id, cv2.CAP_MSMF)

        # Option 4: Try AVFoundation backend (macOS specific) - Uncomment if on macOS
        # print("Trying AVFoundation backend (cv2.CAP_AVFOUNDATION)...")
        # self.camera = cv2.VideoCapture(camera_id, cv2.CAP_AVFOUNDATION)


        if not self.camera.isOpened():
             # If the chosen backend failed, maybe try the default as a last resort?
             # Or just raise the exception.
             raise Exception(f"Error: Could not open camera with ID {camera_id} using the selected backend.")
        print(f"Camera backend API name: {self.camera.getBackendName()}")
        # ---

        # --- Configure Camera Settings ---
        print("Configuring camera...")
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_HEIGHT)
        time.sleep(0.2) # Allow settings to apply

        if AUTO_EXPOSURE:
            print("Setting Auto Exposure (0.25)...")
            self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        else:
            print(f"Setting Manual Exposure (Value: {MANUAL_EXPOSURE_STOP})...")
            # Try setting manual mode (0 is standard, 0.75 as fallback)
            if not self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0):
                 print("Setting manual mode with 0 failed, trying 0.75...")
                 self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            time.sleep(0.1)
            self.camera.set(cv2.CAP_PROP_EXPOSURE, MANUAL_EXPOSURE_STOP)

        # Verify settings
        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Actual camera resolution: {actual_width}x{actual_height}")
        print(f"Actual Auto Exposure setting: {self.camera.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")
        print(f"Actual Exposure value: {self.camera.get(cv2.CAP_PROP_EXPOSURE)}")
        # --- End Camera Settings ---

        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
        self.zoom_factor = zoom_factor

        # --- Threaded Capture Setup ---
        self.capture_queue = Queue(maxsize=1) # Store only the latest frame
        self.capture_active = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        print("Capture thread started.")
        # ---

    def _capture_loop(self):
        """Continuously captures frames and puts the latest one in the queue."""
        print("Capture loop running...")
        frame_counter = 0
        error_counter = 0
        while self.capture_active:
            ret, frame = self.camera.read()
            if ret and frame is not None:
                # Optional: Basic frame validation (e.g., check dimensions, mean intensity)
                # if frame.shape[0] != REQUESTED_HEIGHT or frame.shape[1] != REQUESTED_WIDTH:
                #    print(f"Warning: Captured frame has unexpected dimensions: {frame.shape}")
                #    error_counter += 1
                #    continue # Skip potentially corrupt frame

                try:
                    if self.capture_queue.full():
                        self.capture_queue.get_nowait() # Discard older frame
                    self.capture_queue.put(frame)
                    frame_counter += 1
                    if frame_counter % 300 == 0: # Print status occasionally
                         print(f"Capture thread status: {frame_counter} frames captured, {error_counter} errors.")

                except Exception as e:
                     print(f"Error putting frame in queue: {e}")
                     error_counter += 1
            else:
                # print("Warning: camera.read() failed in capture thread.") # Reduce noise
                error_counter += 1
                if error_counter > 100 and error_counter % 50 == 0: # Log if many consecutive errors
                     print(f"Warning: {error_counter} consecutive camera read failures!")
                time.sleep(0.05) # Increase sleep slightly on read failure
        print("Capture loop stopped.")

    def stop_capture(self):
        """Signals the capture thread to stop."""
        print("Stopping capture thread...")
        self.capture_active = False

    def get_latest_frame(self):
        """Gets the latest frame from the capture queue."""
        try:
            return self.capture_queue.get(timeout=0.5) # Increased timeout slightly
        except Exception:
            # print("Warning: Capture queue empty or timeout.") # Reduce noise
            return None


    def _apply_zoom(self, frame):
        """Applies center zoom to the frame."""
        if frame is None or frame.size == 0: return None
        if self.zoom_factor <= 1.0: return frame
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        new_w = int(w / self.zoom_factor)
        new_h = int(h / self.zoom_factor)
        x1 = max(0, center_x - new_w // 2)
        y1 = max(0, center_y - new_h // 2)
        x2 = min(w, x1 + new_w)
        y2 = min(h, y1 + new_h)
        if x1 >= x2 or y1 >= y2: return None
        zoomed_frame = frame[y1:y2, x1:x2]
        if zoomed_frame.size == 0: return None
        return zoomed_frame

    def _find_largest_rectangle_contour(self, contours, frame_shape):
        """Finds the largest contour that approximates to 4 vertices."""
        largest_rectangle_contour = None
        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_RECT_AREA: continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0: continue # Avoid division by zero
            approx = cv2.approxPolyDP(contour, APPROX_POLY_EPSILON_FACTOR * perimeter, True)
            if len(approx) == 4:
                # Check convexity and aspect ratio
                if not cv2.isContourConvex(approx): continue
                x, y, w, h = cv2.boundingRect(approx)
                if w <= 0 or h <= 0: continue
                aspect_ratio = float(w) / h if h > 0 else 0
                inv_aspect_ratio = float(h) / w if w > 0 else 0
                is_valid_aspect_ratio = (MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO) or \
                                        (MIN_ASPECT_RATIO < inv_aspect_ratio < MAX_ASPECT_RATIO)
                if not is_valid_aspect_ratio: continue
                # Check if contour is reasonably within frame bounds (helps filter noise near edges)
                # margin = 0.05 # 5% margin
                # if x < margin * frame_shape[1] or y < margin * frame_shape[0] or /
                #    (x + w) > (1 - margin) * frame_shape[1] or (y + h) > (1 - margin) * frame_shape[0]:
                #      continue

                if area > max_area:
                    max_area = area
                    largest_rectangle_contour = contour
        return largest_rectangle_contour

    def _detect_rectangle_info(self, frame):
        """Detects the largest valid rectangle contour."""
        if frame is None or frame.size == 0: return None
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Experiment with blur kernel size
            blurred = cv2.GaussianBlur(gray, (7, 7), 0) # Was (5,5)
            # Experiment with Canny thresholds
            edges = cv2.Canny(blurred, 30, 100) # Lowered thresholds slightly, was 50, 150
            # Experiment with morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_SIZE)
            # Dilate then Erode (Close) helps close gaps in edges
            closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            # Optional: Erode then Dilate (Open) helps remove small noise
            # opened_edges = cv2.morphologyEx(closed_edges, cv2.MORPH_OPEN, kernel)
            edges_processed = closed_edges # Use the closed edges
        except cv2.error as e:
            print(f"OpenCV error during preprocessing: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error during preprocessing: {e}")
            return None

        contours, _ = cv2.findContours(edges_processed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None

        largest_rect_contour = self._find_largest_rectangle_contour(contours, frame.shape)
        if largest_rect_contour is None: return None

        bbox = cv2.boundingRect(largest_rect_contour)
        min_rect = cv2.minAreaRect(largest_rect_contour)
        return {'bbox': bbox, 'min_rect': min_rect}

    def _rotate_and_crop(self, frame, min_rect):
         """Rotates and crops the rectangle."""
         if frame is None or min_rect is None: return None
         center, size, angle = min_rect
         width, height = size
         # Adjust angle interpretation if necessary (common minAreaRect quirk)
         # If angle is close to -90, swap width/height and adjust angle
         if angle < -45:
              angle += 90.0
              width, height = height, width # Swap width and height

         M = cv2.getRotationMatrix2D(center, angle, 1.0)
         try:
              # Calculate bounding box of rotated rectangle to determine output size
              box = cv2.boxPoints(min_rect)
              pts = np.intp(box) # Use np.intp for potentially large coordinates
              x_coords = pts[:, 0]
              y_coords = pts[:, 1]
              rect_w = int(np.linalg.norm(pts[0] - pts[1]))
              rect_h = int(np.linalg.norm(pts[1] - pts[2]))
              # Use a slightly larger size for warpAffine to avoid clipping corners
              warp_w = int(max(rect_w, rect_h) * 1.2)
              warp_h = warp_w

              # Adjust rotation matrix to center the rectangle in the output
              M[0, 2] += (warp_w / 2) - center[0]
              M[1, 2] += (warp_h / 2) - center[1]

              # Perform rotation into the calculated size
              warped = cv2.warpAffine(frame, M, (warp_w, warp_h), flags=cv2.INTER_CUBIC)
              if warped is None: raise Exception("warpAffine returned None")

         except Exception as e:
              print(f"Error during warpAffine: {e}")
              return None

         # Crop the final rectangle from the center of the warped image
         crop_width = int(round(width))
         crop_height = int(round(height))
         if crop_width <= 0 or crop_height <= 0: return None

         try:
              # Calculate the center in the *new* warped image coordinates
              new_center = (warp_w / 2, warp_h / 2)
              cropped = cv2.getRectSubPix(warped, (crop_width, crop_height), new_center)
              if cropped is None: raise Exception("getRectSubPix returned None")
         except Exception as e:
              print(f"Error during getRectSubPix: {e}")
              return None

         # Final check on aspect ratio (sometimes needed if initial angle adjustment wasn't perfect)
         if crop_height > crop_width * 1.2:
              print("Rotating final cropped image by 90 degrees.")
              cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)

         if cropped.size == 0: return None
         return cropped

    def inspect_tag(self):
        """
        Gets latest frame, zooms, retries rectangle detection, rotates/crops,
        runs inference. Falls back to zoomed image if detection fails.
        """
        # 1. Get Latest Raw Image from queue
        raw_image = self.get_latest_frame()
        if raw_image is None:
             print("Capture queue was empty, attempting direct read...")
             ret_direct, raw_image = self.camera.read()
             if not ret_direct or raw_image is None:
                 return {"error": "Image capture failed (queue empty & direct read failed)", "image_data": None}

        raw_image = raw_image.copy()

        # 2. Apply Zoom
        zoomed_image = self._apply_zoom(raw_image)
        if zoomed_image is None:
             return {"error": "Zoom processing failed", "image_data": raw_image}

        # 3. Detect Rectangle Coordinates (with Retries)
        rect_info = None
        for attempt in range(RECT_DETECT_RETRIES):
             rect_info = self._detect_rectangle_info(zoomed_image.copy())
             if rect_info: break

        # --- Process based on detection result ---
        final_image = None
        is_detection_fallback = False
        error_msg = None

        if rect_info:
            # 4. Rotate and Crop if rectangle detected
            final_image = self._rotate_and_crop(zoomed_image, rect_info['min_rect'])
            if final_image is None:
                 print("Warning: Rotation/Cropping failed. Falling back to zoomed image.")
                 final_image = zoomed_image
                 is_detection_fallback = True
                 error_msg = "Rotation/Cropping failed"
        else:
            # 5. Fallback to zoomed image if detection failed
            final_image = zoomed_image
            is_detection_fallback = True
            error_msg = f"Rectangle detection failed after {RECT_DETECT_RETRIES} retries"


        # --- Inference ---
        classification_result = {}
        if final_image is not None:
            timestamp_str = f"{int(time.time())}"
            temp_path = os.path.join(self.results_dir, f"temp_{timestamp_str}.jpg")
            try:
                save_success = cv2.imwrite(temp_path, final_image)
                if not save_success: temp_path = None
            except Exception: temp_path = None
            try:
                source_for_model = final_image if temp_path is None else temp_path
                results = self.model(source=source_for_model)
                if not results or not hasattr(results[0], 'probs') or results[0].probs is None:
                    classification_result = {"class": "Inference Error", "confidence": 0.0}
                    if error_msg is None: error_msg = "Inference failed or returned no probabilities"
                else:
                    predicted_class_index = results[0].probs.top1
                    confidence = float(results[0].probs.top1conf)
                    if results[0].names and predicted_class_index < len(results[0].names):
                        class_name = results[0].names[predicted_class_index]
                    else:
                        class_name = "Unknown"
                    classification_result = {"class": class_name, "confidence": confidence}

            except Exception as model_err:
                print(f"Error during model inference: {model_err}")
                classification_result = {"class": "Inference Error", "confidence": 0.0}
                if error_msg is None: error_msg = f"Model inference failed: {model_err}"
        else:
            classification_result = {"class": "Processing Error", "confidence": 0.0}
            if error_msg is None: error_msg = "final_image was None before inference"
            temp_path = None

        # --- Construct Final Result ---
        result_data = {
            "class": classification_result.get("class", "Error"),
            "confidence": classification_result.get("confidence", 0.0),
            "timestamp": time.time(),
            "image_path": temp_path,
            "image_data": final_image
        }
        if is_detection_fallback:
            result_data["class"] = "Detection Failed"
            result_data["confidence"] = 0.0
        if error_msg:
            result_data["error"] = error_msg

        # --- Save JSON ---
        latest_result_path = os.path.join(self.results_dir, "latest_result.json")
        try:
            json_data = {k: v for k, v in result_data.items() if k != 'image_data'}
            with open(latest_result_path, "w") as f:
                json.dump(json_data, f, indent=4)
        except IOError as e1:
            print(f"Error writing results to {latest_result_path}: {e1}")

        return result_data


    def close(self):
        """Stops the capture thread and releases the camera resource."""
        self.stop_capture() # Signal thread to stop
        if self.capture_thread.is_alive():
             self.capture_thread.join(timeout=1.0) # Wait for thread to finish
             if self.capture_thread.is_alive():
                  print("Warning: Capture thread did not stop gracefully.")
        if self.camera.isOpened():
            self.camera.release()
        print("Camera released and capture thread stopped.")



trigger = False
@app.route("/trigger-inference", methods=["GET"])  # Accept GET
def handle_trigger():
    global trigger
    trigger = True
    print("Triggering inference...")



def start_flask_app():
    app.run(host=HOST, port=PORT, threaded=True)
# Create and start the Flask server thread
# Set daemon=True so the thread exits when the main program exits
threading.Thread(target=start_flask_app, daemon=True).start()



if __name__ == "__main__":

    # --- Check if model file exists ---
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        sys.exit()
    # ---

    qc = None
    quit_flag = False
    live_window_name = "Live View (3X Zoomed)"
    result_window_name = "Inspection Result (Cropped & Rotated)"
    result_window_created = False



    try:
        qc = NameTagQualityControl(MODEL_PATH, camera_id=CAMERA_ID, zoom_factor=ZOOM_FACTOR)

        print("/nStarting inspection loop.")
        print("Live feed shows zoomed view with detected rectangle. Press Enter to capture, crop, rotate and inspect.")
        print("Press 'q' in any window to quit.")
        if AUTO_EXPOSURE:
            print(f"Using Auto Exposure setting: 0.25 (Behavior depends on driver interpretation)")
        else:
            print(f"Using Manual Exposure setting: {MANUAL_EXPOSURE_STOP}")



        while not quit_flag: # Main loop controlled by flag

            # --- Live Video Feed ---
            ret_live, live_frame_raw = qc.camera.read()
            display_live_frame = None  # Frame to actually display

            if ret_live and live_frame_raw is not None:
                try:
                    # 1. Apply zoom for the live feed display
                    zoomed_live_frame = qc._apply_zoom(live_frame_raw)

                    if zoomed_live_frame is not None:
                        # Make a copy to draw on
                        display_live_frame = zoomed_live_frame.copy()

                        # 2. Detect rectangle info (bbox needed for drawing)
                        live_rect_info = qc._detect_rectangle_info(zoomed_live_frame)  # Uses updated epsilon

                        # 3. Draw bounding box if coordinates are found
                        if live_rect_info:
                            min_rect = live_rect_info['min_rect']
                            box = cv2.boxPoints(min_rect)  # Get the 4 corner points
                            box = np.intp(box)  # Convert points to integer type for drawing
                            cv2.drawContours(display_live_frame, [box], 0, (0, 255, 0), 2)

                        # Display the frame (zoomed, potentially with box)
                        cv2.imshow(live_window_name, display_live_frame)
                    else:
                        pass  # Keep previous frame shown if zoom fails

                except Exception as live_process_err:
                    print(f"Error processing live frame: {live_process_err}")
            else:
                print("Warning: Failed to grab frame for live feed.")
            # ---


            # --- Handle Key Press for Quit or Capture ---
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Quit key pressed.")
                quit_flag = True
                break



            if trigger:
                print("Received request on /trigger-inference")



            #elif key == 13: # Enter key - Trigger inspection
                print("/nEnter pressed, inspecting tag (Capture -> Zoom -> Retry Detect -> Rotate/Crop)...")
                result = qc.inspect_tag()
                img_display = result.get("image_data")

                if "error" in result:
                     print(f"Inspection Error: {result['error']}")
                     window_title = f"Error: {result['error']}"
                elif result.get("class") == "Detection Failed":
                     print("Inspection finished: Detection Failed (showing zoomed image).")
                     window_title = "Result: Detection Failed"
                else:
                     print(f"Displaying result: {result['class']} (Confidence: {result['confidence']:.2f})")
                     window_title = f"Result: {result['class']} (Conf: {result['confidence']:.2f})"

                if img_display is not None and img_display.size > 0:
                     try:
                          cv2.imshow(result_window_name, img_display)
                          cv2.setWindowTitle(result_window_name, window_title)
                          print(f"-> '{result_window_name}' updated. Press Enter to capture next, 'q' to quit.")
                     except Exception as display_e:
                          print(f"Error displaying result image in '{result_window_name}': {display_e}")
                else:
                     print("No image data available to display for result.")

            # --- End of Enter Key Handling ---

    except Exception as e:
        print(f"/nA critical error occurred: {e}")
        traceback.print_exc()
    finally:
        print("/nCleaning up...")
        if qc is not None:
            qc.close()
        cv2.destroyAllWindows()
        print("Program finished.")

