import cv2
from ultralytics import YOLO
import time
import os
import traceback  # For detailed error printing
import numpy as np
import threading  # For Flask thread only
from queue import Queue, Empty # Use Queue for thread-safe communication
import sys
from flask import Flask, jsonify
import json # Added for saving JSON results in inspect_tag


# --- Camera Configuration ---
MODEL_PATH = "best 6.pt" # Path to your trained .pt model
CAMERA_ID = 0  # Change if you have multiple cameras
REQUESTED_WIDTH = 1920  # camera width
REQUESTED_HEIGHT = 1080  # camera height
ZOOM_FACTOR = 2  # 1 = no zoom
AUTO_EXPOSURE = False
MANUAL_EXPOSURE_STOP = -7
CAMERA_FPS = 30

# --- Detection Configuration (for live feed bounding box) ---
MIN_RECT_AREA = 10000  # Minimum pixel area
APPROX_POLY_EPSILON_FACTOR = 0.05  # amount of distortion due to camera
MIN_ASPECT_RATIO = 0.2
MAX_ASPECT_RATIO = 5.0
MORPH_KERNEL_SIZE = (5, 5)  # Morphological kernel size
CROP_REDUCTION_FACTOR = 0.8 # Crop to 90% (10% smaller) of the detected bbox size

# --- Display Configuration ---
GOOD_COLOR = (0, 255, 0) # BGR Green
BAD_COLOR = (0, 0, 255)   # BGR Red
ERROR_COLOR = (0, 255, 255) # BGR Yellow
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 1.0
TEXT_THICKNESS = 2
TEXT_ORG = (10, 30) # Top-left corner

# --- Flask ---
app = Flask(__name__)
HOST = "127.0.0.1"
PORT = 2000
# ---

# --- Shared Resources ---
qc = None
display_queue = Queue(maxsize=1)
# ---


class NameTagQualityControl:



    def __init__(self, model_path, camera_id=0, zoom_factor=1):
        """
        Initializes the NameTagQualityControl class. Sets up camera,
        and configures settings.
        """
        self.model = YOLO(model_path)

        # --- Initialize Camera ---
        print(f"Attempting to open camera ID {camera_id}...")
        self.camera = cv2.VideoCapture(camera_id)


        if not self.camera.isOpened():
             raise Exception(f"Error: Could not open camera with ID {camera_id} using the selected backend.")
        print(f"Camera backend API name: {self.camera.getBackendName()}")
        # ---

        # --- Configure Camera Settings ---
        print("Configuring camera...")
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_HEIGHT)

        success_fps = self.camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        if not success_fps:
            print("Warning: Could not set desired FPS using cap.set().")
        else:
            print(f"Set camera FPS to {CAMERA_FPS}")

        time.sleep(1) # Allow settings to apply

        if AUTO_EXPOSURE:
            print("Setting Auto Exposure (0.25)...")
            self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        else:
            print(f"Setting Manual Exposure (Value: {MANUAL_EXPOSURE_STOP})...")
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




    def capture_frame_directly(self):
        """Captures a single frame directly from the camera."""
        if not self.camera.isOpened():
            print("Error: Camera is not open.")
            return None
        ret, frame = self.camera.read()
        if not ret or frame is None:
            print("Warning: Direct frame capture failed.")
            return None
        return frame



    def _apply_zoom(self, frame):
        """Applies center zoom to the frame."""
        if frame is None or frame.size == 0: return None
        if self.zoom_factor <= 1.0: return frame # No zoom needed
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



    @staticmethod
    def _find_largest_rectangle_contour(contours, frame_shape):
        """Finds the largest contour that approximates to 4 vertices and meets criteria."""
        largest_rectangle_contour = None
        max_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_RECT_AREA: continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0: continue
            approx = cv2.approxPolyDP(contour, APPROX_POLY_EPSILON_FACTOR * perimeter, True)
            if len(approx) == 4:
                if not cv2.isContourConvex(approx): continue
                x, y, w, h = cv2.boundingRect(approx)
                if w <= 0 or h <= 0: continue
                aspect_ratio = float(w) / h if h > 0 else 0
                inv_aspect_ratio = float(h) / w if w > 0 else 0
                is_valid_aspect_ratio = (MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO) or \
                                        (MIN_ASPECT_RATIO < inv_aspect_ratio < MAX_ASPECT_RATIO)
                if not is_valid_aspect_ratio: continue
                if area > max_area:
                    max_area = area
                    largest_rectangle_contour = contour
        return largest_rectangle_contour



    def _detect_rectangle_info(self, frame):
        """Detects the largest valid rectangle contour in the frame. Used for live feed bounding box."""
        if frame is None or frame.size == 0: return None

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            edges = cv2.Canny(blurred, 30, 100)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_SIZE)
            closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            edges_processed = closed_edges
        except cv2.error as e:
            print(f"OpenCV error during preprocessing: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error during preprocessing: {e}")
            traceback.print_exc()
            return None

        contours, _ = cv2.findContours(edges_processed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        largest_rect_contour = self._find_largest_rectangle_contour(contours, frame.shape)
        if largest_rect_contour is None: return None
        bbox = cv2.boundingRect(largest_rect_contour)
        min_rect = cv2.minAreaRect(largest_rect_contour) # ((cx,cy),(w,h),angle)
        return {'bbox': bbox, 'min_rect': min_rect}

    def _crop_frame_to_bbox(self, frame, bbox):
         """
         Crops the frame to a region 10% smaller than the provided bounding box,
         centered within the original bbox.
         """
         if frame is None or bbox is None:
              print("Warning: Invalid input to _crop_frame_to_bbox")
              return None
         x, y, w_rect, h_rect = bbox

         # Calculate the center of the original bounding box
         center_x = x + w_rect / 2.0
         center_y = y + h_rect / 2.0

         # Calculate the new width and height (reduced by 10%)
         new_w = int(w_rect * CROP_REDUCTION_FACTOR)
         new_h = int(h_rect * CROP_REDUCTION_FACTOR)

         # Ensure new dimensions are at least 1 pixel
         if new_w <= 0 or new_h <= 0:
              print("Warning: Reduced crop dimensions are too small.")
              # Fallback: return original crop? Or None? Let's try original crop.
              return self._crop_frame_to_bbox_exact(frame, bbox) # Call a helper for exact crop


         # Calculate new top-left corner based on center and new dimensions
         new_x1 = int(round(center_x - new_w / 2.0))
         new_y1 = int(round(center_y - new_h / 2.0))

         # Ensure new coordinates are within the frame boundaries
         new_x1 = max(0, new_x1)
         new_y1 = max(0, new_y1)
         # Calculate bottom-right based on new top-left and new dimensions
         new_x2 = min(frame.shape[1], new_x1 + new_w)
         new_y2 = min(frame.shape[0], new_y1 + new_h)

         # Recalculate width/height based on clamped coordinates
         final_w = new_x2 - new_x1
         final_h = new_y2 - new_y1

         if final_w <= 0 or final_h <= 0:
              print(f"Error: Final calculated crop dimensions invalid: {final_w}x{final_h}")
              return None

         # Perform the smaller, centered crop
         cropped_frame = frame[new_y1:new_y2, new_x1:new_x2]

         if cropped_frame.size == 0:
              print("Warning: Cropping to smaller bbox resulted in empty image.")
              return None
         return cropped_frame

    def _crop_frame_to_bbox_exact(self, frame, bbox):
         """Helper function for exact bounding box cropping (used as fallback)."""
         if frame is None or bbox is None: return None
         x, y, w_rect, h_rect = bbox
         if x < 0 or y < 0 or w_rect <= 0 or h_rect <= 0 or (x + w_rect > frame.shape[1]) or (y + h_rect > frame.shape[0]):
              return None
         cropped_frame = frame[y:y + h_rect, x:x + w_rect]
         if cropped_frame.size == 0: return None
         return cropped_frame


    def inspect_tag(self):
        """
        Captures frame, zooms, detects rectangle, crops to bounding box (10% smaller),
        and runs inference on the cropped image. Falls back to zoomed image if detection fails.
        Determines display text and color based on result.
        """
        result_data = {
            "status": "Error", "error_message": "", "class": "",
            "confidence": 0.0, "image_data": None,
            "display_text": "Error", "display_color": ERROR_COLOR # Default display info
        }
        final_image_for_inference = None
        is_detection_fallback = False

        # 1. Capture Raw Image directly
        raw_image = self.capture_frame_directly()
        if raw_image is None:
             result_data["error_message"] = "Image capture failed"
             return result_data

        raw_image_copy = raw_image.copy()

        # 2. Apply Zoom
        zoomed_image = self._apply_zoom(raw_image_copy)
        if zoomed_image is None:
             result_data["error_message"] = "Zoom failed"
             result_data["image_data"] = raw_image_copy # Store raw if zoom failed
             return result_data

        # 3. Detect Rectangle Info (bbox is needed for cropping)
        rect_info = self._detect_rectangle_info(zoomed_image.copy())

        if rect_info and 'bbox' in rect_info:
            # 4. Crop to Bounding Box (10% smaller)
            print("Rectangle detected, cropping smaller area...")
            final_image_for_inference = self._crop_frame_to_bbox(zoomed_image, rect_info['bbox']) # Uses the new logic
            if final_image_for_inference is None:
                 print("Warning: Cropping failed. Falling back to zoomed image.")
                 final_image_for_inference = zoomed_image # Fallback to zoomed
                 is_detection_fallback = True
                 result_data["error_message"] = "Cropping failed after detection"
                 result_data["display_text"] = "Detect Fail"
                 result_data["display_color"] = ERROR_COLOR
            # else: Cropping successful
        else:
            # 5. Fallback to zoomed image if detection failed
            print("Rectangle detection failed. Using zoomed image for inference.")
            final_image_for_inference = zoomed_image
            is_detection_fallback = True
            result_data["error_message"] = "Rectangle detection failed"
            result_data["display_text"] = "Detect Fail"
            result_data["display_color"] = ERROR_COLOR

        # Store the image that will be used for inference
        result_data["image_data"] = final_image_for_inference

        # --- Inference ---
        if final_image_for_inference is not None and final_image_for_inference.size > 0:
            temp_path = None
            try: # Save temp image
                timestamp_str = f"{int(time.time())}"
                temp_path = os.path.join(self.results_dir, f"temp_inspect_{timestamp_str}.jpg")
                save_success = cv2.imwrite(temp_path, final_image_for_inference)
                if not save_success: temp_path = None
            except Exception as save_err: print(f"Error saving temp image: {save_err}"); temp_path = None

            try: # Run Inference
                height, width = final_image_for_inference.shape[:2]
                print(f"Running inference on image size: {width}x{height}")
                results = self.model(source=final_image_for_inference, rect=True, imgsz=(height, width))

                if not results or not hasattr(results[0], 'probs') or results[0].probs is None:
                    if not is_detection_fallback: # Only update if not already detection fail
                       result_data["error_message"] = "Inference error (no probabilities)"
                       result_data["display_text"] = "Inference Error"
                       result_data["display_color"] = ERROR_COLOR
                    result_data["status"] = "Failed" if is_detection_fallback else "Error"
                else: # Inference Success
                    predicted_class_index = results[0].probs.top1
                    confidence = float(results[0].probs.top1conf)
                    if results[0].names and predicted_class_index < len(results[0].names):
                        class_name = results[0].names[predicted_class_index]
                    else:
                        class_name = f"Unknown Index ({predicted_class_index})"

                    result_data["class"] = class_name
                    result_data["confidence"] = confidence
                    result_data["status"] = "Success"

                    # Determine display text/color based on class_name
                    if class_name.lower() == 'good':
                         result_data["display_text"] = "Good"
                         result_data["display_color"] = GOOD_COLOR
                    else:
                         result_data["display_text"] = "Bad" # Assume others are Bad
                         result_data["display_color"] = BAD_COLOR

            except Exception as model_err:
                print(f"Error during model inference: {model_err}")
                traceback.print_exc()
                if not is_detection_fallback: # Only update if not already detection fail
                    result_data["error_message"] = f"Inference failed: {model_err}"
                    result_data["display_text"] = "Inference Error"
                    result_data["display_color"] = ERROR_COLOR
                result_data["status"] = "Failed" if is_detection_fallback else "Error"
        else:
            result_data["error_message"] = "Image for inference was None or empty"
            result_data["display_text"] = "Processing Error"
            result_data["display_color"] = ERROR_COLOR
            result_data["status"] = "Error"


        # Ensure class/confidence are cleared if status is Error/Failed
        if result_data["status"] != "Success":
            result_data["class"] = ""
            result_data["confidence"] = 0.0
            # Keep the specific error message and display text/color if one was set

        # --- Save JSON ---
        latest_result_path = os.path.join(self.results_dir, "latest_result.json")
        try:
            json_data_to_save = {
                "status": result_data["status"],
                "class": result_data["class"],
                "confidence": result_data["confidence"]
            }
            if "error_message" in result_data and result_data["error_message"]:
                 json_data_to_save["error_message"] = result_data["error_message"]
            with open(latest_result_path, "w") as f:
                json.dump(json_data_to_save, f, indent=4)
        except IOError as e:
            print(f"Error writing results to {latest_result_path}: {e}")

        return result_data



    def close(self):
        """Releases the camera resource."""
        print("Initiating shutdown...")
        if self.camera.isOpened():
            print("Releasing camera resource...")
            self.camera.release()
            print("Camera released.")
        else:
             print("Camera was not open.")
        print("Shutdown complete.")



# --- Flask Route ---
@app.route("/trigger-inference", methods=["GET"])
def handle_trigger():
    """
    Handles GET requests to trigger inspection.
    Returns JSON {status, class, confidence}.
    Puts image data, title, text, and color onto display_queue for the main thread.
    """
    global qc, display_queue
    print("Received request on /trigger-inference")
    error_response = {"status": "Error", "class": "", "confidence": 0.0}
    if qc is None:
        print("Error: QualityControl object (qc) not initialized.")
        error_response["error_message"] = "Inspection system not ready"
        return jsonify(error_response), 500

    try:
        # Perform the inspection
        result = qc.inspect_tag() # Returns the detailed dict including display info

        # Prepare the final JSON response
        response_data = {
            "status": result.get("status", "Error"),
            "class": result.get("class", ""),
            "confidence": result.get("confidence", 0.0),
        }
        if "error_message" in result and result["error_message"]:
             response_data["error_message"] = result["error_message"]

        # --- Put display info onto the queue ---
        img_display = result.get("image_data")
        display_text = result.get("display_text", "Error")
        display_color = result.get("display_color", ERROR_COLOR)
        window_title = f"API Result: {display_text}" # Simpler title for API

        if img_display is not None:
            # Clear previous item if queue is full
            if display_queue.full():
                try: display_queue.get_nowait()
                except Empty: pass
            # Put new item: (title, text, color, image)
            try:
                 display_queue.put_nowait((window_title, display_text, display_color, img_display))
                 print("Display info put on queue.")
            except Exception as q_err:
                 print(f"Error putting item on display queue: {q_err}")
        else:
             print("No image data to put on display queue.")
        # --- End queue logic ---

        print(f"Inspection complete. Returning result: {response_data}")
        return jsonify(response_data)

    except Exception as e:
        print(f"Critical error during triggered inspection: {e}")
        traceback.print_exc()
        error_response["error_message"] = f"Inspection failed: {str(e)}"
        return jsonify(error_response), 500



def start_flask_app():
    """Starts the Flask development server in a background thread."""
    print(f"Starting Flask server on http://{HOST}:{PORT}")
    try:
        app.run(host=HOST, port=PORT, threaded=True, debug=False, use_reloader=False)
    except Exception as e:
        print(f"Error to start Flask app: {e}")
        traceback.print_exc()



# --- Main Execution Block ---
if __name__ == "__main__":

    if not os.path.exists(MODEL_PATH):
        print(f"CRITICAL ERROR: Model file not found at {MODEL_PATH}")
        sys.exit(1)

    quit_flag = False
    live_window_name = "Live View (Zoomed + Rotated Box)"
    result_window_name = "Last Inspection Result" # Simplified name
    result_window_created = False
    flask_thread = None

    try:
        print("Initializing Quality Control system...")
        qc = NameTagQualityControl(MODEL_PATH, camera_id=CAMERA_ID, zoom_factor=ZOOM_FACTOR)
        print("Quality Control system initialized.")

        print("Starting Flask server thread...")
        flask_thread = threading.Thread(target=start_flask_app, daemon=True)
        flask_thread.start()
        time.sleep(1)
        if not flask_thread.is_alive():
             print("CRITICAL ERROR: Flask server thread failed to start.")
             if qc: qc.close()
             sys.exit(1)
        print("Flask server thread started.")

        print("\n--- System Ready ---")
        print(f"Live view shows zoomed view with rotated bounding box. Trigger inspection via GET request to http://{HOST}:{PORT}/trigger-inference")
        print("Press Enter for manual inspection (capture->zoom->detect->crop->inference).")
        print("Press 'q' in any OpenCV window to quit.")
        if AUTO_EXPOSURE:
            print(f"Using Auto Exposure (Behavior depends on driver)")
        else:
            print(f"Using Manual Exposure setting: {MANUAL_EXPOSURE_STOP}")
        print("--------------------\n")

        while not quit_flag:
            if qc is None or not qc.camera.isOpened():
                print("Error: QC object not available or camera closed unexpectedly. Exiting.")
                quit_flag = True
                break

            # --- Live Video View ---
            live_frame_raw = qc.capture_frame_directly()
            display_live_frame = None

            if live_frame_raw is not None:
                try:
                    live_frame_copy = live_frame_raw.copy()
                    zoomed_live_frame = qc._apply_zoom(live_frame_copy)
                    if zoomed_live_frame is not None:
                        display_live_frame = zoomed_live_frame.copy()
                        live_rect_info = qc._detect_rectangle_info(zoomed_live_frame.copy())

                        # --- Draw Rotated Bounding Box ---
                        if live_rect_info and 'min_rect' in live_rect_info:
                            box = cv2.boxPoints(live_rect_info['min_rect'])
                            box = np.intp(box)
                            cv2.drawContours(display_live_frame, [box], 0, (0, 255, 0), 2)

                        cv2.imshow(live_window_name, display_live_frame)
                except Exception as live_process_err:
                    print(f"Error processing live frame: {live_process_err}")

            # --- Check Display Queue from Flask ---
            try:
                # Check queue without blocking
                title, text, color, img = display_queue.get_nowait()
                print(f"Displaying result from API request: {title}")
                if img is not None and img.size > 0:
                    try:
                        # Draw text overlay before showing
                        img_to_show = img.copy() # Draw on a copy
                        cv2.putText(img_to_show, text, TEXT_ORG, TEXT_FONT, TEXT_SCALE, color, TEXT_THICKNESS, cv2.LINE_AA)
                        cv2.imshow(result_window_name, img_to_show)
                        cv2.setWindowTitle(result_window_name, title)
                        result_window_created = True
                    except Exception as display_e:
                        print(f"Error displaying queued image: {display_e}")
                else:
                     print("Queued image data was None or empty.")
                     if result_window_created:
                          try: # Try to clear window with placeholder
                               placeholder = np.zeros((100, 400, 3), dtype=np.uint8)
                               cv2.putText(placeholder, "No Image Data", (10, 30), TEXT_FONT, 0.5, (255, 255, 255), 1)
                               cv2.putText(placeholder, title, (10, 70), TEXT_FONT, 0.4, (255, 255, 255), 1)
                               cv2.imshow(result_window_name, placeholder)
                               cv2.setWindowTitle(result_window_name, title + " (No Image)")
                          except cv2.error: result_window_created = False
                          except Exception as placeholder_e: print(f"Error displaying placeholder: {placeholder_e}")

            except Empty:
                pass # No update from Flask thread
            except Exception as q_err:
                print(f"Error checking display queue: {q_err}")
            # --- End Display Queue Check ---


            # --- Handle Key Press for Quit or Manual Capture ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit key ('q') pressed. Shutting down...")
                quit_flag = True
                break
            elif key == 13:
                print("\nEnter key pressed, performing manual inspection...")
                if qc:
                    result = qc.inspect_tag() # Handles capture, zoom, detect, crop, inference
                    img_display = result.get("image_data") # This is cropped or zoomed
                    display_text = result.get("display_text", "Error")
                    display_color = result.get("display_color", ERROR_COLOR)
                    window_title = result_window_name # Base title

                    # Determine title based on success, failure, or error
                    status = result.get("status")
                    cls = result.get("class", "")
                    conf = result.get("confidence", 0.0)
                    err_msg = result.get("error_message", None)

                    if status == "Success":
                         print(f"Manual, Result - class: {cls} confidence: {conf:.2f}")
                         window_title = f"Manual: {display_text} ({conf:.2f})"
                    elif status == "Failed": # Detection failed, used zoomed
                         print(f"Manual, Warning: {err_msg}. Inference on zoomed. Class: {cls}, Conf: {conf:.2f}")
                         window_title = f"Manual: {display_text}"
                    else: # Error status
                         print(f"Manual, Inspection Error: {err_msg or 'Unknown Critical Error'}")
                         window_title = f"Manual: {display_text}"


                    # Display the result image (cropped or zoomed fallback)
                    if img_display is not None and img_display.size > 0:
                         try:
                              # Draw text overlay before showing
                              img_to_show = img_display.copy()
                              cv2.putText(img_to_show, display_text, TEXT_ORG, TEXT_FONT, TEXT_SCALE, display_color, TEXT_THICKNESS, cv2.LINE_AA)
                              cv2.imshow(result_window_name, img_to_show)
                              cv2.setWindowTitle(result_window_name, window_title)
                              result_window_created = True
                              print(f"-> '{result_window_name}' updated. Press Enter for next manual, 'q' to quit.")
                         except cv2.error as display_e:
                              print(f"OpenCV Error displaying result image: {display_e}")
                         except Exception as display_e:
                              print(f"Error displaying result image: {display_e}")
                    else:
                         print("No image data available to display for manual result.")
                         if result_window_created:
                              try: # Try to clear window with placeholder
                                   placeholder = np.zeros((100, 400, 3), dtype=np.uint8)
                                   cv2.putText(placeholder, "No Image Data", (10, 30), TEXT_FONT, 0.5, (255, 255, 255), 1)
                                   cv2.putText(placeholder, window_title, (10, 70), TEXT_FONT, 0.4, (255, 255, 255), 1)
                                   cv2.imshow(result_window_name, placeholder)
                                   cv2.setWindowTitle(result_window_name, window_title + " (No Image)")
                              except cv2.error: result_window_created = False
                              except Exception as placeholder_e: print(f"Error displaying placeholder: {placeholder_e}")
                else:
                    print("Cannot perform manual inspection: QC object not initialized.")

            if flask_thread and not flask_thread.is_alive():
                print("CRITICAL ERROR: Flask server thread has stopped unexpectedly. Exiting.")
                quit_flag = True

    except Exception as e:
        print(f"\nA critical error occurred in the main setup or loop: {e}")
        traceback.print_exc()
    finally:
        print("\n--- Initiating Final Cleanup ---")
        if qc is not None:
            print("Closing Quality Control system...")
            qc.close()
        else:
            print("QC object was not initialized, skipping QC close.")
        print("Destroying OpenCV windows...")
        cv2.destroyAllWindows()
        print("OpenCV windows destroyed.")
        if 'flask_thread' in locals() and flask_thread is not None and flask_thread.is_alive():
            print("Flask server thread is still alive (as expected for daemon).")
        print("--- Program Finished ---")
