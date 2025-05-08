import cv2
from ultralytics import YOLO
import time
import os
import traceback  # For detailed error printing
import numpy as np
import threading  # For dedicated capture thread
from queue import Queue, Empty
import sys
from flask import Flask, jsonify



# --- Camera Configuration ---
MODEL_PATH = "runs/classify/train3/weights/best.pt" # Path to your trained .pt model
CAMERA_ID = 0  # Change if you have multiple cameras
REQUESTED_WIDTH = 1920  # camera width
REQUESTED_HEIGHT = 1080  # camera height
ZOOM_FACTOR = 3  # 1 is no zoom
AUTO_EXPOSURE = False
MANUAL_EXPOSURE_STOP = -5
CAMERA_FPS = 30

# --- Detection Configuration ---
MIN_RECT_AREA = 10000  # Minimum pixel area
RECT_DETECT_RETRIES = 10  # Number of times to retry; setting it too high will cause image corruption
APPROX_POLY_EPSILON_FACTOR = 0.05  # amount of distortion due to camera
MIN_ASPECT_RATIO = 0.2
MAX_ASPECT_RATIO = 5.0
MORPH_KERNEL_SIZE = (5, 5)  # Morphological kernel size

# --- Flask ---
app = Flask(__name__)
HOST = "127.0.0.1"
PORT = 2000
# ---

# This allows the Flask route to access the qc instance created in __main__
qc = None
# ---



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

        # Add other backend options here if needed (DSHOW, MSMF, AVFOUNDATION)
        # ...

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

        self.zoom_factor = zoom_factor

        # --- Threaded Capture Setup ---
        self.capture_queue = Queue(maxsize=5)
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
                try:
                    # Ensure the queue is not full before putting (non-blocking)
                    if self.capture_queue.full():
                        self.capture_queue.get_nowait() # Discard older frame if full
                    self.capture_queue.put(frame) # Put the new frame
                    frame_counter += 1
                    # Optional: Reduce frequency of status printing
                    # if frame_counter % 1000 == 0:
                    #      print(f"Capture thread status: {frame_counter} frames captured, {error_counter} errors.")

                except Empty: # Catch Empty exception from get_nowait
                    # This case should ideally not happen if .full() check is done first,
                    # but good to handle defensively.
                    print("Warning: get_nowait() called on an empty queue unexpectedly.")
                    error_counter += 1
                except Exception as e:
                     print(f"Error putting frame in queue: {e}")
                     error_counter += 1
            else:
                # Error to read frame
                error_counter += 1
                if error_counter > 100 and error_counter % 50 == 0: # Log if many consecutive errors
                     print(f"Warning: {error_counter} consecutive camera read failures!")
                time.sleep(0.05) # Wait a bit before retrying camera read
        print("Capture loop stopped.")

    def stop_capture(self):
        """Signals the capture thread to stop."""
        print("Stopping capture thread...")
        self.capture_active = False
        # Attempt to clear the queue to potentially unblock the put() call if thread is stuck
        while not self.capture_queue.empty():
            try:
                self.capture_queue.get_nowait()
            except Empty:
                break



    def get_latest_frame(self):
        """Gets the latest frame from the capture queue."""
        try:
            # Use a timeout to prevent blocking indefinitely if the queue is empty
            return self.capture_queue.get(timeout=0.5)
        except Empty: # Use specific exception
            # print("Warning: Capture queue empty or timeout.") # Reduce noise
            return None
        except Exception as e: # Catch other potential errors
            print(f"Error getting frame from queue: {e}")
            return None



    def _apply_zoom(self, frame):
        """Applies center zoom to the frame."""
        if frame is None or frame.size == 0: return None
        if self.zoom_factor <= 1.0: return frame # No zoom needed
        h, w = frame.shape[:2]
        # Calculate center
        center_x, center_y = w // 2, h // 2
        # Calculate new dimensions
        new_w = int(w / self.zoom_factor)
        new_h = int(h / self.zoom_factor)
        # Calculate top-left corner, ensuring it's within bounds
        x1 = max(0, center_x - new_w // 2)
        y1 = max(0, center_y - new_h // 2)
        # Calculate bottom-right corner, ensuring it's within bounds
        x2 = min(w, x1 + new_w)
        y2 = min(h, y1 + new_h)
        # Check for valid crop dimensions
        if x1 >= x2 or y1 >= y2: return None
        # Perform the crop (zoom)
        zoomed_frame = frame[y1:y2, x1:x2]
        # Final check if the zoomed frame is valid
        if zoomed_frame.size == 0: return None
        return zoomed_frame



    @staticmethod
    def _find_largest_rectangle_contour(contours, frame_shape):
        """Finds the largest contour that approximates to 4 vertices and meets criteria."""
        largest_rectangle_contour = None
        max_area = 0

        for contour in contours:
            # Calculate contour area
            area = cv2.contourArea(contour)
            # Filter small contours
            if area < MIN_RECT_AREA: continue

            # Calculate perimeter
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0: continue # Avoid division by zero in approxPolyDP

            # Approximate the contour shape
            approx = cv2.approxPolyDP(contour, APPROX_POLY_EPSILON_FACTOR * perimeter, True)

            # Check if the approximation has 4 vertices (is a quadrilateral)
            if len(approx) == 4:
                # Check if the quadrilateral is convex
                if not cv2.isContourConvex(approx): continue

                # Get the bounding box to check aspect ratio
                x, y, w, h = cv2.boundingRect(approx)
                if w <= 0 or h <= 0: continue # Invalid dimensions

                # Calculate aspect ratios
                aspect_ratio = float(w) / h if h > 0 else 0
                inv_aspect_ratio = float(h) / w if w > 0 else 0

                # Check if aspect ratio is within the allowed range (either orientation)
                is_valid_aspect_ratio = (MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO) or \
                                        (MIN_ASPECT_RATIO < inv_aspect_ratio < MAX_ASPECT_RATIO)
                if not is_valid_aspect_ratio: continue

                # If this contour is larger than the previous largest valid one, update
                if area > max_area:
                    max_area = area
                    largest_rectangle_contour = contour

        return largest_rectangle_contour



    def _detect_rectangle_info(self, frame):
        """Detects the largest valid rectangle contour in the frame."""
        if frame is None or frame.size == 0: return None # Check for valid input

        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (7, 7), 0) # Slightly larger kernel might help
            # Apply Canny edge detection
            edges = cv2.Canny(blurred, 30, 100) # Adjusted thresholds
            # Define a morphological kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_SIZE)
            # Apply Morphological Closing (dilate then erode) to close gaps in edges
            closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            # Use the closed edges for contour finding
            edges_processed = closed_edges
        except cv2.error as e:
            print(f"OpenCV error during preprocessing: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error during preprocessing: {e}")
            traceback.print_exc() # Print traceback for unexpected errors
            return None

        # Find contours in the processed edge image
        contours, _ = cv2.findContours(edges_processed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If no contours are found, return None
        if not contours: return None

        # Find the largest valid rectangle among the contours
        largest_rect_contour = self._find_largest_rectangle_contour(contours, frame.shape)

        # If no valid rectangle contour is found, return None
        if largest_rect_contour is None: return None

        # If a valid rectangle is found, calculate its bounding box and minimum area rectangle
        bbox = cv2.boundingRect(largest_rect_contour) # Axis-aligned bounding box
        min_rect = cv2.minAreaRect(largest_rect_contour) # Minimum area (rotated) bounding box

        # Return the information
        return {'bbox': bbox, 'min_rect': min_rect}



    def _rotate_and_crop(self, frame, min_rect):
         """Rotates the frame to align the min_rect and crops it."""
         if frame is None or min_rect is None: return None

         # Unpack the minimum area rectangle parameters
         center, (width, height), angle = min_rect

         # Adjust angle and dimensions if necessary
         if angle < -45:
              angle += 90.0
              width, height = height, width # Swap width and height

         # Get the rotation matrix
         M = cv2.getRotationMatrix2D(center, angle, 1.0) # angle, scale=1.0

         try:
              # Calculate the bounding box of the rotated rectangle
              box = cv2.boxPoints(min_rect)
              pts = np.intp(box) # Use np.intp for potentially large coordinates

              cropped = pts

              x_coords = pts[:, 0]
              y_coords = pts[:, 1]
              rotated_bbox_w = int(np.max(x_coords) - np.min(x_coords))
              rotated_bbox_h = int(np.max(y_coords) - np.min(y_coords))

              # Use the dimensions of the rotated bounding box for warp size
              warp_w = rotated_bbox_w
              warp_h = rotated_bbox_h
              if warp_w <= 0 or warp_h <= 0: raise Exception("Invalid warp dimensions calculated")

              # Adjust the rotation matrix translation component to center the rectangle
              M[0, 2] += (warp_w / 2) - center[0]
              M[1, 2] += (warp_h / 2) - center[1]

              # Perform the affine transformation
              warped = cv2.warpAffine(frame, M, (warp_w, warp_h), flags=cv2.INTER_CUBIC)
              if warped is None: raise Exception("warpAffine returned None")

         except Exception as e:
              print(f"Error during warpAffine: {e}")
              traceback.print_exc()
              return None # Return None if rotation fails

         # Crop the final rectangle from the center of the warped image
         crop_width = int(round(width))
         crop_height = int(round(height))
         if crop_width <= 0 or crop_height <= 0: return None # Invalid crop dimensions

         try:
              # Calculate the center of the *warped* image
              new_center = (warp_w / 2, warp_h / 2)
              # Crop the rectangle using getRectSubPix
              cropped = cv2.getRectSubPix(warped, (crop_width, crop_height), new_center)
              if cropped is None: raise Exception("getRectSubPix returned None")
         except Exception as e:
              print(f"Error during getRectSubPix: {e}")
              traceback.print_exc()
              return None # Return None if cropping fails

         # Final check on aspect ratio
         if crop_height > crop_width * 1.2: # Add a tolerance
              print("Rotating final cropped image by 90 degrees.")
              cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)

         # Final check for empty result
         if cropped.size == 0: return None
         return cropped



    def inspect_tag(self):
        """
        Captures frame, processes, runs inference. Returns a dictionary with
        status, class, confidence, and image_data, structured as requested.
        """
        # Initialize default return structure for error cases
        result_data = {
            "status": "Error",
            "error_message": "",
            "class": "",
            "confidence": 0.0,
            "image_data": None
        }
        final_image = None # Define final_image early

        # 1. Get Latest Raw Image
        raw_image = self.get_latest_frame()
        if raw_image is None:
             print("Capture queue was empty, attempting direct read...")
             ret_direct, raw_image = self.camera.read()
             if not ret_direct or raw_image is None:
                 print("Error: Image capture failed (queue empty & direct read failed)")
                 result_data["error_message"] = "Image capture failed" # Optional: add error detail
                 return result_data # Return default error structure

        # Work on a copy
        raw_image = raw_image.copy()
        result_data["image_data"] = raw_image # Store raw initially, might be overwritten



        # 2. Apply Zoom
        zoomed_image = self._apply_zoom(raw_image)
        if zoomed_image is None:
             print("Warning: Zoom processing failed.")
             # Image data remains the raw_image stored earlier
             result_data["status"] = "Error"
             result_data["error_message"] = "Zoom failed"
             return result_data # Return structure indicating processing issue

        result_data["image_data"] = zoomed_image # Update image data to zoomed

        # 3. Detect Rectangle Coordinates (with Retries)
        rect_info = None
        for attempt in range(RECT_DETECT_RETRIES):
             rect_info = self._detect_rectangle_info(zoomed_image.copy())
             if rect_info:
                 break
        # else: # Loop completed without break (detection failed)
             # No need to do anything here, rect_info remains None

        # --- Process based on detection result ---
        if rect_info:
            # 4. Rotate and Crop if rectangle was detected successfully
            final_image = self._rotate_and_crop(zoomed_image, rect_info['min_rect'])
            if final_image is None:
                 # Rotation/cropping failed, fall back to the zoomed image
                 print("Warning: Rotation/Cropping failed. Using zoomed image for inference.")
                 final_image = zoomed_image # Use zoomed as fallback
                 result_data["status"] = "Error"
                 result_data["error_message"] = "Rotation/Cropping failed"
            else:
                 # Cropping successful, update image data
                 result_data["image_data"] = final_image
        else:
            # Detection failed after retries, use the zoomed image for potential inference
            print(f"Warning: Rectangle detection Failed after {RECT_DETECT_RETRIES} retries. Using zoomed image.")
            final_image = zoomed_image # Use zoomed image
            result_data["status"] = "Failed"
            result_data["error_message"] = f"Warning: Rectangle detection Failed after {RECT_DETECT_RETRIES} retries. Using zoomed image."



        # --- Inference ---
        # Proceed only if we have a valid final image (even if it's just the zoomed one)
        if final_image is not None and final_image.size > 0:

            try:
                # Run YOLO model inference
                # imgsz=(height, width)
                height, width = final_image.shape[:2]
                results = self.model(source=final_image, rect=True, imgsz=(height, width))

                # Process results
                if not results or not hasattr(results[0], 'probs') or results[0].probs is None:
                    # Inference failed or returned unexpected format
                    print("Warning: Inference failed or returned no probabilities.")
                    result_data["status"] = "Error"
                    if "error_message" not in result_data: # Add error if not already set
                         result_data["error_message"] = "Inference error (no probabilities)"
                else:
                    # Inference successful, extract results
                    predicted_class_index = results[0].probs.top1
                    confidence = float(results[0].probs.top1conf)
                    if results[0].names and predicted_class_index < len(results[0].names):
                        class_name = results[0].names[predicted_class_index]
                    else:
                        class_name = f"Unknown Index ({predicted_class_index})"

                    result_data["class"] = class_name
                    result_data["confidence"] = confidence
                    result_data["status"] = "Success"

            except Exception as model_err:
                # Catch errors during the inference process itself
                print(f"Error during model inference: {model_err}")
                traceback.print_exc()
                if "error_message" not in result_data: # Add error if not already set
                    result_data["error_message"] = f"Inference failed: {model_err}"

        else: # Handle case where final_image is None or empty before inference
            print("Warning: Final image was None or empty before inference step.")
            if "error_message" not in result_data: # Add error if not already set
                 result_data["error_message"] = "Processing resulted in no image for inference"

        # ignore inference result if Error
        if result_data["status"] == "Error":
            result_data["class"] = ""
            result_data["confidence"] = 0.0

        return result_data



    def close(self):
        """Stops the capture thread and releases the camera resource."""
        print("Initiating shutdown...")
        self.stop_capture() # Signal capture thread to stop

        # Wait for the capture thread to finish
        if self.capture_thread.is_alive():
             print("Waiting for capture thread to join...")
             self.capture_thread.join(timeout=2.0) # Wait up to 2 seconds
             if self.capture_thread.is_alive():
                  print("Warning: Capture thread did not stop gracefully after timeout.")

        # Release the camera resource
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
    Handles GET requests to trigger inspection and returns results
    in the specified JSON format {status, class, confidence}.
    """
    global qc
    print("Received request on /trigger-inference")

    # Define default error response structure
    error_response = {
        "status": "Error",
        "class": "",
        "confidence": 0.0
    }

    if qc is None:
        print("Error: QualityControl object (qc) not initialized.")
        error_response["error_message"] = "Inspection system not ready" # Add detail
        return jsonify(error_response), 500 # Return 500 status code

    try:
        # Perform the inspection
        result = qc.inspect_tag() # This now returns the structured dict

        # Prepare the final JSON response, excluding non-serializable image_data
        response_data = {
            "status": result.get("status", "Error"), # Default to Error if missing
            "class": result.get("class", ""),
            "confidence": result.get("confidence", 0.0),
        }
        # Optionally include the error message if present
        if "error_message" in result:
             response_data["error_message"] = result["error_message"]


        print(f"Inspection complete. Returning result: {response_data}")
        # Return the result as a JSON response with 200 OK status
        return jsonify(response_data)

    except Exception as e:
        # Catch any unexpected errors during the inspection call
        print(f"Critical error during triggered inspection: {e}")
        traceback.print_exc() # Log the full traceback for debugging
        error_response["error_message"] = f"Inspection failed: {str(e)}" # Add detail
        # Return a 500 Internal Server Error
        return jsonify(error_response), 500



def start_flask_app():
    """Starts the Flask development server in a background thread."""
    print(f"Starting Flask server on http://{HOST}:{PORT}")
    try:
        # use_reloader=False is important when running in a thread
        app.run(host=HOST, port=PORT, threaded=True, debug=False, use_reloader=False)
    except Exception as e:
        print(f"Error to start Flask app: {e}")
        traceback.print_exc()



# --- Main Execution Block ---
if __name__ == "__main__":

    # --- Check if model file exists ---
    if not os.path.exists(MODEL_PATH):
        print(f"CRITICAL ERROR: Model file not found at {MODEL_PATH}")
        sys.exit(1) # Use non-zero exit code for errors
    # ---

    quit_flag = False # Flag to control the main loop
    live_window_name = "Live View (Zoomed)"
    result_window_name = "Last Inspection Result (Press Enter for Manual)"
    result_window_created = False # Flag to track if result window is open

    # --- Initialize QC and Start Flask Server ---
    try:
        print("Initializing Quality Control system...")
        # Initialize the Quality Control object and assign it to the global variable
        qc = NameTagQualityControl(MODEL_PATH, camera_id=CAMERA_ID, zoom_factor=ZOOM_FACTOR)
        print("Quality Control system initialized.")

        # Start the Flask server in a separate daemon thread
        print("Starting Flask server thread...")
        flask_thread = threading.Thread(target=start_flask_app, daemon=True)
        flask_thread.start()
        time.sleep(1)   # Give the server a moment to start up
        if not flask_thread.is_alive():
             print("CRITICAL ERROR: Flask server thread failed to start.")
             # Clean up QC before exiting if Flask fails to start
             if qc: qc.close()
             sys.exit(1)
        print("Flask server thread started.")

        print("\n--- System Ready ---")
        print(f"Live view shows zoomed view. Trigger inspection via GET request to http://{HOST}:{PORT}/trigger-inference")
        print("Press Enter in the Live View window for manual capture/inspection.")
        print("Press 'q' in any OpenCV window to quit.")
        if AUTO_EXPOSURE:
            print(f"Using Auto Exposure (Behavior depends on driver)")
        else:
            print(f"Using Manual Exposure setting: {MANUAL_EXPOSURE_STOP}")
        print("--------------------\n")

        # --- Main Loop (Live View and Manual Interaction) ---
        while not quit_flag:
            if qc is None or not qc.camera.isOpened():
                print("Error: QC object not available or camera closed unexpectedly. Exiting.")
                quit_flag = True
                break

            # --- Live Video View ---
            ret_live, live_frame_raw = qc.camera.read()
            display_live_frame = None # Frame to actually display

            if ret_live and live_frame_raw is not None:
                try:
                    # 1. Apply zoom for the live view display
                    zoomed_live_frame = qc._apply_zoom(live_frame_raw)

                    if zoomed_live_frame is not None:
                        # Make a copy to draw on
                        display_live_frame = zoomed_live_frame.copy()

                        # 2. Detect rectangle info for drawing overlay
                        live_rect_info = qc._detect_rectangle_info(zoomed_live_frame.copy())

                        # 3. Draw bounding box if detected
                        if live_rect_info and 'min_rect' in live_rect_info:
                            min_rect = live_rect_info['min_rect']
                            box = cv2.boxPoints(min_rect)
                            box = np.intp(box)
                            cv2.drawContours(display_live_frame, [box], 0, (0, 255, 0), 2) # Green box

                        # Display the potentially annotated frame
                        cv2.imshow(live_window_name, display_live_frame)

                except Exception as live_process_err:
                    print(f"Error processing live frame: {live_process_err}")

            # --- Handle Key Press for Quit or Manual Capture ---
            key = cv2.waitKey(1) & 0xFF # Check for key press (1ms delay)

            if key == ord('q'):
                print("Quit key ('q') pressed. Shutting down...")
                quit_flag = True
                break # Exit the main loop immediately

            # Handle Enter key for manual inspection
            elif key == 13: # Enter key
                print("\nEnter key pressed, performing manual inspection...")
                if qc:
                    result = qc.inspect_tag() # Call the inspection function
                    img_display = result.get("image_data") # Get the processed image

                    # Determine window title based on result structure
                    window_title = result_window_name # Default title
                    status = result.get("status")
                    cls = result.get("class", "")
                    conf = result.get("confidence", 0.0)
                    err_msg = result.get("error_message", None)

                    if status == "Error":
                         print(f"Manual, Inspection Error: {err_msg or 'Unknown Critical Error'}")
                         window_title = f"Manual, Inspection Error: {err_msg or 'Unknown Critical Error'}"
                    elif status == "Failed":
                         print(f"Manual, Warning: {err_msg}")
                         window_title = f"Manual, Warning: {err_msg} Manual Result - class: {cls} confidence: {conf:.2f}"
                    else: #Success
                         print(f"Manual, Result - class: {cls} confidence: {conf:.2f}")
                         window_title = f"Manual, Result - class: {cls} confidence: {conf:.2f}"

                    # Display the result image in a separate window
                    if img_display is not None and img_display.size > 0:
                         try:
                              # Create/update the result window
                              cv2.imshow(result_window_name, img_display)
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
                              try:
                                   # Show placeholder if window exists but no image
                                   placeholder = np.zeros((100, 400, 3), dtype=np.uint8)
                                   cv2.putText(placeholder, "No Image Data", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                   cv2.putText(placeholder, window_title, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                                   cv2.imshow(result_window_name, placeholder)
                                   cv2.setWindowTitle(result_window_name, window_title + " (No Image)")
                              except cv2.error:
                                   result_window_created = False # Window likely closed
                              except Exception as placeholder_e:
                                   print(f"Error displaying placeholder: {placeholder_e}")
                else:
                    print("Cannot perform manual inspection: QC object not initialized.")

            # --- End of Key Handling ---

            # Check if Flask thread is still alive
            if not flask_thread.is_alive():
                print("CRITICAL ERROR: Flask server thread has stopped unexpectedly. Exiting.")
                quit_flag = True # Stop main loop if server dies

    except Exception as e:
        print(f"\nA critical error occurred in the main setup or loop: {e}")
        traceback.print_exc()
    finally:
        # --- Cleanup ---
        print("\n--- Initiating Final Cleanup ---")
        if qc is not None:
            print("Closing Quality Control system...")
            qc.close()
        else:
            print("QC object was not initialized, skipping QC close.")

        print("Destroying OpenCV windows...")
        cv2.destroyAllWindows()
        print("OpenCV windows destroyed.")

        if 'flask_thread' in locals() and flask_thread.is_alive():
            print("Flask server thread is still alive (as expected for daemon).")

        print("--- Program Finished ---")