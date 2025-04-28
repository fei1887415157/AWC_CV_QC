import cv2
from ultralytics import YOLO
import time
import json
import os
import traceback # For detailed error printing
import numpy as np



# best to worst model: n, m, x/s/l


# ---
MODEL_PATH = "runs/classify/11s/weights/best.pt"
# ---

# --- Camera ---
CAMERA_ID = 0 # Change if you have multiple cameras
REQUESTED_WIDTH = 1920 # camera width
REQUESTED_HEIGHT = 1080 # camera height
ZOOM_FACTOR = 3 # 1 is no zoom
AUTO_EXPOSURE = False
MANUAL_EXPOSURE_STOP = -5
# ---

# --- Rectangle Detection ---
MIN_RECT_AREA = 10000 # Minimum pixel area
RECT_DETECT_RETRIES = 100 # Number of times to retry
APPROX_POLY_EPSILON_FACTOR = 0.05 # amount of distortion due to camera, do not use too high value
MIN_ASPECT_RATIO = 0.2
MAX_ASPECT_RATIO = 5.0
MORPH_KERNEL_SIZE = (5, 5) # Morphological kernel size
# ---



class NameTagQualityControl:
    def __init__(self, model_path, camera_id=0, zoom_factor=1):
        """
        Initializes the NameTagQualityControl class. Attempts to set camera
        resolution and exposure mode.

        Args:
            model_path (str): Path to the trained YOLO classification model.
            camera_id (int): ID of the camera to use (default is 0).
            zoom_factor (float): The desired zoom factor (e.g., 2.0 for 2x zoom).
        """
        self.model = YOLO(model_path)
        self.camera = cv2.VideoCapture(camera_id)
        if not self.camera.isOpened():
             raise Exception(f"Error: Could not open camera with ID {camera_id}")

        # --- Attempt to set camera resolution ---
        print(f"Attempting to set camera resolution to {REQUESTED_WIDTH}x{REQUESTED_HEIGHT}...")
        set_w = self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_WIDTH)
        set_h = self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_HEIGHT)

        # --- Verify the actual resolution set ---
        time.sleep(0.2)
        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if set_w and set_h and actual_width == REQUESTED_WIDTH and actual_height == REQUESTED_HEIGHT:
             print(f"Successfully set resolution to {actual_width}x{actual_height}.")
        else:
             print(f"Warning: Could not set requested resolution {REQUESTED_WIDTH}x{REQUESTED_HEIGHT} or it wasn't applied.")
             print(f"Actual camera resolution is: {actual_width}x{actual_height}")
        # ---

        # --- Set Exposure Mode ---
        if AUTO_EXPOSURE:
            print("Setting Auto Exposure (0.25)...")
            self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        else:
            print(f"Setting Manual Exposure (Value: {MANUAL_EXPOSURE_STOP})...")
            set_manual_mode = self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75) # Try non-standard first
            time.sleep(0.1)
            if not set_manual_mode:
                 print("Setting manual mode with 0.75 failed, trying 0...")
                 self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0) # Standard manual
                 time.sleep(0.1)
            self.camera.set(cv2.CAP_PROP_EXPOSURE, MANUAL_EXPOSURE_STOP)
        print(f"Current Auto Exposure setting reported: {self.camera.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")
        print(f"Current Exposure value reported: {self.camera.get(cv2.CAP_PROP_EXPOSURE)}")
        # ---

        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
        self.zoom_factor = zoom_factor

    def _apply_zoom(self, frame):
        """Applies center zoom to the frame."""
        if frame is None or frame.size == 0:
             print("Warning: Invalid frame passed to zoom.")
             return None
        if self.zoom_factor <= 1.0:
            return frame # Return original if no zoom needed

        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2

        new_w = int(w / self.zoom_factor)
        new_h = int(h / self.zoom_factor)

        x1 = max(0, center_x - new_w // 2)
        y1 = max(0, center_y - new_h // 2)
        x2 = min(w, x1 + new_w)
        y2 = min(h, y1 + new_h)

        # Ensure coordinates are valid before slicing
        if x1 >= x2 or y1 >= y2:
             print(f"Warning: Invalid zoom coordinates ({x1},{y1}) to ({x2},{y2}).")
             return None

        zoomed_frame = frame[y1:y2, x1:x2]

        if zoomed_frame.size == 0:
            print("Warning: Zoom resulted in empty frame.")
            return None
        return zoomed_frame



    def _find_largest_rectangle_contour(self, contours, frame_shape):
        """
        Finds the largest contour that approximates to 4 vertices and meets
        geometric criteria.

        Args:
            contours: A list of contours found by cv2.findContours.
            frame_shape: The shape (height, width) of the frame for aspect ratio calculation.

        Returns:
            The largest valid original contour, or None.
        """
        largest_rectangle_contour = None
        max_area = 0

        frame_height, frame_width = frame_shape[:2]

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_RECT_AREA: # Filter small contours early
                continue

            perimeter = cv2.arcLength(contour, True)
            # Approximate the contour shape
            approx = cv2.approxPolyDP(contour, APPROX_POLY_EPSILON_FACTOR * perimeter, True)

            # --- Validation Checks ---
            if len(approx) == 4: # Check if it approximates to a quadrilateral
                # 1. Convexity Check
                if not cv2.isContourConvex(approx):
                    continue # Skip non-convex shapes

                # 2. Bounding Box and Aspect Ratio Check
                x, y, w, h = cv2.boundingRect(approx) # Use approx for aspect ratio
                if w <= 0 or h <= 0:
                    continue
                # Calculate aspect ratio carefully to avoid division by zero
                aspect_ratio = float(w) / h if h > 0 else 0
                inv_aspect_ratio = float(h) / w if w > 0 else 0

                # Ensure aspect ratio is within reasonable bounds
                is_valid_aspect_ratio = (MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO) or \
                                        (MIN_ASPECT_RATIO < inv_aspect_ratio < MAX_ASPECT_RATIO)

                if not is_valid_aspect_ratio:
                     # print(f"Skipping contour due to aspect ratio: {aspect_ratio:.2f}") # Debugging
                     continue

                # 3. Area Check (using original contour for accuracy)
                if area > max_area:
                    # Optional: Add angle check logic here if needed for more robustness

                    max_area = area
                    largest_rectangle_contour = contour # Return the original contour

        return largest_rectangle_contour



    def _detect_rectangle_info(self, frame):
        """
        Detects the largest 4-sided polygon contour using Canny edge detection,
        morphological operations, and refined filtering.

        Args:
            frame (numpy.ndarray): The input frame.

        Returns:
            dict: {'bbox': (x,y,w,h), 'min_rect': ((cx,cy),(w,h),a)}
                  or None if not found.
        """
        if frame is None or frame.size == 0:
            print("Error: Input frame is empty.")
            return None

        # --- Preprocessing ---
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Increased blur slightly to help reduce noise before Canny
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            # Canny Edge Detection - Parameters might need tuning!
            edges = cv2.Canny(blurred, 50, 150)

            # --- Morphological Operations ---
            # Create a kernel for morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_SIZE)
            # Closing: Dilate then Erode - Fills small holes and gaps in the edges
            closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            # Optional: Opening (Erode then Dilate) can remove small noise specks
            # opened_edges = cv2.morphologyEx(closed_edges, cv2.MORPH_OPEN, kernel)
            # edges_processed = opened_edges # Use this if you add opening

            edges_processed = closed_edges # Use the result after closing

            # --- Debugging: Show intermediate steps ---
            # cv2.imshow("Gray", gray)
            # cv2.imshow("Blurred", blurred)
            # cv2.imshow("Edges", edges)
            # cv2.imshow("Closed Edges", edges_processed)
            # cv2.waitKey(1) # Add a small delay if displaying debug windows

        except cv2.error as e:
            print(f"OpenCV error during preprocessing: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error during preprocessing: {e}")
            return None


        # --- Contour Detection ---
        # Find contours on the morphologically processed edges
        contours, _ = cv2.findContours(edges_processed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            # print("No contours found.") # Debugging
            return None

        # --- Find Largest Valid Rectangle Contour ---
        largest_rect_contour = self._find_largest_rectangle_contour(contours, frame.shape)
        if largest_rect_contour is None:
             # print("No valid rectangle contour found.") # Debugging
             return None

        # --- Calculate Bounding Boxes ---
        # Use the *original* contour for final bounding boxes for precision
        bbox = cv2.boundingRect(largest_rect_contour) # (x, y, w, h) for drawing
        min_rect = cv2.minAreaRect(largest_rect_contour) # ((cx,cy),(w,h),a) for rotation

        return {'bbox': bbox, 'min_rect': min_rect}



    def _rotate_and_crop(self, frame, min_rect):
         """
         Rotates the frame based on the minAreaRect angle and crops the rectangle.

         Args:
             frame: The source image (usually the zoomed frame).
             min_rect: The tuple returned by cv2.minAreaRect ((cx,cy),(w,h),a).

         Returns:
             The upright, cropped rectangle image, or None on error.
         """
         if frame is None or min_rect is None:
              return None

         center, size, angle = min_rect
         width, height = size

         # Get rotation matrix
         M = cv2.getRotationMatrix2D(center, angle, 1.0)

         # Perform rotation
         try:
              warped = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_CUBIC)
              if warped is None: raise Exception("warpAffine returned None")
         except Exception as e:
              print(f"Error during warpAffine: {e}")
              return None

         # Crop the rotated rectangle
         crop_width = int(round(width))
         crop_height = int(round(height))
         if crop_width <= 0 or crop_height <= 0:
              print(f"Warning: Invalid crop dimensions from minAreaRect: {crop_width}x{crop_height}")
              return None

         try:
              cropped = cv2.getRectSubPix(warped, (crop_width, crop_height), center)
              if cropped is None: raise Exception("getRectSubPix returned None")
         except Exception as e:
              print(f"Error during getRectSubPix: {e}")
              return None

         # Rotate 90 degrees if height > width (common issue with minAreaRect angle)
         if crop_height > crop_width * 1.1: # Adjusted threshold slightly
              print("Rotating cropped image by 90 degrees due to aspect ratio.")
              cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)


         if cropped.size == 0:
              print("Warning: Rotation/cropping resulted in empty image.")
              return None

         return cropped



    def capture_raw_image(self):
        """Captures ONE raw image from the camera."""
        ret, frame = self.camera.read()
        if not ret or frame is None:
            print("Warning: Frame capture failed or returned None, retrying...")
            time.sleep(0.1)
            ret, frame = self.camera.read()
            if not ret or frame is None:
                 print("Error: Failed to capture image after retry")
                 return None
        return frame


    def inspect_tag(self):
        """
        Captures raw image, zooms, retries rectangle detection, rotates/crops,
        runs inference. Falls back to zoomed image if detection fails.

        Returns:
            dict: Result dictionary with 'image_data' being the rotated/cropped
                  rectangle, or the zoomed image on detection failure.
        """
        # 1. Capture Raw Image
        raw_image = self.capture_raw_image()
        if raw_image is None:
             return {"error": "Image capture failed", "image_data": None}

        # 2. Apply Zoom
        zoomed_image = self._apply_zoom(raw_image)
        if zoomed_image is None:
             return {"error": "Zoom processing failed", "image_data": None}

        # 3. Detect Rectangle Coordinates (with Retries)
        rect_info = None
        print(f"Attempting rectangle detection (max {RECT_DETECT_RETRIES} times)...")
        for attempt in range(RECT_DETECT_RETRIES):
             rect_info = self._detect_rectangle_info(zoomed_image) # Uses updated epsilon
             if rect_info:
                  print(f"Rectangle detected on attempt {attempt + 1}.")
                  break # Found it
        else: # Loop finished without break
             print(f"Rectangle detection failed after {RECT_DETECT_RETRIES} attempts.")

        # --- Process based on detection result ---
        final_image = None
        is_detection_fallback = False
        error_msg = None # Initialize error message

        if rect_info:
            # 4. Rotate and Crop if rectangle detected
            final_image = self._rotate_and_crop(zoomed_image, rect_info['min_rect'])
            if final_image is None:
                 print("Warning: Rotation/Cropping failed. Falling back to zoomed image.")
                 final_image = zoomed_image # Fallback to zoomed
                 is_detection_fallback = True
                 error_msg = "Rotation/Cropping failed"
            # else: error_msg remains None (Success)
        else:
            # 5. Fallback to zoomed image if detection failed after retries
            print("Using zoomed image due to detection failure.")
            final_image = zoomed_image
            is_detection_fallback = True
            error_msg = f"Rectangle detection failed after {RECT_DETECT_RETRIES} retries"


        # --- Inference ---
        classification_result = {}
        if final_image is not None:
            # Save final_image temporarily before inference
            timestamp_str = f"{int(time.time())}"
            temp_path = os.path.join(self.results_dir, f"temp_{timestamp_str}.jpg")
            try:
                 save_success = cv2.imwrite(temp_path, final_image)
                 if not save_success: temp_path = None
            except Exception: temp_path = None

            # Run inference
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
                      else: class_name = "Unknown"
                      classification_result = {"class": class_name, "confidence": confidence}

            except Exception as model_err:
                 print(f"Error during model inference: {model_err}")
                 classification_result = {"class": "Inference Error", "confidence": 0.0}
                 if error_msg is None: error_msg = f"Model inference failed: {model_err}"

        else: # Should not happen if zoomed_image fallback works, but as safety
             classification_result = {"class": "Processing Error", "confidence": 0.0}
             if error_msg is None: error_msg = "final_image was None before inference"
             temp_path = None


        # --- Construct Final Result ---
        result_data = {
            "class": classification_result.get("class", "Error"),
            "confidence": classification_result.get("confidence", 0.0),
            "timestamp": time.time(),
            "image_path": temp_path,
            "image_data": final_image # This is rotated/cropped or zoomed
        }
        if is_detection_fallback:
             # Overwrite class/confidence if detection failed, even if inference ran
             result_data["class"] = "Detection Failed"
        if error_msg:
             result_data["error"] = error_msg


        # --- Save JSON ---
        latest_result_path = os.path.join(self.results_dir, "latest_result.json")
        try:
            json_data = {k: v for k, v in result_data.items() if k != 'image_data'}
            with open(latest_result_path, "w") as f:
                json.dump(json_data, f, indent=4)
        except IOError as e:
            print(f"Error writing results to {latest_result_path}: {e}")

        return result_data


    def close(self):
        """Releases the camera resource."""
        if self.camera.isOpened():
            self.camera.release()
        print("Camera released.")


if __name__ == "__main__":
    # --- Check if model file exists ---
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        exit()
    # ---

    qc = None
    quit_flag = False
    live_window_name = "Live Feed (Zoomed + Box)"
    result_window_name = "Inspection Result (Cropped & Rotated)"
    result_window_created = False

    try:
        qc = NameTagQualityControl(MODEL_PATH, camera_id=CAMERA_ID, zoom_factor=ZOOM_FACTOR)

        print("\nStarting inspection loop.")
        print("Live feed shows zoomed view with detected rectangle. Press Enter to capture, crop, rotate and inspect.")
        print("Press 'q' in any window to quit.")
        if AUTO_EXPOSURE:
            print("Using Auto Exposure.")
        else:
            print(f"Using Manual Exposure setting: {MANUAL_EXPOSURE_STOP}")


        while not quit_flag: # Main loop controlled by flag

            # --- Live Video Feed ---
            ret_live, live_frame_raw = qc.camera.read()
            display_live_frame = None # Frame to actually display

            if ret_live and live_frame_raw is not None:
                try:
                    # 1. Apply zoom for the live feed display
                    zoomed_live_frame = qc._apply_zoom(live_frame_raw)

                    if zoomed_live_frame is not None:
                         # Make a copy to draw on
                         display_live_frame = zoomed_live_frame.copy()

                         # 2. Detect rectangle info (bbox needed for drawing)
                         live_rect_info = qc._detect_rectangle_info(zoomed_live_frame) # Uses updated epsilon

                         # 3. Draw bounding box if coordinates are found
                         if live_rect_info:
                              min_rect = live_rect_info['min_rect']
                              box = cv2.boxPoints(min_rect)  # Get the 4 corner points
                              box = np.intp(box)  # Convert points to integer type for drawing
                              cv2.drawContours(display_live_frame, [box], 0, (0, 255, 0), 2)

                         # Display the frame (zoomed, potentially with box)
                         cv2.imshow(live_window_name, display_live_frame)
                    else:
                         pass # Keep previous frame shown if zoom fails

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

            elif key == 13: # Enter key - Trigger inspection
                print("\nEnter pressed, inspecting tag (Capture -> Zoom -> Retry Detect -> Rotate/Crop)...")

                # 1. Inspect the tag (handles all steps internally including retries and fallback)
                result = qc.inspect_tag()

                # Get the image data to display (could be rotated/cropped or zoomed fallback)
                img_display = result.get("image_data")

                # Update window title based on result
                if "error" in result:
                     print(f"Inspection Error: {result['error']}")
                     window_title = f"Error: {result['error']}"
                elif result.get("class") == "Detection Failed":
                     print("Inspection finished: Detection Failed (showing zoomed image).")
                     window_title = "Result: Detection Failed"
                else:
                     print(f"Displaying result: {result['class']} (Confidence: {result['confidence']:.2f})")
                     window_title = f"Result: {result['class']} (Conf: {result['confidence']:.2f})"


                # Display the image (if any) and update title
                if img_display is not None and img_display.size > 0:
                     try:
                          cv2.imshow(result_window_name, img_display)
                          result_window_created = True
                          cv2.setWindowTitle(result_window_name, window_title)
                          print(f"-> '{result_window_name}' updated. Press Enter to capture next, 'q' to quit.")
                     except Exception as display_e:
                          print(f"Error displaying result image in '{result_window_name}': {display_e}")
                else:
                     # Handle case where even fallback image is missing
                     print("No image data available to display for result.")


            # --- End of Enter Key Handling ---

    except Exception as e:
        print(f"\nA critical error occurred: {e}")
        traceback.print_exc()
    finally:
        print("\nCleaning up...")
        if qc is not None:
            qc.close()
        cv2.destroyAllWindows()
        print("Program finished.")

