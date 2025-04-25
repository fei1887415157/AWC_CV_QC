import cv2
from ultralytics import YOLO
import time
import json
import os
import traceback # For detailed error printing



# --- Global Configuration ---
MODEL_PATH = "runs/classify/train9/weights/best.pt" # Path to your trained .pt model
CAMERA_ID = 0 # Change if you have multiple cameras
REQUESTED_WIDTH = 1920 # Desired camera width
REQUESTED_HEIGHT = 1080 # Desired camera height
ZOOM_FACTOR = 2.0 # Set desired zoom factor (1.0 = no zoom, 2.0 = 2x zoom)
TARGET_ASPECT_RATIO = 22/9 # Define the target aspect ratio (22:9)
AUTO_EXPOSURE = True
MANUAL_EXPOSURE_STOP = -5
# ---



class NameTagQualityControl:
    def __init__(self, model_path, camera_id=0, zoom_factor=2.0):
        """
        Initializes the NameTagQualityControl class. Attempts to set camera
        resolution and enables auto-exposure mode (using 0.25 as requested).

        Args:
            model_path (str): Path to the trained YOLO classification model.
            camera_id (int): ID of the camera to use (default is 0).
            zoom_factor (float): The desired zoom factor (e.g., 2.0 for 2x zoom).
        """
        self.model = YOLO(model_path)
        # Try different backends if default doesn't allow setting parameters
        # self.camera = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW) # Example for Windows DirectShow
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

        # --- Set Exposure Mode
        if AUTO_EXPOSURE:
            self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        else:
            self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            self.camera.set(cv2.CAP_PROP_EXPOSURE, MANUAL_EXPOSURE_STOP)

        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
        self.zoom_factor = zoom_factor
        self.target_aspect_ratio = TARGET_ASPECT_RATIO # Store aspect ratio

    def _process_frame(self, frame):
        """
        Applies aspect ratio cropping (22:9) and zooming to a given frame.

        Args:
            frame (numpy.ndarray): The input frame from the camera.

        Returns:
            numpy.ndarray: The processed (cropped and zoomed) frame.

        Raises:
            Exception: If cropping or zooming results in an invalid frame.
        """
        if frame is None or frame.size == 0:
             # Return None or raise error if frame is invalid before processing
             # Returning None might be safer for the live loop
             print("Warning: Invalid frame received in _process_frame.")
             return None
             # raise ValueError("Input frame to _process_frame is invalid.")


        h, w = frame.shape[:2]
        if h == 0 or w == 0:
             print(f"Warning: Input frame has invalid dimensions: {w}x{h}")
             return None # Return None if dimensions are invalid
             # raise ValueError(f"Input frame has invalid dimensions: {w}x{h}")


        original_aspect_ratio = w / h

        # --- Aspect Ratio Cropping (22:9) ---
        try:
            if original_aspect_ratio > self.target_aspect_ratio:
                crop_h = h
                crop_w = int(crop_h * self.target_aspect_ratio)
            else:
                crop_w = w
                crop_h = int(crop_w / self.target_aspect_ratio)

            if crop_w <= 0 or crop_h <= 0:
                 raise ValueError(f"Calculated invalid crop dimensions: {crop_w}x{crop_h} from frame {w}x{h}")

            x1 = max(0, (w - crop_w) // 2)
            y1 = max(0, (h - crop_h) // 2)
            x2 = x1 + crop_w
            y2 = y1 + crop_h

            aspect_cropped_frame = frame[y1:y2, x1:x2]

            if aspect_cropped_frame.size == 0:
                 raise ValueError(f"Aspect ratio cropping resulted in an empty image. Original: {w}x{h}, Crop: {crop_w}x{crop_h}, Coords: ({x1},{y1}) to ({x2},{y2})")
        except Exception as crop_err:
             print(f"Error during aspect ratio cropping: {crop_err}")
             return None # Return None on cropping error
        # --- End Aspect Ratio Cropping ---

        # --- Zoom Logic (Applied to the 22:9 cropped image) ---
        try:
            if self.zoom_factor > 1.0:
                ch, cw = aspect_cropped_frame.shape[:2]
                if ch == 0 or cw == 0:
                     raise ValueError(f"Aspect cropped frame has invalid dimensions before zoom: {cw}x{ch}")

                center_x, center_y = cw // 2, ch // 2
                zoom_w = int(cw / self.zoom_factor)
                zoom_h = int(ch / self.zoom_factor)

                if zoom_w <= 0 or zoom_h <= 0:
                     raise ValueError(f"Calculated invalid zoom dimensions: {zoom_w}x{zoom_h} from cropped frame {cw}x{ch}")

                zx1 = max(0, center_x - zoom_w // 2)
                zy1 = max(0, center_y - zoom_h // 2)
                zx2 = min(cw, zx1 + zoom_w)
                zy2 = min(ch, zy1 + zoom_h)

                final_frame = aspect_cropped_frame[zy1:zy2, zx1:zx2]

                if final_frame.size == 0:
                     raise ValueError(f"Zoom cropping resulted in an empty image. Cropped: {cw}x{ch}, Zoom Crop: {zoom_w}x{zoom_h}, Coords: ({zx1},{zy1}) to ({zx2},{zy2})")

                return final_frame
            else:
                # If no zoom is applied, return the 22:9 cropped frame directly
                return aspect_cropped_frame
        except Exception as zoom_err:
             print(f"Error during zoom processing: {zoom_err}")
             return None # Return None on zoom error
        # --- End Zoom Logic ---

    def capture_image(self):
        """
        Captures ONE image from the camera and processes it (crop/zoom).

        Returns:
            numpy.ndarray: The processed image frame, or None if capture/processing fails.

        Raises:
            Exception: Only if retrying capture fails definitively.
        """
        ret, frame = self.camera.read()
        if not ret or frame is None:
            print("Warning: Frame capture for processing failed or returned None, retrying...")
            time.sleep(0.1)
            ret, frame = self.camera.read()
            if not ret or frame is None:
                 # Raise exception only if retry fails completely
                 raise Exception("Failed to capture image for processing after retry")


        # Process the captured frame using the helper method
        processed_frame = self._process_frame(frame)
        if processed_frame is None:
             print("Warning: Frame processing returned None.")
             # Decide whether to raise an error or return None
             # Returning None might allow the main loop to continue more gracefully
             return None
        return processed_frame


    def inspect_tag(self):
        """
        Captures ONE image specifically for processing/inference, runs inference,
        and saves the results. Does NOT handle the live view display.

        Returns:
            dict: A dictionary containing the classification result,
                  confidence, timestamp, image path, and image data FOR THE CAPTURED FRAME.
                  Returns an error dict if capture/processing/inference fails.
        """
        try:
            # Capture and process image specifically for this inspection
            image = self.capture_image() # Gets the processed (cropped/zoomed) frame
            if image is None: # Handle case where capture_image returned None
                 raise Exception("capture_image returned None, cannot inspect.")
        except Exception as capture_err:
             print(f"Error during image capture/processing for inspection: {capture_err}")
             # Return error dictionary
             return {
                 "class": "Error",
                 "confidence": 0.0,
                 "timestamp": time.time(),
                 "image_path": None, # No image saved
                 "image_data": None, # No image data
                 "error": f"Image capture for inspection failed: {capture_err}"
             }

        # Save image temporarily before inference
        timestamp_str = f"{int(time.time())}" # Using only seconds for timestamp
        temp_path = os.path.join(self.results_dir, f"temp_{timestamp_str}.jpg")
        try:
             # Save the potentially smaller, correctly aspected image
             save_success = cv2.imwrite(temp_path, image)
             if not save_success:
                  print(f"Warning: Failed to save temporary image to {temp_path}")
                  temp_path = None # Indicate image wasn't saved
        except Exception as save_err:
             print(f"Error saving temporary image: {save_err}")
             temp_path = None # Indicate image wasn't saved


        # Run inference only if image was successfully captured
        # Note: We proceed even if saving failed, but image_path will be None
        try:
             # Ensure your model can handle the input size resulting from the crop/zoom
             # Use the image in memory directly if saving failed, otherwise use path
             source_for_model = image if temp_path is None else temp_path
             results = self.model(source=source_for_model) # Pass image or path
        except Exception as model_err:
             print(f"Error during model inference: {model_err}")
             return {
                 "class": "Error",
                 "confidence": 0.0,
                 "timestamp": time.time(),
                 "image_path": temp_path, # Image might exist even if model failed
                 "image_data": image, # Still return image data if inference fails
                 "error": f"Model inference failed: {model_err}"
             }

        # --- Check if results are valid ---
        if not results or not hasattr(results[0], 'probs') or results[0].probs is None:
             print("Warning: Model did not return valid results.")
             return {
                 "class": "Error",
                 "confidence": 0.0,
                 "timestamp": time.time(),
                 "image_path": temp_path,
                 "image_data": image, # Still return image data
                 "error": "Inference failed or returned no probabilities"
             }
        # ---

        # Extract results for classification
        predicted_class_index = results[0].probs.top1
        confidence = float(results[0].probs.top1conf)
        # Ensure the class index is within the bounds of the names list
        if results[0].names and predicted_class_index < len(results[0].names):
             class_name = results[0].names[predicted_class_index]
        else:
             print(f"Warning: Predicted class index {predicted_class_index} out of bounds or names list empty.")
             class_name = "Unknown" # Assign a default name


        # Create result dict
        result_data = {
            "class": class_name,
            "confidence": confidence,
            "timestamp": time.time(),
            "image_path": temp_path # Store the path (or None if save failed)
        }

        # Save result JSON only if inference was successful
        latest_result_path = os.path.join(self.results_dir, "latest_result.json")
        try:
            # Create a copy for JSON to avoid serializing the large image data
            json_data = result_data.copy()
            with open(latest_result_path, "w") as f:
                json.dump(json_data, f, indent=4) # Add indent for readability
        except IOError as e:
            print(f"Error writing results to {latest_result_path}: {e}")


        # Add the actual image data (the processed one) to the dictionary returned by the function
        result_data["image_data"] = image
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

    qc = None # Initialize qc to None for finally block safety
    quit_flag = False
    live_window_name = "Live Feed (Processed)" # Updated window name
    # --- Use a fixed name for the result window ---
    result_window_name = "Inspection Result"
    result_window_created = False # Flag to track if window exists
    # ---

    try:
        # Use the global ZOOM_FACTOR constant during instantiation
        qc = NameTagQualityControl(MODEL_PATH, camera_id=CAMERA_ID, zoom_factor=ZOOM_FACTOR)

        print("\nStarting inspection loop.")
        print("Press Enter to capture/inspect the current view and update the 'Inspection Result' window.")
        print("Press 'q' in any window to quit.")
        print(f"Using Auto Exposure setting: 0.25 (Behavior depends on driver interpretation)")


        while not quit_flag: # Main loop controlled by flag

            # --- Live Video Feed ---
            ret_live, live_frame_raw = qc.camera.read() # Read the raw frame
            processed_live_frame = None # Initialize

            if ret_live and live_frame_raw is not None:
                try:
                    # Process the raw frame using the same logic as capture_image
                    processed_live_frame = qc._process_frame(live_frame_raw)
                    if processed_live_frame is not None:
                         # Display the PROCESSED frame in the live view
                         cv2.imshow(live_window_name, processed_live_frame)
                    else:
                         # Optionally show raw frame if processing failed
                         # cv2.imshow(live_window_name, live_frame_raw)
                         pass # Or just don't update the window
                except Exception as live_process_err:
                    print(f"Error processing live frame: {live_process_err}")
                    # Optionally display the raw frame on error or just skip display
                    # cv2.imshow(live_window_name, live_frame_raw) # Fallback to raw
            else:
                print("Warning: Failed to grab frame for live feed.")
                # Optional: break or continue based on whether live feed is critical
                # break
            # ---

            # --- Handle Key Press for Quit or Capture ---
            # Check for key press. waitKey is essential for imshow to work.
            key = cv2.waitKey(1) & 0xFF # Check frequently for responsiveness

            if key == ord('q'):
                print("Quit key pressed.")
                quit_flag = True
                break # Exit the main loop immediately

            elif key == 13: # Enter key - Trigger inspection
                print("\nEnter pressed, inspecting tag...")

                # --- Removed destroyWindow call ---

                # 1. Inspect the tag (captures, processes, saves, infers)
                # This already uses the processed frame internally via capture_image -> _process_frame
                result = qc.inspect_tag()

                # 2. Handle potential errors from inspection
                if "error" in result:
                     print(f"Inspection Error: {result['error']}")
                     if result.get('image_path'):
                          print(f"  (Image attempted save at: {result['image_path']})")
                     print("Error occurred during inspection. Press Enter to retry, 'q' to quit...")
                     # Do not update result window on error
                else:
                    # 3. Display the processed/inspected image if available in result
                    # This is the same image data obtained from qc.inspect_tag()
                    img_display = result.get("image_data")
                    if img_display is not None and img_display.size > 0:
                         print(f"Displaying result: {result['class']} (Confidence: {result['confidence']:.2f})")
                         # --- Use the fixed result_window_name ---
                         try:
                            # This will create the window if it doesn't exist,
                            # or update it if it does.
                            cv2.imshow(result_window_name, img_display)
                            result_window_created = True # Mark that the window exists
                            # Update window title dynamically (optional, but good UX)
                            cv2.setWindowTitle(result_window_name, f"Result: {result['class']} (Conf: {result['confidence']:.2f})")
                            print(f"-> '{result_window_name}' updated. Press Enter to capture next, 'q' to quit.")
                         except Exception as display_e:
                             print(f"Error displaying result image in '{result_window_name}': {display_e}")
                             # Consider closing the specific window if display fails repeatedly
                             # if result_window_created:
                             #    try: cv2.destroyWindow(result_window_name)
                             #    except: pass
                             #    result_window_created = False

                    else:
                        print(f"Result: {result['class']} (Confidence: {result['confidence']:.2f}) - No processed image data available to display.")
                        # Optionally clear the result window if no data?
                        # if result_window_created:
                        #    # Create a blank image or similar to clear
                        #    pass


            # --- End of Enter Key Handling ---

            # Small sleep if no key was pressed, prevents high CPU in some cases,
            # but waitKey(1) already includes a small delay.
            # time.sleep(0.01)

    except Exception as e:
        # Catch critical errors during setup or loop
        print(f"\nA critical error occurred: {e}")
        traceback.print_exc() # Print detailed traceback for debugging
    finally:
        # Ensure cleanup happens
        print("\nCleaning up...")
        if qc is not None: # Check if qc was successfully initialized
            qc.close()
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        print("Program finished.")

