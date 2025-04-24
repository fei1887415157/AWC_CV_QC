import cv2
from ultralytics import YOLO
import time
import json
import os
import numpy as np # Import numpy for array operations
import traceback # For detailed error printing

# --- Global Configuration ---
# Define constants at the top level
MODEL_PATH = "runs/classify/train5/weights/best.pt" # Path to your trained .pt model
CAMERA_ID = 0 # Change if you have multiple cameras
REQUESTED_WIDTH = 1920 # Desired camera width
REQUESTED_HEIGHT = 1080 # Desired camera height
ZOOM_FACTOR = 2.0 # Set desired zoom factor (1.0 = no zoom, 2.0 = 2x zoom)
TARGET_ASPECT_RATIO = 22 / 9 # Define the target aspect ratio (22:9)
# ---

class NameTagQualityControl:
    def __init__(self, model_path, camera_id=0, zoom_factor=2.0):
        """
        Initializes the NameTagQualityControl class. Attempts to set camera
        resolution and adjust auto-exposure mode before proceeding.

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
        # It's good practice to wait briefly after setting properties
        time.sleep(0.2)
        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if set_w and set_h and actual_width == REQUESTED_WIDTH and actual_height == REQUESTED_HEIGHT:
             print(f"Successfully set resolution to {actual_width}x{actual_height}.")
        else:
             print(f"Warning: Could not set requested resolution {REQUESTED_WIDTH}x{REQUESTED_HEIGHT} or it wasn't applied.")
             print(f"Actual camera resolution is: {actual_width}x{actual_height}")
        # ---

        # --- Attempt to adjust Auto Exposure Mode ---
        # Direct metering control isn't standard. We try to influence behavior.
        # Mode 1 often corresponds to 'Aperture Priority' which might help.
        # Mode 0 usually means 'Manual Exposure'.
        # The default is often 3 ('Auto Exposure').
        print("Attempting to set Auto Exposure mode (e.g., Aperture Priority)...")
        # Value '1' = Auto mode, Value '0' = Manual mode
        # Setting to 1 might let camera adjust gain/shutter based on aperture, potentially better for varying light
        set_auto_exposure = self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        time.sleep(0.2) # Allow time for setting to apply
        current_auto_exposure = self.camera.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        if set_auto_exposure:
            print(f"Successfully requested Auto Exposure mode change. Current mode value: {current_auto_exposure}")
            # Note: Getting the value back might not always reflect the mode accurately depending on driver.
        else:
            print(f"Warning: Could not set Auto Exposure mode. Current mode value: {current_auto_exposure}")
            print("  Consider trying a different camera backend (e.g., cv2.CAP_DSHOW on Windows).")

        # --- If setting Auto Exposure to 1 didn't work, you might need Manual (0) ---
        # if current_auto_exposure != 1: # Or based on visual result
        #     print("Attempting to set Manual Exposure mode...")
        #     set_manual_exposure = self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        #     if set_manual_exposure:
        #         print("Set to Manual Exposure mode. Now attempting to set exposure value.")
        #         # Setting exposure requires experimentation! Range varies by camera.
        #         # Negative values typically reduce exposure. Start around -5 or -6.
        #         exposure_value = -6
        #         set_exp = self.camera.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
        #         time.sleep(0.2)
        #         current_exposure = self.camera.get(cv2.CAP_PROP_EXPOSURE)
        #         if set_exp:
        #             print(f"Successfully set Exposure value to {exposure_value}. Current value: {current_exposure}")
        #         else:
        #             print(f"Warning: Could not set Exposure value. Current value: {current_exposure}")
        #     else:
        #         print("Warning: Could not set Manual Exposure mode.")
        # ---

        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
        self.zoom_factor = zoom_factor
        self.target_aspect_ratio = TARGET_ASPECT_RATIO # Store aspect ratio

    def capture_image(self):
        """
        Captures an image at the set resolution, crops it to the target aspect
        ratio (22:9) from the center, and applies digital zoom to that cropped
        region. The final output preserves the aspect ratio of the zoomed region.

        Returns:
            numpy.ndarray: The processed image frame (cropped and zoomed, undistorted).

        Raises:
            Exception: If capturing the image fails.
        """
        ret, frame = self.camera.read()
        if not ret or frame is None: # Added check for None frame
            # Attempt to re-read once if frame is None
            print("Warning: Initial frame capture failed or returned None, retrying...")
            time.sleep(0.1) # Short delay before retry
            ret, frame = self.camera.read()
            if not ret or frame is None:
                 raise Exception("Failed to capture image after retry")

        # Use the actual height and width obtained from the camera
        h, w = frame.shape[:2]
        if h == 0 or w == 0:
             raise Exception(f"Captured frame has invalid dimensions: {w}x{h}")

        original_aspect_ratio = w / h

        # --- Aspect Ratio Cropping (22:9) ---
        # Determine the dimensions of the largest 22:9 rectangle that fits within the actual frame
        if original_aspect_ratio > self.target_aspect_ratio:
            # Frame is wider than 22:9, crop width
            crop_h = h
            crop_w = int(crop_h * self.target_aspect_ratio)
        else:
            # Frame is taller than or equal to 22:9, crop height
            crop_w = w
            crop_h = int(crop_w / self.target_aspect_ratio)

        # Ensure crop dimensions are valid
        if crop_w <= 0 or crop_h <= 0:
             raise Exception(f"Calculated invalid crop dimensions: {crop_w}x{crop_h} from frame {w}x{h}")


        # Calculate top-left corner for centered crop
        x1 = max(0, (w - crop_w) // 2)
        y1 = max(0, (h - crop_h) // 2)
        # Calculate bottom-right corner
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        # Crop the frame to the target 22:9 aspect ratio
        aspect_cropped_frame = frame[y1:y2, x1:x2]
        # --- End Aspect Ratio Cropping ---

        # --- Check if cropping resulted in a valid image ---
        if aspect_cropped_frame.size == 0:
             raise Exception(f"Aspect ratio cropping resulted in an empty image. Original: {w}x{h}, Crop: {crop_w}x{crop_h}, Coords: ({x1},{y1}) to ({x2},{y2})")
        # ---

        # --- Zoom Logic (Applied to the 22:9 cropped image) ---
        if self.zoom_factor > 1.0:
            ch, cw = aspect_cropped_frame.shape[:2] # Cropped height, width
            if ch == 0 or cw == 0:
                 raise Exception(f"Aspect cropped frame has invalid dimensions before zoom: {cw}x{ch}")

            # Calculate the center of the cropped image
            center_x, center_y = cw // 2, ch // 2

            # Calculate the dimensions of the zoomed-in region within the cropped image
            # These dimensions will maintain the 22:9 aspect ratio
            zoom_w = int(cw / self.zoom_factor)
            zoom_h = int(ch / self.zoom_factor)

            # Ensure zoom dimensions are valid
            if zoom_w <= 0 or zoom_h <= 0:
                 raise Exception(f"Calculated invalid zoom dimensions: {zoom_w}x{zoom_h} from cropped frame {cw}x{ch}")

            # Calculate the top-left corner of the zoom crop rectangle
            zx1 = max(0, center_x - zoom_w // 2)
            zy1 = max(0, center_y - zoom_h // 2)

            # Calculate the bottom-right corner (ensure it doesn't exceed bounds of cropped image)
            zx2 = min(cw, zx1 + zoom_w)
            zy2 = min(ch, zy1 + zoom_h)

            # Crop the central region for zooming
            final_frame = aspect_cropped_frame[zy1:zy2, zx1:zx2]

            # --- Check if zoom cropping resulted in a valid image ---
            if final_frame.size == 0:
                 raise Exception(f"Zoom cropping resulted in an empty image. Cropped: {cw}x{ch}, Zoom Crop: {zoom_w}x{zoom_h}, Coords: ({zx1},{zy1}) to ({zx2},{zy2})")
            # ---

            # Return the zoomed region directly.
            return final_frame
        else:
            # If no zoom is applied, return the 22:9 cropped frame directly
            return aspect_cropped_frame
        # --- End Zoom Logic ---


    def inspect_tag(self):
        """
        Captures an image, runs inference, and saves the results.

        Returns:
            dict: A dictionary containing the classification result,
                  confidence, timestamp, and image path.
        """
        try:
            # Capture image (high-res, cropped to 22:9 and potentially zoomed, undistorted)
            image = self.capture_image() # Now returns the correctly sized image
        except Exception as capture_err:
             print(f"Error during image capture/processing: {capture_err}")
             # Return error dictionary
             return {
                 "class": "Error",
                 "confidence": 0.0,
                 "timestamp": time.time(),
                 "image_path": None, # No image saved
                 "error": f"Image capture failed: {capture_err}"
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
            with open(latest_result_path, "w") as f:
                json.dump(result_data, f, indent=4) # Add indent for readability
        except IOError as e:
            print(f"Error writing results to {latest_result_path}: {e}")


        # Add the actual image data to the dictionary returned by the function
        # This avoids reading it again later if saving failed or is slow
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
    try:
        # Use the global ZOOM_FACTOR constant during instantiation
        qc = NameTagQualityControl(MODEL_PATH, camera_id=CAMERA_ID, zoom_factor=ZOOM_FACTOR)

        print("\nStarting inspection loop. Press Enter after viewing an image to capture the next one.")
        print("Press 'q' to quit.")

        while not quit_flag: # Main loop controlled by flag
            # 1. Inspect the tag (captures, processes, saves, infers)
            print("\nInspecting tag...") # Add newline for clarity
            result = qc.inspect_tag() # result now contains 'image_data'

            # 2. Handle potential errors from inspection
            window_name = None # Reset window name each iteration
            if "error" in result:
                 print(f"Inspection Error: {result['error']}")
                 if result.get('image_path'):
                      print(f"  (Image attempted save at: {result['image_path']})")
                 print("Error occurred. Press Enter to retry, 'q' to quit...")
                 # Wait for key press after error
                 while True:
                     key = cv2.waitKey(1) & 0xFF # Use mask for compatibility
                     if key == 13: # Enter
                         break # Retry outer loop
                     elif key == ord('q'):
                         quit_flag = True
                         break # Quit outer loop
                     time.sleep(0.01)
                 if quit_flag: break # Exit outer loop if q pressed
                 continue # Skip display if error, retry capture

            # 3. Display the image if available in result
            img_display = result.get("image_data")
            if img_display is not None and img_display.size > 0:
                 print(f"Displaying result: {result['class']} (Confidence: {result['confidence']:.2f})")
                 window_name = f"Result: {result['class']} (Conf: {result['confidence']:.2f})"
                 try:
                    cv2.imshow(window_name, img_display)
                 except Exception as display_e:
                     print(f"Error displaying image: {display_e}")
                     cv2.destroyAllWindows() # Close any lingering windows
                     window_name = None # Flag that window wasn't shown
            else:
                # This case should be less likely now image_data is included
                print(f"Result: {result['class']} (Confidence: {result['confidence']:.2f}) - No image data available to display.")
                window_name = None # Flag that window wasn't shown

            # 4. Wait for Enter key (13) or 'q' (113)
            if window_name:
                print("-> Window focused. Press Enter to capture next, 'q' to quit.")
            else:
                print("-> No window. Press Enter to capture next, 'q' to quit.")

            while True:
                key = cv2.waitKey(1) & 0xFF # Check for key press, use mask
                if key == 13: # Enter key
                    print("Enter pressed, capturing next image...")
                    if window_name:
                        try:
                            cv2.destroyWindow(window_name)
                            # Add a tiny delay to ensure window closes before next capture potentially opens one
                            cv2.waitKey(1)
                        except cv2.error: # Handle case where window might already be closed
                            pass
                    break # Exit inner wait loop, continue outer loop for next capture
                elif key == ord('q'):
                    print("Quit key pressed.")
                    quit_flag = True # Set flag to break outer loop
                    if window_name:
                         try:
                            cv2.destroyWindow(window_name)
                            cv2.waitKey(1)
                         except cv2.error:
                            pass
                    break # Exit inner wait loop
                # Add a small sleep to prevent high CPU usage in the wait loop
                time.sleep(0.01)

            if quit_flag: # Check flag after inner loop breaks
                break # Break outer loop

    except Exception as e:
        # Catch critical errors during setup or loop
        print(f"\nA critical error occurred: {e}")
        traceback.print_exc() # Print detailed traceback for debugging
    finally:
        # Ensure cleanup happens
        print("\nCleaning up...")
        if qc is not None: # Check if qc was successfully initialized
            qc.close()
        # Close any remaining OpenCV windows
        cv2.destroyAllWindows()
        print("Program finished.")

