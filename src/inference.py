import cv2
from ultralytics import YOLO
import time
import json
import os
import numpy as np # Import numpy for array operations

# --- Global Configuration ---
# Define constants at the top level
MODEL_PATH = "runs/classify/train5/weights/best.pt" # Path to your trained .pt model
CAMERA_ID = 0 # Change if you have multiple cameras
REQUESTED_WIDTH = 1920 # Desired camera width
REQUESTED_HEIGHT = 1080 # Desired camera height
ZOOM_FACTOR = 2   # Set desired zoom factor (1.0 = no zoom, 2.0 = 2x zoom)
TARGET_ASPECT_RATIO = 22 / 9 # Define the target aspect ratio (22:9)
# ---

class NameTagQualityControl:
    def __init__(self, model_path, camera_id=0, zoom_factor=2.0):
        """
        Initializes the NameTagQualityControl class. Attempts to set camera
        resolution before proceeding.

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
        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if set_w and set_h:
             print(f"Successfully requested {REQUESTED_WIDTH}x{REQUESTED_HEIGHT}.")
        else:
             print(f"Warning: Could not set requested resolution {REQUESTED_WIDTH}x{REQUESTED_HEIGHT}.")
        print(f"Actual camera resolution set to: {actual_width}x{actual_height}")
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
             return {
                 "class": "Error",
                 "confidence": 0.0,
                 "timestamp": time.time(),
                 "image_path": None, # No image saved
                 "error": f"Image capture failed: {capture_err}"
             }

        # --- Display the processed image (Optional) ---
        # cv2.imshow("Processed Image (1080p -> 22:9 Crop + Zoom)", image)
        # cv2.waitKey(1) # Add a small delay for the window to update
        # ---

        # Save image temporarily
        timestamp_str = f"{int(time.time())}"
        temp_path = os.path.join(self.results_dir, f"temp_{timestamp_str}.jpg")
        try:
             # Save the potentially smaller, correctly aspected image
             save_success = cv2.imwrite(temp_path, image)
             if not save_success:
                  print(f"Warning: Failed to save temporary image to {temp_path}")
                  # Decide how to handle this - maybe return error or proceed without image path
                  temp_path = None # Indicate image wasn't saved
        except Exception as save_err:
             print(f"Error saving temporary image: {save_err}")
             temp_path = None # Indicate image wasn't saved


        # Run inference only if image was successfully captured and saved (or if proceeding without save)
        if temp_path is not None: # Or adjust logic if you want to infer even if save failed
            try:
                 # Ensure your model can handle the input size resulting from the crop/zoom
                 results = self.model(source=temp_path) # Pass image path as source
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
            if predicted_class_index < len(results[0].names):
                 class_name = results[0].names[predicted_class_index]
            else:
                 print(f"Warning: Predicted class index {predicted_class_index} out of bounds for names list (length {len(results[0].names)}).")
                 class_name = "Unknown" # Assign a default name

        else: # Handle case where image saving failed earlier
             print("Skipping model inference because temporary image could not be saved.")
             class_name = "Error"
             confidence = 0.0
             result = {
                 "class": class_name,
                 "confidence": confidence,
                 "timestamp": time.time(),
                 "image_path": temp_path, # Will be None
                 "error": "Image saving failed prior to inference"
             }
             # Save error result to JSON
             latest_result_path = os.path.join(self.results_dir, "latest_result.json")
             try:
                 with open(latest_result_path, "w") as f:
                     json.dump(result, f, indent=4)
             except IOError as e:
                 print(f"Error writing error results to {latest_result_path}: {e}")
             return result


        # Create result dict
        result = {
            "class": class_name,
            "confidence": confidence,
            "timestamp": time.time(),
            "image_path": temp_path
        }

        # Save result to file (for integration)
        latest_result_path = os.path.join(self.results_dir, "latest_result.json")
        try:
            with open(latest_result_path, "w") as f:
                json.dump(result, f, indent=4) # Add indent for readability
        except IOError as e:
            print(f"Error writing results to {latest_result_path}: {e}")


        return result

    def close(self):
        """Releases the camera resource."""
        if self.camera.isOpened():
            self.camera.release()
        # --- Close any OpenCV windows if they were opened ---
        # cv2.destroyAllWindows() # Moved to finally block in main
        # ---
        print("Camera released.")


if __name__ == "__main__":
    # --- Check if model file exists ---
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        exit()
    # ---

    qc = None # Initialize qc to None for finally block safety
    try:
        # Use the global ZOOM_FACTOR constant during instantiation
        qc = NameTagQualityControl(MODEL_PATH, camera_id=CAMERA_ID, zoom_factor=ZOOM_FACTOR)

        while True:
            key = input("Press Enter to inspect a name tag (or 'q' to quit)... ")
            if key.lower() == 'q':
                break
            try:
                result = qc.inspect_tag()
                if "error" in result:
                     print(f"Inspection Error: {result['error']}")
                     if result['image_path']:
                          print(f"  (Image saved at: {result['image_path']})")
                else:
                     # Display the result along with the image path
                     print(f"Quality: {result['class']} (Confidence: {result['confidence']:.2f}) - Image: {result['image_path']}")

                     #--- Optionally display the saved image ---
                     if result['image_path']: # Check if image path exists
                         try:
                             img_display = cv2.imread(result['image_path'])
                             if img_display is not None:
                                 # Resize for display if too large? Optional.
                                 # display_h, display_w = img_display.shape[:2]
                                 # scale = min(800/display_w, 600/display_h) # Example scaling
                                 # if scale < 1:
                                 #    img_display = cv2.resize(img_display, (int(display_w*scale), int(display_h*scale)))

                                 cv2.imshow(f"Result: {result['class']}", img_display)
                                 cv2.waitKey(0) # Wait until a key is pressed
                                 cv2.destroyWindow(f"Result: {result['class']}")
                             else:
                                 print(f"Warning: Could not read image {result['image_path']} for display.")
                         except Exception as display_e:
                              print(f"Error displaying image: {display_e}")


            except Exception as e:
                 # Catch errors from inspect_tag if not handled internally
                 print(f"An unexpected error occurred during inspection loop: {e}")
                 # Optionally break the loop or attempt recovery
                 # break

    except Exception as e:
        print(f"An error occurred during initialization: {e}")
    finally:
        # Ensure the camera is released even if errors occur
        if qc is not None: # Check if qc was successfully initialized
            qc.close()
        # Close any remaining OpenCV windows
        cv2.destroyAllWindows()
        print("Program finished.")

