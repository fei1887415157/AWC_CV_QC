import cv2
from ultralytics import YOLO
import time
import json
import os



class NameTagQualityControl:
	def __init__(self, model_path, camera_id=0):
		self.model = YOLO(model_path)
		self.camera = cv2.VideoCapture(camera_id)
		self.results_dir = "results"
		os.makedirs(self.results_dir, exist_ok=True)

	def capture_image(self):
		ret, frame = self.camera.read()
		if not ret:
			raise Exception("Failed to capture image")
		return frame

	def inspect_tag(self):
		# Capture image
		image = self.capture_image()

		# Save image temporarily
		temp_path = os.path.join(self.results_dir, f"temp_{int(time.time())}.jpg")
		cv2.imwrite(temp_path, image)

		# Run inference
		results = self.model(temp_path)

		# Extract results
		predicted_class = results[0].probs.top1
		confidence = float(results[0].probs.top1conf)
		class_name = results[0].names[predicted_class]

		# Create result dict
		result = {
			"class": class_name,
			"confidence": confidence,
			"timestamp": time.time(),
			"image_path": temp_path
		}

		# Save result to file (for integration)
		with open(os.path.join(self.results_dir, "latest_result.json"), "w") as f:
			json.dump(result, f)

		return result

	def close(self):
		self.camera.release()


# Example usage
if __name__ == "__main__":
	qc = NameTagQualityControl("runs/classify/name_tag_quality/weights/best.pt")

	while True:
		input("Press Enter to inspect a name tag...")
		result = qc.inspect_tag()
		print(f"Quality: {result['class']} (Confidence: {result['confidence']:.2f})")