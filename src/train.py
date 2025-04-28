"""
SCU 2025 Senior Design
NAME IT! - Automated Work Cell - CV QC (Computer Vision Quality Control)
automated manufacturing line - DexArm and laser cut name tags

Classification
YOLOv11
YOLOv12 only has detection pretrained, and it does not work on my PC.
Build a new model has poor performance, likely due to small dataset.
new model: high ceiling, low floor
pretrained model: low ceiling, high floor
"""
from ultralytics import YOLO

# Load a model
#model = YOLO("yolo12n-cls.yaml")  # build a new model from YAML
#model = YOLO("F:/JetBrains/PycharmProjects/AWC_CV_QC/src/runs/classify/train2/weights/best.pt")
#model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)

# best to worst model: can not differentiate based on loss, need inference



# Train / fine tune the model
if __name__ == '__main__':
	# original fake resolution: ~3000 * ~1000
	# original real resolution: ~750 * ~250
	# optimized resolution (naked eye): 300 * 100
	# Laptop: C:/Users/fei18/PycharmProjects/AWC_CV_QC/dataset
	# Desktop: F:/JetBrains/PycharmProjects/AWC_CV_QC/dataset

	results = model.train(data="F:/JetBrains/PycharmProjects/AWC_CV_QC/dataset",
	                      pretrained=True,
	                      epochs=1000,
	                      patience=100,
		# training must use square image size, multiple of 32, will auto padding
		# inference can use other aspect ratio
						  imgsz=768,
	                      workers=7,          # CPU intensive, number of cores
	                      batch=32            # GPU VRAM / RAM intensive
	                      )