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

Notes
Ultralytics uses PyTorch, use below command to install if using PyTorch+CUDA:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

This model needs good lighting.
"""



from ultralytics import YOLO



# Load a model
#model = YOLO("yolo12n-cls.yaml")  # build a new model from YAML
#model = YOLO("F:/JetBrains/PycharmProjects/AWC_CV_QC/src/runs/classify/train2/weights/best.pt")

# Use small model since dataset is small
model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)

# best to worst model: can not differentiate based on loss, need inference



# Train / Fine-Tune the model
if __name__ == '__main__':
	# original fake resolution: ~3000 * ~1000
	# original real resolution: ~750 * ~250
	# optimized resolution (naked eye): 300 * 100
	# Laptop:  C:/Users/fei18/PycharmProjects/AWC_CV_QC/dataset
	# Desktop: F:/JetBrains/PycharmProjects/AWC_CV_QC/dataset

	results = model.train(data="../dataset",
	                      pretrained=True,
	                      epochs=200,
	                      patience=100,
		# training must use square image size, stride of 32
	    # rect=True enables auto padding
		# inference can use other aspect ratio
						  imgsz=768,
	                      rect=True,
	                      cos_lr=True,        # Cosine Annealing, learning rate schedule
	                      weight_decay=0.001,   # penalty for large weights, less overfit
						  workers=7,          # CPU intensive, number means number of cores
	                      batch=32            # GPU VRAM / RAM intensive
	                      )