from ultralytics import YOLO

# Classification
# YOLO 12
# use only yolo12n-cls.yaml, yolo12 has no pretrained model

# Load a model
model = YOLO("yolo12n-cls.yaml")  # build a new model from YAML

#model = YOLO("F:/JetBrains/PycharmProjects/AWC_CV_QC/src/runs/classify/train2/weights/best.pt")
#model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)
#model = YOLO("yolo11n-cls.yaml").load("yolo11n-cls.pt")  # build from YAML and transfer weights



# Train the model
if __name__ == '__main__':
	# original fake resolution: 4000 * 1638
	# original real resolution: 1000 * 410
	# optimized resolution (naked eye): 500 * 205
	results = model.train(data = "F:/JetBrains/PycharmProjects/AWC_CV_QC/dataset",
	                      pretrained = False,
	                      epochs = 1000,
	                      patience = 50,
	                      imgsz = [1000, 410],
	                      rect = True,
	                      workers = 7,          # CPU intensive
	                      batch = 128           # GPU VRAM intensive
	                      )