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
	# original fake resolution: ~3000 * ~1000
	# original real resolution: ~750 * ~250
	# optimized resolution (naked eye): 300 * 100
	# Laptop: C:/Users/fei18/PycharmProjects/AWC_CV_QC/dataset
	# Desktop: F:/JetBrains/PycharmProjects/AWC_CV_QC/dataset

	results = model.train(#data="F:/JetBrains/PycharmProjects/AWC_CV_QC/dataset",
						  data="C:/Users/fei18/PycharmProjects/AWC_CV_QC/dataset",
	                      pretrained=False,
	                      epochs=200,
	                      patience=0,
		# training must use square image size, multiple of 32, will auto padding
		# inference can use other aspect ratio
						  #imgsz=[750, 250],
						  #imgsz=[300, 100],
						  imgsz=640,
	                      rect=True,
	                      workers=3,          # CPU intensive, number of cores
	                      batch=16           # GPU VRAM / RAM intensive
	                      )