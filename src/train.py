from ultralytics import YOLO

# classification
# YOLO 12
# use only yolo12n-cls.yaml, yolo12 has no pretrained model

# Load a model
model = YOLO("yolo12n-cls.yaml")  # build a new model from YAML
#model = YOLO("YOLO11n-cls.pt")  # load a pretrained model (recommended for training)
#model = YOLO("yolo11n-cls.yaml").load("yolo11n-cls.pt")  # build from YAML and transfer weights



# image resolution: 4000 * 1638

# Train the model
results = model.train(data="C:/Users/fei18/PycharmProjects/AWC_CV_QC/dataset",
                      epochs=50,
                      imgsz=[4000/4, 1638/4])

