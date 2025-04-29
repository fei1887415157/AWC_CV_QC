"""
ONNX: CPU and GPU, slow
TensorRT: Nvidia GPU, fast
"""

from ultralytics import YOLO

# Load a model
model = YOLO("runs/classify/11s/weights/best.pt")  # load a custom trained model

# Export the model
#model.export(format="onnx")
model.export(format="tensorrt")