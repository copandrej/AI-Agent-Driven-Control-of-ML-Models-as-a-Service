"""Download and cache all model weights during Docker build."""

import torchvision.models as models

print("Downloading MobileNetV2 weights...")
models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

print("Downloading ResNet50 weights...")
models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

print("All models cached.")
