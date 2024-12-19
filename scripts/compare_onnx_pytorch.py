import os
import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import onnxruntime
import numpy as np
import cv2

classnames = []
file = open("outputs/pos_classes.txt")
lines = file.readlines()
for line in lines:
    classnames.append(line.strip())
print("classnames", classnames)

# Define the image preprocessing pipeline
preprocess = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to load and preprocess images from a directory
def load_images_from_directory(directory):
    images = []
    original_images = []
    filenames = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = Image.open(img_path)
       # img = cv2.imread(img_path)
       # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_images.append(img.copy())
        img = preprocess(img)
        img = img.cuda()
        images.append(img)
        filenames.append(filename)
    return images, original_images, filenames

# Function to perform inference using PyTorch model
def infer_pytorch(model, images):
    model.eval()
    with torch.no_grad():
        inputs = torch.stack(images)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
    return preds

# Function to perform inference using ONNX model
def infer_onnx(onnx_model_path, images):
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = onnxruntime.InferenceSession(onnx_model_path, options, providers=['CPUExecutionProvider'])
    inputs = np.stack([img.cpu().numpy() for img in images])
    ort_inputs = {ort_session.get_inputs()[0].name: inputs}
    ort_outs = ort_session.run(None, ort_inputs)
    preds = np.argmax(ort_outs[0], axis=1)
    return preds

# Function to save images with predicted labels
def save_images_with_labels(images, labels, filenames, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    font = ImageFont.load_default()
    for img, label_index, filename in zip(images, labels, filenames):
        label = classnames[label_index]
        save_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_pred_{label}.png")
        img.save(save_path)
       # cv2.imwrite(save_path, img)

# Load fine-tuned PyTorch model
model = torch.load('outputs/best_model.pt')

# Load images
images, original_images, filenames = load_images_from_directory('inference_dataset')

# Perform inference with PyTorch model
pytorch_preds = infer_pytorch(model, images)

# Perform inference with ONNX model
#onnx_preds = infer_onnx("outputs/pos_classification_v8.onnx", images)

# Save the resulting images with predicted labels
save_images_with_labels(original_images, pytorch_preds.cpu().numpy(), filenames, 'output_pytorch')
#save_images_with_labels(original_images, onnx_preds, filenames, 'output_onnx')

print("Inference complete. Results saved in 'output_pytorch' and 'output_onnx' directories.")
