import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# 1. Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Model Loader
def load_cardiomegaly_model(weights_path):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file '{weights_path}' not found in current directory.")
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# 3. Preprocessing (Matches training transforms)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. Prediction Execution
def predict(image_path, weights_path='cardiomegaly_mobilenetv2_finetuned.pt'):
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found.")
        return

    model = load_cardiomegaly_model(weights_path)
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_class = torch.max(probabilities, 0)

    class_names = ['Normal', 'Cardiomegaly']
    result = class_names[predicted_class.item()]
    
    print("\n" + "="*30)
    print(" CARDIOMEGALY DETECTION RESULT ")
    print("="*30)
    print(f"Image File   : {image_path}")
    print(f"Diagnosis    : {result}")
    print(f"Confidence   : {confidence.item() * 100:.2f}%")
    print(f"Normal Prob  : {probabilities[0].item() * 100:.2f}%")
    print(f"Disease Prob : {probabilities[1].item() * 100:.2f}%")
    print("="*30 + "\n")

if __name__ == '__main__':
    # Usage: python predict.py <path_to_xray_image>
    if len(sys.argv) > 1:
        target_image = sys.argv[1]
    else:
        target_image = 'test_xray.png'  # Default image name if none supplied
        
    predict(target_image)