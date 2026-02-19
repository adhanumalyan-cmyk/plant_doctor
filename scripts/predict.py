import torch
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import sys

# --------------------------
# Configuration
# --------------------------
NUM_CLASSES = 15                     # <-- set to your number of classes
MODEL_PATH = "data/models/best_model.pth"
CLASS_NAMES_PATH = "data/models/class_names.pth"

# --------------------------
# Load class names
# --------------------------
if Path(CLASS_NAMES_PATH).exists():
    class_names = torch.load(CLASS_NAMES_PATH)
else:
    # Fallback – replace with your actual class names in order
    class_names = [f"Class_{i}" for i in range(NUM_CLASSES)]
    print("Warning: class_names.pth not found. Using generic indices.")

# --------------------------
# Load model
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)   # <-- fixed
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# --------------------------
# Transforms (same as validation)
# --------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --------------------------
# Predict on image from command line
# --------------------------
if len(sys.argv) < 2:
    print("Usage: python predict.py <image_path>")
    sys.exit(1)

img_path = sys.argv[1]
image = Image.open(img_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)
    predicted_idx = output.argmax(dim=1).item()

predicted_class = class_names[predicted_idx] if predicted_idx < len(class_names) else f"Class_{predicted_idx}"
print(f"Predicted: {predicted_class}")