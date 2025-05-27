# utils.py

import torch
from torchvision import transforms
from PIL import Image
import io

def load_model(model_class, model_path, device=None):
    """
    Load a trained PyTorch model.

    Args:
        model_class (nn.Module): The model architecture class (not instance).
        model_path (str): Path to the .pth file.
        device (str or torch.device, optional): 'cpu' or 'cuda'. Default is auto-detect.

    Returns:
        model (nn.Module): Loaded model ready for inference.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def transform_image(image):
    """
    Transforms a PIL Image to a normalized tensor ready for model input.
    """
    try:
        if not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL.Image object")
        
        # Directly apply transforms; do NOT call Image.open here
        image_tensor = transform(image).unsqueeze(0)  # add batch dimension
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return image_tensor.to(device)
    
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")

