import cv2
import base64
import torch
import torchvision

def load_image(image_path: str) -> torch.Tensor:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torchvision.transforms.ToTensor()(img)
    return img

def encode_image_to_base64(image_path: str) -> str:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

def save_tensor_as_image(tensor: torch.Tensor, image_path: str):
    torchvision.utils.save_image(tensor, image_path)
