import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

CLASS_NAMES = [
    'Actinic Keratoses (akiec)', 'Basal Cell Carcinoma (bcc)', 'Benign Keratosis (bkl)', 
    'Dermatofibroma (df)', 'Melanoma (mel)', 'Melanocytic Nevi (nv)', 'Vascular Lesions (vasc)'
]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def remove_hair(image_cv):
    gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    final_image = cv2.inpaint(image_cv, thresh, 1, cv2.INPAINT_TELEA)
    return final_image

def crop_contour(image_cv):
    gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image_cv
    
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    h_img, w_img, _ = image_cv.shape
    if w < w_img * 0.1 or h < h_img * 0.1: 
        return image_cv
        
    cropped = image_cv[y:y+h, x:x+w]
    return cropped

def process_image_opencv(pil_image, use_hair_removal=True, use_cropping=True):
    img_cv = np.array(pil_image)
    processed = img_cv.copy()
    
    if use_cropping:
        try:
            processed = crop_contour(processed)
        except Exception:
            pass 
            
    if use_hair_removal:
        try:
            processed = remove_hair(processed)
        except Exception:
            pass

    return Image.fromarray(processed)

def predict_probabilities(model, image_tensor):
    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)
        outputs, attn_map = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()[0], attn_map.cpu()

def generate_heatmap(pil_image, attn_map_tensor, model_name=""):
    img_np = np.array(pil_image)
    h, w, _ = img_np.shape
    
    attn_np = attn_map_tensor[0, 0].numpy()
    
    attn_min = attn_np.min()
    attn_max = attn_np.max()
    
    # Normalisasi dasar
    attn_norm = (attn_np - attn_min) / (attn_max - attn_min + 1e-8)
    
    attn_resized = cv2.resize(attn_norm, (w, h))
    
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    
    return Image.fromarray(overlay)