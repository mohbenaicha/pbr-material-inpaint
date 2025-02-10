import random
from PIL import Image
import numpy as np
import torch, cv2
from PIL import Image
from PyQt5.QtGui import QImage


def resize_to_target(image, target_size=(1024, 1024)):
    """Resize PIL image to target size."""
    return image.resize(target_size, Image.LANCZOS)


def color_to_canny(init_image):
    # Load color image, convert to guidance image through Canny edge detection
    initial_image_np = np.array(init_image)
    initial_image_gray = cv2.cvtColor(initial_image_np, cv2.COLOR_RGB2GRAY)
    
    # debug: canny edge detection
    initial_image_canny = cv2.Canny(initial_image_gray, 50, 100)

    initial_image_canny_rgb = cv2.cvtColor(initial_image_canny, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(initial_image_canny_rgb)

def detect_nan_hook(module, input, output):
    if isinstance(output, tuple):
        for i, out in enumerate(output):
            if isinstance(out, torch.Tensor) and torch.isnan(out).any():
                print(
                    f"[NaN DETECTED] {module.__class__.__name__} - Output Tensor {i} has NaNs!"
                )
    elif isinstance(output, torch.Tensor) and torch.isnan(output).any():
        print(f"[NaN DETECTED] {module.__class__.__name__} has NaNs!")


def scale_and_apply_lora_weights(pipe, original_lora_weights, rank):
    scale = 1 / rank
    for name, param in pipe.unet.named_parameters():
        if "lora" in name.lower():
            param.data = original_lora_weights[name] * scale


def scale_and_apply_lora_weights_inplace(pipe, rank):
    scale = 1 / rank
    for name, param in pipe.unet.named_parameters():
        if "lora" in name.lower():
            param.data.mul_(scale)


def reset_lora_weights(pipe, original_lora_weights):
    for name, param in pipe.unet.named_parameters():
        if "lora" in name.lower():
            param.data = original_lora_weights[name]


def rescale(img) -> np.array:
    try:
        ar = np.array(img)
        mn = np.linalg.norm(np.min(ar))
        mx = np.linalg.norm(np.max(ar))
        if mx == mn:
            raise ValueError("Max and min values are the same, cannot rescale image.")
        norm = (ar - mn) * (1.0 / (mx - mn))
        return norm
    except Exception as e:
        print(f"Error in rescaling image: {e}")
        return np.array(img)  # Return the original image array in case of error


def create_mask(user_image, brush_strokes) -> np.array:
    # Create a mask from brush strokes
    mask = np.zeros(user_image.size, dtype=np.uint8)
    # Logic to create mask from brush strokes goes here
    return mask


def save_image(image, path) -> None:
    image.save(path)


def mask_to_canny(mask, blur_radius=5, canny_threshold1=30, canny_threshold2=200) -> Image:
   
    # Load mask in grayscale
    mask_np = np.array(mask)

    # Invert mask (so white = generation area)
    mask_inverted = cv2.bitwise_not(mask_np)

    # Apply Gaussian blur to soften mask edges
    blurred_mask = cv2.GaussianBlur(mask_inverted, (blur_radius, blur_radius), 0)

    # Apply Canny edge detection
    mask_canny = cv2.Canny(blurred_mask, canny_threshold1, canny_threshold2)

    # Convert edges to RGB format (required by ControlNet)
    mask_canny_rgb = cv2.cvtColor(mask_canny, cv2.COLOR_GRAY2RGB)

    return Image.fromarray(mask_canny_rgb)

def update_user_prompt(prompt):
    mapPrompts = {
        "albedo": f"an albedo map of {prompt}. VAR2, colormap, muted realistic colors",
        "rough": f"a roughness map of {prompt}. black and white, roughmap",
        "normal": f"a normal map of {prompt}. normal map",
        "ambientocl": f"an ambient occlusion map of {prompt}, black and white, ambmap",
        "metal": f"a metallic map of {prompt}, black and white, metalmap,black and white, metalmap",
        "specular": f"a specular map of {prompt}, black and white, specmap",
        "height": f"a height map of {prompt}, grayscale, heighmap,black and white, heighmap",
    }
    return mapPrompts


negativePrompts = {
    "height": "weird, ugly, low quality, messed up, unrealistic, VAR1, blurry, low resolution",
    "albedo": "weird, ugly, low quality, messed up, unrealistic, VAR1, blurry, low resolution, garish colors, very vibrant",
    "rough": "weird, ugly, low quality, messed up, unrealistic, VAR1",
    "normal": "colormap",
    "specular": "weird, ugly, low quality, messed up, unrealistic, VAR1",
    "ambientocl": "weird, ugly, low quality, messed up, unrealistic, VAR1",
    "metal": "weird, ugly, low quality, messed up, unrealistic, VAR1",
}


def qimage_to_pil(qimage) -> Image:
    """Convert QImage to PIL Image"""
    # Convert QImage to a format compatible with PIL
    qimage = qimage.convertToFormat(QImage.Format_RGBA8888)

    # Extract byte data from QImage
    byte_data = qimage.bits().asstring(qimage.byteCount())

    # Create PIL Image from byte data
    pil_image = Image.frombytes("RGBA", (qimage.width(), qimage.height()), byte_data)
    return pil_image


def pil_to_qimage(pil_image) -> QImage:
    """Convert PIL Image to QImage"""
    pil_image = pil_image.convert("RGBA")
    data = pil_image.tobytes("raw", "RGBA")
    qimage = QImage(data, pil_image.width, pil_image.height, QImage.Format_RGBA8888)
    return qimage

def generate_random_seed():
     # Set random seed for reproducibility
    random_seed = random.randint(0, 10000)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

def apply_stylesheet(obj) -> None:
    style_sheet = """
    QWidget {
        background-color: #2E2E2E;
        color: #FFFFFF;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 16px;
        font-weight: bold;
    }
    QLineEdit, QSlider, QPushButton, QTabWidget::pane, QFrame {
        border-radius: 10px;
    }
    QLineEdit {
        background-color: #3E3E3E;
        color: #00FF00;
        padding: 5px;
        font-size: 16px;
        font-weight: bold;
    }
    QPushButton {
        background-color: #3E3E3E;
        color: #00FF00;
        padding: 10px;
        font-size: 16px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #5E5E5E;
    }
    QTabWidget::pane {
        border: 1px solid #3E3E3E;
    }
    QTabBar::tab {
        background: #3E3E3E;
        color: #00FF00;
        padding: 10px;
        border-top-left-radius: 10px;
        border-top-right-radius: 10px;
        font-size: 16px;
        font-weight: bold;
    }
    QTabBar::tab:selected {
        background: #5E5E5E;
    }
    QLabel {
        color: #00FF00;
        font-size: 16px;
        font-weight: bold;
    }
    """
    obj.setStyleSheet(style_sheet)
