
import torch
import faulthandler
import gc
faulthandler.enable()

from experiments.exp_demo_imgs.config import load_model_llama_3_2_vision
from image_hijacks.models.transformers_vlm import Llama3Vision
from PIL import Image

def print_memory(step):
    if torch.cuda.is_available():
        print(f"[{step}] Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"[{step}] Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

def debug_segfault():
    print("Loading model...")
    model = load_model_llama_3_2_vision()
    print("Model loaded.")
    
    # Check gradient checkpointing
    if hasattr(model.model, "is_gradient_checkpointing"):
        print(f"Gradient checkpointing enabled: {model.model.is_gradient_checkpointing}")
    else:
        print("Gradient checkpointing attribute not found.")
        
    # Check if params are frozen
    param = next(model.model.parameters())
    print(f"Model parameter requires_grad: {param.requires_grad}")
        
    print_memory("After Load")
    
    print("Creating dummy inputs...")
    # Try a smaller image to see if it fits
    # 560x560 -> 336x336 (Llama 3.2 tile size is often related to 336 or 448, let's try single tile)
    # If 336 fits, it's likely a single tile.
    img_size = 560
    print(f"Using image size: {img_size}x{img_size}")
    img = Image.new('RGB', (img_size, img_size), color='red')
    
    # 2. Preprocess (differentiable path)
    # We need a tensor input to trigger the differentiable path
    import torchvision.transforms.functional as F
    img_tensor = F.to_tensor(img).to(model.device)
    img_tensor.requires_grad = True
    
    print("Running preprocess_image...")
    image_inputs = model.preprocess_image(img_tensor)
    print(f"Pixel values shape: {image_inputs['pixel_values'].shape}")
    print("Preprocess done.")
    print_memory("After Preprocess")
    
    # 3. Forward pass
    print("Running forward pass...")
    # Create dummy tokens
    text = "Hello world " * 50 # ~100+ tokens
    tokens, mask = model.tokenize(text, mode="encoder", max_length=200, pad_to_max_length=True)
    
    # get_embeddings_from_image_and_tokens
    embs, attn_mask = model.get_embeddings_from_image_and_tokens(image_inputs, tokens)
    
    # get_logits_from_embeddings
    logits = model.get_logits_from_embeddings(embs, attention_mask=attn_mask, image_inputs=image_inputs)
    print("Forward pass done.")
    print(f"Logits shape: {logits.shape}")
    print_memory("After Forward")
    
    # Test backward
    print("Testing backward...")
    loss = logits.sum()
    loss.backward()
    print("Backward done.")
    print_memory("After Backward")

if __name__ == "__main__":
    debug_segfault()
