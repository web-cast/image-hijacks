
import torch
from transformers import AutoProcessor
from PIL import Image
import numpy as np

def test_processor_grad():
    model_id = "/ai/gpt/code/image-hijacks/downloads/model_checkpoints/Llama-3.2-11B-Vision-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Create a dummy image tensor with gradients
    image_tensor = torch.rand(1, 3, 560, 560, requires_grad=True)
    
    print("Testing processor with tensor input...")
    print(f"Image processor config: {processor.image_processor}")
    try:
        # AutoProcessor usually expects list of numpy arrays or PIL images
        # It might not accept a torch Tensor directly for 'images' argument in a way that preserves grad.
        # But let's try passing it as is.
        # inputs = processor(images=image_tensor, return_tensors="pt")
        
        # Manual differentiable preprocessing
        # 1. Resize (if needed) - assuming input is already correct size for now or we resize
        # Llama 3.2 Vision uses 560x560 tiles.
        # For simplicity, let's assume we just want to pass the image as a single tile + global view (if applicable)
        # Or just replicate the processor's output structure manually.
        
        # Let's see what the processor does with a numpy version of our tensor
        image_np = image_tensor.detach().cpu().numpy()[0].transpose(1, 2, 0) # HWC
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
        
        inputs_ref = processor(images=image_pil, return_tensors="pt")
        print(f"Reference pixel_values shape: {inputs_ref['pixel_values'].shape}")
        
        # Reconstruct pixel_values from image_tensor in a differentiable way
        # Shape is [1, 1, 4, 3, 560, 560] -> [Batch, Num_Images, Max_Tiles, Channels, H, W]
        # The processor splits the image into tiles.
        # If we assume the input image IS the size of the canvas (e.g. 1120x1120 for 4 tiles), we can unfold.
        # But our input is 560x560.
        
        # If input is 560x560, it likely fits in 1 tile + 1 global view?
        # Let's check the reference shape again.
        # If ref shape is [1, 1, 4, 3, 560, 560], it means it padded to 4 tiles.
        
        # We need to normalize
        mean = torch.tensor(processor.image_processor.image_mean).view(1, 3, 1, 1).to(image_tensor.device)
        std = torch.tensor(processor.image_processor.image_std).view(1, 3, 1, 1).to(image_tensor.device)
        
        normalized_image = (image_tensor - mean) / std
        
        # Construct the 4 tiles. 
        # If the image is small, maybe it just pads?
        # For 560x560 input, it probably puts it in one tile and pads the rest.
        # We can try to "hack" it by just expanding our tensor to the target shape
        # and using the reference inputs for the other non-differentiable parts (mask, ids).
        
        # Create a zero tensor of the target shape
        target_shape = inputs_ref['pixel_values'].shape
        pixel_values = torch.zeros(target_shape, device=image_tensor.device, dtype=image_tensor.dtype)
        
        # We need to know WHERE the image went.
        # For a single 560x560 image, it probably goes into the first tile (or global view?).
        # Let's assume we can just replace the content of the reference tensor with our grad-enabled tensor
        # BUT we need to do the normalization ourselves.
        
        # Actually, a better way is:
        # 1. Run processor on detached image to get layout (padding, tile positions).
        # 2. Create a mask of where the original pixels ended up.
        # 3. Fill those positions with our normalized_image.
        
        # Simplified test: Just put the image in the first tile and see if we can backprop.
        # (This is just a test of the concept, not the full implementation)
        
        # Let's assume the first tile [0, 0, 0] is the global view or the first crop.
        pixel_values[:, :, 0, :, :, :] = normalized_image
        
        # Enable grad
        # pixel_values is created from normalized_image which has grad.
        
        print(f"Output pixel_values shape: {pixel_values.shape}")
        if "aspect_ratio_mask" in inputs_ref:
            print(f"Aspect ratio mask: {inputs_ref['aspect_ratio_mask']}")
        print(f"Output requires_grad: {pixel_values.requires_grad}")
        
        if pixel_values.requires_grad:
            print("Success: Gradients preserved!")
            loss = pixel_values.sum()
            loss.backward()
            print("Backward pass successful.")
            print(f"Input grad norm: {image_tensor.grad.norm()}")
        else:
            print("Failure: Gradients NOT preserved.")
            
    except Exception as e:
        print(f"Processor failed with tensor input: {e}")

if __name__ == "__main__":
    test_processor_grad()
