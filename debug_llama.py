
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
from PIL import Image
import requests

def test_llama_forward():
    model_id = "/ai/gpt/code/image-hijacks/downloads/model_checkpoints/Llama-3.2-11B-Vision-Instruct"
    
    print(f"Loading model: {model_id}")
    try:
        model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )
        processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Create dummy image
    image = Image.new('RGB', (224, 224), color='red')
    text = "<|image|>If I had to write a haiku for this one, it would be: "

    print("Processing inputs...")
    inputs = processor(images=image, text=text, return_tensors="pt").to(model.device)

    print("Running forward pass...")
    try:
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=10)
        print("Forward pass successful.")
        print("Output:", processor.decode(output[0]))
    except Exception as e:
        print(f"Forward pass failed: {e}")

if __name__ == "__main__":
    test_llama_forward()
