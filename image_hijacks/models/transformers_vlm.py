from typing import List, Tuple, Union, Optional, Literal, Any
import torch
from torch import Tensor
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoConfig, AutoModelForCausalLM
from image_hijacks.models import AbstractLensModel
from jaxtyping import Float, Int64, Bool
import numpy as np

class TransformersVLM(AbstractLensModel):
    def __init__(self, model_id: str, model_dtype: torch.dtype = torch.float16, device_map="auto"):
        super().__init__()
        self.model_dtype = model_dtype
        self.model_id = model_id
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        
        # Load model
        # Try loading as Vision2Seq first, then CausalLM (for some models like GLM/Qwen sometimes)
        try:
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_id, 
                torch_dtype=model_dtype, 
                device_map=device_map,
                trust_remote_code=True
            )
        except Exception:
             self.model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=model_dtype, 
                device_map=device_map,
                trust_remote_code=True
            )
            
        self.model.eval()
        
        # Freeze model parameters since we only optimize the input
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Enable gradient checkpointing to save memory
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            
        self.static_image_info = {}

    def set_static_image_info(self, info: dict):
        self.static_image_info = {k: v for k, v in info.items() if k != "pixel_values"}

    def input_image_dims(self) -> Tuple[int, int]:
        # This varies by model and processor configuration.
        # For many modern VLMs, they handle variable sizes or have specific crops.
        # We'll try to infer from processor or return a default.
        if hasattr(self.processor, "image_processor"):
            if hasattr(self.processor.image_processor, "crop_size"):
                s = self.processor.image_processor.crop_size
                if isinstance(s, dict):
                    return (s.get("height", 224), s.get("width", 224))
                return (s, s)
            if hasattr(self.processor.image_processor, "size"):
                s = self.processor.image_processor.size
                if isinstance(s, dict):
                    return (s.get("height", 224), s.get("width", 224))
                return (s, s)
        return (224, 224) # Default fallback

    def preprocess_image(
        self, img: Union[Image.Image, List[Image.Image]]
    ) -> Tuple[Float[Tensor, "b c h w"], Bool[Tensor, "b img_seq_len"]]:
        # This method in the base class seems to expect a tensor output for "unnormalized" images
        # But AutoProcessor usually does normalization.
        # We might need to bypass normalization if the attack requires optimizing raw pixels.
        # For now, let's use the processor but maybe we need to handle the "unnormalized" part carefully.
        
        # If the attack framework expects [0,1] tensors to optimize, we should return that.
        if isinstance(img, Image.Image):
            imgs = [img]
        else:
            imgs = img
            
        # Convert to tensor [0, 1]
        import torchvision.transforms.functional as F
        tensors = [F.to_tensor(i) for i in imgs]
        pixel_values = torch.stack(tensors).to(self.device, dtype=self.model_dtype)
        
        # Dummy mask, assuming all valid
        # The shape of mask depends on how many tokens the image produces.
        # This is tricky for models like Qwen2-VL where token count is dynamic.
        # We might need a dummy implementation or run a forward pass to get lengths.
        
        # For now, return a placeholder mask.
        return pixel_values, torch.ones((len(imgs), 1), dtype=torch.bool, device=self.device)

    def normalize_image(
        self, pixel_values: Float[Tensor, "b c h w"]
    ) -> Float[Tensor, "b c h w"]:
        # Apply processor's normalization
        if hasattr(self.processor, "image_processor"):
            mean = self.processor.image_processor.image_mean
            std = self.processor.image_processor.image_std
            if mean is not None and std is not None:
                mean = torch.tensor(mean, device=pixel_values.device, dtype=pixel_values.dtype).view(1, -1, 1, 1)
                std = torch.tensor(std, device=pixel_values.device, dtype=pixel_values.dtype).view(1, -1, 1, 1)
                return (pixel_values - mean) / std
        return pixel_values

    def tokenize(
        self,
        text: Union[str, List[str]],
        mode: Literal["encoder", "decoder", "no_special_tokens"],
        max_length: Optional[int] = None,
        pad_to_max_length: bool = False,
        randomly_sample_system_prompt: bool = False,
    ) -> Tuple[Int64[Tensor, "b max_seq_len"], Bool[Tensor, "b max_seq_len"]]:
        
        if isinstance(text, str):
            text = [text]
            
        # Handle system prompt if needed (omitted for brevity, can add later)
        
        padding = "max_length" if pad_to_max_length else "longest"
        truncation = True if max_length else False
        
        inputs = self.processor.tokenizer(
            text,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            add_special_tokens=(mode != "no_special_tokens")
        )
        
        return inputs.input_ids.to(self.device), inputs.attention_mask.to(self.device).bool()

    def to_string(
        self, tokens: Int64[Tensor, "b seq_len"], skip_special_tokens=True
    ) -> List[str]:
        return self.processor.tokenizer.batch_decode(tokens, skip_special_tokens=skip_special_tokens)

    def get_image_embeddings(
        self,
        pixel_values: Float[Tensor, "b c h w"],
        tokens: Optional[Float[Tensor, "b tok_seq_len h_lm"]] = None,
        token_attention_mask: Optional[Bool[Tensor, "b tok_seq_len"]] = None,
    ) -> Float[Tensor, "b img_seq_len h_lm"]:
        # This is highly model specific.
        # We need to call the vision encoder.
        raise NotImplementedError("Subclasses must implement get_image_embeddings")

    def get_token_embeddings(
        self, tokens: Int64[Tensor, "b max_seq_len"]
    ) -> Float[Tensor, "b max_seq_len h_lm"]:
        return self.model.get_input_embeddings()(tokens)

    def get_embeddings_from_image_and_tokens(
        self,
        pixel_values: Float[Tensor, "b c h w"],
        tokens: Float[Tensor, "b tok_seq_len h_lm"],
        image_attention_mask: Optional[Bool[Tensor, "b img_seq_len"]] = None,
        token_attention_mask: Optional[Bool[Tensor, "b tok_seq_len"]] = None,
    ):
        # This is used for the attack to combine optimized image embeddings with text embeddings.
        # Also model specific.
        raise NotImplementedError

    def pad_token_id(self) -> int:
        if self.processor.tokenizer.pad_token_id is not None:
            return self.processor.tokenizer.pad_token_id
        return self.processor.tokenizer.eos_token_id

    def loss(self, logits, label_toks, padding_tok=None):
        if padding_tok is None:
            padding_tok = self.pad_token_id()
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = label_toks[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=padding_tok)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss

    @classmethod
    def load_model(cls, model_dtype=torch.half, requires_grad=False):
        raise NotImplementedError("Use constructor directly")

    def get_logits_from_embeddings(
        self,
        input_embeddings: Float[Tensor, "b src_seq_len h_lm"],
        attention_mask: Optional[Bool[Tensor, "b src_seq_len"]] = None,
        decoder_input_ids: Optional[Int64[Tensor, "b tgt_seq_len"]] = None,
        decoder_attention_mask: Optional[Bool[Tensor, "b tgt_seq_len"]] = None,
        image_inputs: Optional[Any] = None,
    ) -> Float[Tensor, "b seq_len n_tokens"]:
        
        kwargs = {}
        if image_inputs is not None and isinstance(image_inputs, dict):
             kwargs.update(image_inputs)
        
        return self.model(inputs_embeds=input_embeddings, attention_mask=attention_mask, **kwargs).logits

    def generate_end_to_end(
        self,
        image_inputs: Any,
        tokens: Float[Tensor, "b tok_seq_len h_lm"],
        image_attention_mask: Optional[Bool[Tensor, "b img_seq_len"]] = None,
        token_attention_mask: Optional[Bool[Tensor, "b tok_seq_len"]] = None,
        max_new_tokens: int = 20,
    ) -> Int64[Tensor, "b tok_seq_len n_tokens"]:
        
        kwargs = {}
        if isinstance(image_inputs, dict):
             kwargs.update(image_inputs)
        elif isinstance(image_inputs, torch.Tensor):
             kwargs["pixel_values"] = image_inputs
             if hasattr(self, "static_image_info") and self.static_image_info:
                 kwargs.update(self.static_image_info)
             
        if tokens.dtype in [torch.int64, torch.int32, torch.long]:
            return self.model.generate(input_ids=tokens, attention_mask=token_attention_mask, max_new_tokens=max_new_tokens, **kwargs)
        
        return self.model.generate(inputs_embeds=tokens, attention_mask=token_attention_mask, max_new_tokens=max_new_tokens, **kwargs)

    def generate_from_embeddings(
        self,
        input_embeddings: Float[Tensor, "b src_seq_len h_lm"],
        attention_mask: Optional[Bool[Tensor, "b src_seq_len"]] = None,
        max_new_tokens: int = 20,
        image_inputs: Optional[Any] = None,
    ) -> Int64[Tensor, "b new_seq_len n_tokens"]:
        
        kwargs = {}
        if image_inputs is not None and isinstance(image_inputs, dict):
             kwargs.update(image_inputs)
             
        return self.model.generate(inputs_embeds=input_embeddings, attention_mask=attention_mask, max_new_tokens=max_new_tokens, **kwargs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def get_logits_from_embeddings(
        self,
        input_embeddings: Float[Tensor, "b src_seq_len h_lm"],
        attention_mask: Optional[Bool[Tensor, "b src_seq_len"]] = None,
        decoder_input_ids: Optional[Int64[Tensor, "b tgt_seq_len"]] = None,
        decoder_attention_mask: Optional[Bool[Tensor, "b tgt_seq_len"]] = None,
        image_inputs: Any = None,
    ) -> Float[Tensor, "b seq_len n_tokens"]:
        # For Mllama, we need to pass pixel_values separately if they are not integrated into embeddings
        # But wait, if get_embeddings_from_image_and_tokens returned just text embeddings,
        # we need to pass image_inputs here.
        
        # If decoder_input_ids is None, we assume we are doing causal LM on input_embeddings
        # But Mllama might need pixel_values.
        
        kwargs = {}
        if image_inputs is not None and isinstance(image_inputs, dict):
             kwargs.update(image_inputs)
             
        if decoder_input_ids is not None:
            # Causal LM training with teacher forcing
            decoder_embeddings = self.get_token_embeddings(decoder_input_ids)
            full_embeddings = torch.cat([input_embeddings, decoder_embeddings], dim=1)
            
            if attention_mask is not None:
                if decoder_attention_mask is None:
                    decoder_attention_mask = torch.ones_like(decoder_input_ids, dtype=torch.bool, device=attention_mask.device)
                full_attention_mask = torch.cat([attention_mask, decoder_attention_mask], dim=1)
            else:
                full_attention_mask = None
                
            outputs = self.model(
                inputs_embeds=full_embeddings,
                attention_mask=full_attention_mask,
                **kwargs
            )
            # DEBUG PRINTS
            print(f"DEBUG: input_embeddings shape: {input_embeddings.shape}")
            print(f"DEBUG: decoder_input_ids shape: {decoder_input_ids.shape}")
            print(f"DEBUG: full_embeddings shape: {full_embeddings.shape}")
            print(f"DEBUG: outputs.logits shape: {outputs.logits.shape}")
            
            # Return logits corresponding to the decoder inputs
            return outputs.logits[:, input_embeddings.shape[1]:, :]

        outputs = self.model(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            **kwargs
        )
        return outputs.logits

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

class Llama3Vision(TransformersVLM):
    def preprocess_image(self, img):
        # If input is already a tensor (from attack optimization), we need to handle it carefully
        # to preserve gradients.
        if isinstance(img, torch.Tensor):
            # Assuming img is [B, C, H, W] or [C, H, W] and normalized/unnormalized?
            # The attack framework usually passes unnormalized [0,1] tensors.
            
            # Ensure tensor is [B, C, H, W]
            if img.dim() == 3:
                img = img.unsqueeze(0)
                
            h, w = img.shape[-2:]
            
            # 1. Run processor on dummy data to get the container
            # Use the actual size of the tensor to let processor decide tiling
            dummy_img = Image.new('RGB', (w, h)) 
            inputs = self.processor(images=dummy_img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 2. Normalize the input tensor
            # MllamaImageProcessor normalization
            mean = torch.tensor(self.processor.image_processor.image_mean, device=self.device, dtype=self.model_dtype).view(1, 3, 1, 1)
            std = torch.tensor(self.processor.image_processor.image_std, device=self.device, dtype=self.model_dtype).view(1, 3, 1, 1)
            
            # We don't resize here, we assume the attack loop provides the size it wants to optimize.
            # But we must ensure the processor didn't resize the dummy image to something else.
            # The processor returns pixel_values.
            # For Mllama, pixel_values is [B, Num_Images, Max_Tiles, C, H, W]
            # The H, W in pixel_values are the tile size (e.g. 336 or 448 or 560).
            # If our input img is not that size, we might need to resize it to match the tile size
            # OR we assume the input img IS the full image and we need to tile it manually?
            
            # WAIT. The processor does tiling. If we pass a full image tensor, we need to replicate the tiling logic differentiably.
            # That is very hard.
            
            # SIMPLIFICATION:
            # We assume the attack optimizes a "pre-tiled" image or a single tile image.
            # If the image fits in one tile (e.g. 336x336), the processor returns 1 tile (plus maybe global view).
            # If we want to support multi-tile optimization, we need to implement differentiable tiling.
            
            # For now, let's assume we are optimizing a single-tile image (e.g. 336x336 or 560x560).
            # And we assume the processor puts it in the first tile (or we need to find where).
            
            # Let's check the shape of pixel_values from dummy run.
            target_shape = inputs["pixel_values"].shape
            # target_shape: [B, Num_Images, Max_Tiles, C, TileH, TileW]
            
            tile_h, tile_w = target_shape[-2:]
            
            # If our input img matches tile size, great.
            if (h, w) != (tile_h, tile_w):
                # Resize input to match tile size
                img_resized = torch.nn.functional.interpolate(img, size=(tile_h, tile_w), mode='bilinear', align_corners=False)
            else:
                img_resized = img
                
            normalized_img = (img_resized - mean) / std
            
            # Create new pixel_values with gradient support
            new_pixel_values = torch.zeros(target_shape, device=self.device, dtype=self.model_dtype)
            
            # We inject into the first tile (index 0).
            # This assumes the processor mapped the image to the first tile.
            # For single tile images, this is usually true.
            new_pixel_values[:, :, 0, :, :, :] = normalized_img.unsqueeze(1) 
            
            inputs["pixel_values"] = new_pixel_values
            return inputs

        if isinstance(img, Image.Image):
            imgs = [img]
        else:
            imgs = img
        inputs = self.processor(images=imgs, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def normalize_image(self, image_inputs):
        return image_inputs

    def get_image_embeddings(self, image_inputs, tokens=None, token_attention_mask=None):
        if isinstance(image_inputs, torch.Tensor):
             pixel_values = image_inputs
             aspect_ratio_ids = self.static_image_info.get("aspect_ratio_ids")
             aspect_ratio_mask = self.static_image_info.get("aspect_ratio_mask")
        else:
             pixel_values = image_inputs["pixel_values"]
             aspect_ratio_ids = image_inputs.get("aspect_ratio_ids")
             aspect_ratio_mask = image_inputs.get("aspect_ratio_mask")

        vision_outputs = self.model.vision_model(
            pixel_values=pixel_values,
            aspect_ratio_ids=aspect_ratio_ids,
            aspect_ratio_mask=aspect_ratio_mask
        )
        return self.model.multi_modal_projector(vision_outputs[0])

    def get_embeddings_from_image_and_tokens(self, image_inputs, tokens, image_attention_mask=None, token_attention_mask=None):
        # Mllama keeps them separate. We return text tokens as embeddings.
        # The caller must handle passing image_inputs to the model forward separately.
        if tokens.dtype in [torch.int64, torch.int32, torch.long]:
            tokens = self.get_token_embeddings(tokens)
        return tokens, token_attention_mask

    def generate_end_to_end(
        self,
        image_inputs: Any,
        tokens: Float[Tensor, "b tok_seq_len h_lm"],
        image_attention_mask: Optional[Bool[Tensor, "b img_seq_len"]] = None,
        token_attention_mask: Optional[Bool[Tensor, "b tok_seq_len"]] = None,
        max_new_tokens: int = 20,
    ) -> Int64[Tensor, "b tok_seq_len n_tokens"]:
        
        if isinstance(image_inputs, torch.Tensor):
            # Mllama expects [B, Num_Images, Max_Tiles, C, H, W] (6 dims)
            # Current shape might be [B, Tiles, C, H, W] (5 dims) or [B, C, H, W] (4 dims)
            
            if image_inputs.dim() == 4:
                # [B, C, H, W] -> [B, 1, 1, C, H, W]
                image_inputs = image_inputs.unsqueeze(1).unsqueeze(1)
            elif image_inputs.dim() == 5:
                # [B, T, C, H, W] -> [B, 1, T, C, H, W]
                # We assume the 2nd dim is tiles for a single image
                image_inputs = image_inputs.unsqueeze(1)
        
        return super().generate_end_to_end(
            image_inputs, tokens, image_attention_mask, token_attention_mask, max_new_tokens
        )

class Qwen2VL(TransformersVLM):
    def preprocess_image(self, img):
        if isinstance(img, Image.Image):
            imgs = [img]
        else:
            imgs = img
        inputs = self.processor.image_processor(images=imgs, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def normalize_image(self, image_inputs):
        return image_inputs

    def get_image_embeddings(self, image_inputs, tokens=None, token_attention_mask=None):
        return self.model.visual(image_inputs["pixel_values"], grid_thw=image_inputs.get("image_grid_thw"))

    def get_embeddings_from_image_and_tokens(self, image_inputs, tokens, image_attention_mask=None, token_attention_mask=None):
        raise NotImplementedError("Qwen2-VL embedding mixing not implemented yet")

class GLM4V(TransformersVLM):
    def preprocess_image(self, img):
        if isinstance(img, Image.Image):
            imgs = [img]
        else:
            imgs = img
        inputs = self.processor(images=imgs, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def normalize_image(self, image_inputs):
        return image_inputs

    def get_image_embeddings(self, image_inputs, tokens=None, token_attention_mask=None):
        # Placeholder
        return self.model.transformer.vision.transformer(image_inputs["pixel_values"])

    def get_embeddings_from_image_and_tokens(self, image_inputs, tokens, image_attention_mask=None, token_attention_mask=None):
        raise NotImplementedError("GLM-4V embedding mixing not implemented yet")

