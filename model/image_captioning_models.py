
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BlipProcessor, BlipForConditionalGeneration, VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, AutoProcessor, AutoModelForCausalLM

import re
import gc

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def test_blip2(image):
    device = get_device()
    
    model_name = "Salesforce/blip2-opt-2.7b"
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    
    inputs = processor(image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_length=200,
            min_length=25,
            num_beams=8,
            length_penalty=1.2,
            repetition_penalty=1.3,
            temperature=0.8,
            top_p=0.9,
        )
    
    caption = processor.decode(generated_ids[0], skip_special_tokens=True)
    
    # Clean up memory
    del model, processor
    cleanup_memory()
    
    return caption.strip()

def test_blip(image):
    device = get_device()
    
    model_name = "Salesforce/blip-image-captioning-large"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
    inputs = processor(image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_length=200,
            min_length=25,
            length_penalty=1.2,
            num_beams=8,
            early_stopping=True,
            repetition_penalty=1.3,
        )
    
    caption = processor.decode(generated_ids[0], skip_special_tokens=True)
    
    del model, processor
    cleanup_memory()
    
    return caption.strip()

def test_git_model(image):
    device = get_device()
    model_name = "microsoft/git-base-coco"
    
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    # Process image
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values=inputs.pixel_values,
            max_length=200,
            min_length=20,
            num_beams=8,
            length_penalty=1.2,
            repetition_penalty=1.2,
            temperature=0.7,
            top_p=0.9,
            early_stopping=True,
        )
    
    # Decode the result
    caption = processor.decode(generated_ids[0], skip_special_tokens=True)
    
    del model, processor
    cleanup_memory()
    
    return caption.strip()

def test_vit_gpt2(image):
    device = get_device()
    
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    
    model.to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
    
    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    
    gen_kwargs = {
        "max_length": 20,
        "num_beams": 5,
        "length_penalty": 0.8,
        "repetition_penalty": 1.5,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    with torch.no_grad():
        output_ids = model.generate(pixel_values, **gen_kwargs)
    
    # Decode
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    caption = caption.strip()
    caption = re.sub(r'_+', '', caption)
    caption = re.sub(r'\s+', ' ', caption)
    
    if caption and not caption.endswith('.'):
        caption = caption.rstrip('.,!?') + '.'
    
    del model, feature_extractor, tokenizer
    cleanup_memory()
    
    return caption
