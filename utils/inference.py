import json
import os
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'qwen-vl-finetune'))
from qwenvl.data.data_processor import preprocess_qwen_visual

MODEL_PATH = "/home/mrao/DarrellGroup/Qwen3-VL/qwen-vl-finetune/output/checkpoint-60"
TEST_DATA_PATH = "/home/mrao/DarrellGroup/Qwen3-VL/data/test.json"
DATA_BASE_PATH = "/home/mrao/DarrellGroup/Qwen2.5-VL/data/rlbench_icl_small/train"

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

with open(TEST_DATA_PATH, 'r') as f:
    test_data = json.load(f)

for sample in test_data:
    # Add data_path to sample for image loading
    sample_with_path = sample.copy()
    sample_with_path["data_path"] = DATA_BASE_PATH
    
    # Create a sample with only the human message for inference
    inference_sample = {
        "id": sample["id"],
        "image": sample["image"],
        "conversations": [sample["conversations"][0]],
        "data_path": DATA_BASE_PATH
    }
    
    # Use the same preprocessing as training
    processed_data = preprocess_qwen_visual([inference_sample], processor, add_gen_prompt=True)
    
    # Extract the processed inputs
    input_ids = processed_data["input_ids"].to(model.device)
    pixel_values = processed_data["pixel_values"].to(model.device) if "pixel_values" in processed_data else None
    image_grid_thw = processed_data["image_grid_thw"].to(model.device) if "image_grid_thw" in processed_data else None
    
    # Prepare inputs dict
    inputs = {"input_ids": input_ids}
    if pixel_values is not None:
        inputs["pixel_values"] = pixel_values
    if image_grid_thw is not None:
        inputs["image_grid_thw"] = image_grid_thw
    
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    
    prediction = processor.tokenizer.decode(generated_ids_trimmed[0], skip_special_tokens=True)
    gt = sample['conversations'][1]['value']

    print(f"Sample {sample['id']}: {prediction == gt}")
    print(f"Prediction: {prediction}")
    print(f"Ground truth: {gt}\n")