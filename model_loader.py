# model_loader.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

model_name = "ibm-granite/granite-3.2-2b-instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

print("🚀 Loading Granite model (this will run only once)...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

print("✅ Granite model loaded successfully!")