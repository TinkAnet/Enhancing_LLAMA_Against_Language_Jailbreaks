import unsloth
# Import other libraries
from unsloth import FastLanguageModel
import torch
from peft import PeftModel
import os
import json 
import logging
import time

# Load the base model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Load the LoRA adapter
model = PeftModel.from_pretrained(model, "bebop404/my-lora-adapter")
print("Complete loading model and model adapter")

def generate(text,sysprompt):
    inputs = tokenizer(text,return_tensors = 'pt').to("cuda")
    output = model.generate(**inputs,max_new_tokens = 128,temperature = 0.7)

