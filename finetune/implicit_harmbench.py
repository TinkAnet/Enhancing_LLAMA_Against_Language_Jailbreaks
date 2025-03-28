import json
import os
from datasets import load_dataset
from dotenv import load_dotenv
from llamaapi import LlamaAPI

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
llama_api_key = os.getenv("LLAMA_API_KEY")
if not llama_api_key:
    raise ValueError("LLAMA_API_KEY not found in .env file")

# Initialize the Llama API
llama = LlamaAPI(llama_api_key)

def get_llama_response(prompt):
    api_request_json = {
        "model": "llama3.3-70b",
        "messages": [
            {"role": "system", "content": "Answer the question."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "max_tokens": 1500
    }
    try:
        response = llama.run(api_request_json)
        response_data = response.json()
        content = response_data["choices"][0]["message"]["content"]
        return content
    except Exception as e:
        print(f"Error processing prompt: {e}")
        return None

# Load the HarmBench dataset from Hugging Face
ds = load_dataset("walledai/HarmBench", "standard")
data_split = ds["train"] if "train" in ds else ds

# --- Debug: Print unique categories ---
unique_categories = set(entry.get("category") for entry in data_split if entry.get("category") is not None)
print("Unique categories in the dataset:", unique_categories)

output_data = []

# Process each entry (process all entries regardless of category)
for entry in data_split:
    input_text = entry.get("prompt")
    category = entry.get("category")
    
    if input_text:
        print(f"Processing prompt: {input_text[:50]}...")  # preview first 50 characters
        response_text = get_llama_response(input_text)
        if response_text is None:
            response_text = ""
        
        output_data.append({
            "input": input_text,
            "response": response_text,
            "category": category
        })
    else:
        print("Skipping an entry due to missing 'input' field.")

# Create the output directory if it doesn't exist
output_dir = "harmbench"
os.makedirs(output_dir, exist_ok=True)

# Save the output to a JSON file
output_file = os.path.join(output_dir, "before_finetune.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print(f"Processing complete. Results saved to {output_file}.")
