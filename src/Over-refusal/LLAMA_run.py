import json
from llamaapi import LlamaAPI
from dotenv import load_dotenv
import os
from filter import is_refuse
from tqdm import tqdm

# Initialize the Llama API
load_dotenv()
API_KEY = os.getenv("llama_api_key")
llama = LlamaAPI(API_KEY)

def process_attack_prompt(attack_prompt):
    #print("="*100 + "\n")
    #print(f"Input: {attack_prompt}\n")  
    
    api_request_json = {
        "model": "llama3.3-70b", 
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": attack_prompt},
        ],
        "stream": False,
        "max_tokens": 1500  # Increase token limit to allow for longer outputs
    }

    try:
        response = llama.run(api_request_json)
        response_data = response.json()
        print(f"Response data: {json.dumps(response_data, indent=4)}")  # Print full response to check structure
        content = response_data["choices"][0]["message"]["content"]
        print(f"Output:\n{content}\n")  
        return content
    except Exception as e:
        print(f"Error processing prompt: {e}")
        return None


def main(root):
    with open(f"{root}/Over-refusal/sap_sentences.json","r") as file:
        data = json.load(file)
        output = []
        index = 0
        print("Processing...")
        for item in tqdm(data):
            index += 1
            response = process_attack_prompt(item['sentence'])
            try:
                is_refuse_resp = is_refuse(response)
            except:
                is_refuse_resp = "Error"
            output.append(
                {
                    "input":item['sentence'],
                    "system_prompt":"",
                    "output":response,
                    "is_refuse":is_refuse_resp,
                    "index":index,
                }
            )
            if index % 10 == 0:
                with open(f"{root}/Over-refusal/implicit-result/raw/LLAMA3.3-70b/implicit_results.json",'w') as temp:
                    json.dump(output,temp,indent=2)
        with open(f"{root}/Over-refusal/implicit-result/implicit_results_LLAMA.json",'w') as storage:
            json.dump(output,storage,indent=2)

if __name__ == '__main__':
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    main(PROJECT_ROOT)
