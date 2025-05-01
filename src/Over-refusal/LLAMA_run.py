import ollama
import os
import json
from filter import *
from tqdm import tqdm

def generate(text):
    
    response = ollama.generate(
        model = "llama3.3",
        prompt=text,
        options={'system':""}
    )
    #print(response)
    content = response['response']
    #print(content)
    return content

def main(root):
    with open(f"{root}/Over-refusal/sap_sentences.json",'r') as file:
        sentences = json.load(file)
        index = 0
        data = []
        print("Generating refusal dataset......")
        for text in tqdm(sentences,desc = "Processing"):
            index += 1
            result = generate(text['sentence'])
            item = {
                "input":text,
                "system_prompt":"",
                "output":result,
                "is_refuse":is_refuse(result),
                "id":index,
            }
            data.append(item)
            if index % 10 == 0:
                with open(f"{root}/Over-refusal/implicit-result/raw/LLAMA3.3-70b/implicit_results.json",'w') as storage:
                    json.dump(data,storage,indent=2)
        with open(f"{root}/Over-refusal/implicit-result/implicit_results_LLAMA.json",'w') as storage:
            json.dump(data,storage,indent=2)

if __name__ == '__main__':
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    main(PROJECT_ROOT)
