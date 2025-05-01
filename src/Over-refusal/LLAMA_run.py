import ollama
import os
import json
from filter import *

def generate(text):
    try:
        response = ollama.generate(
            model = "LLAMA3.3-70b",
            prompt=text,
            options={'system':""}
        )
        content = response['response']
    except:
        return "Error"
    return content

def main(root):
    with open(f"{root}/Over-refusal/sap_sentences.json",'r') as file:
        sentences = json.load(file)
        index = 0
        data = []
        print("Generating refusal dataset......")
        for text in sentences:
            index += 1
            result = generate(text)
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