import requests
from dotenv import load_dotenv
import os
import json

def evaluate_sentence(sentence):
    payload = {
        "model": "gemini-2.0-flash",
        "messages": [
            #{"role": "system", "content": ""},
            {"role": "user", "content": sentence}
        ],
        "stream": False
    }
    
    headers = {"Authorization": f"Bearer {API_KEY}"}
    resp = requests.post(URL, headers=headers, json=payload, timeout=60)
    if resp.json()['choices'][0]['finish_reason'] == 'content_filter: PROHIBITED_CONTENT':
        print("Prohibited Content")
        return "Prohibited_content"
    response_content = resp.json()["choices"][0]["message"]["content"]
    return response_content

load_dotenv()
API_KEY = os.getenv("api_key")
URL = "https://api.ohmygpt.com/v1/chat/completions"

with open('/Users/wangjinghan/Documents/GitHub/Enhancing_LLAMA_Against_Language_Jailbreaks/results/implicit_judgment/implicit_contrast/gemini/pornography_sexual_minors_implicit.json','w+') as file:
    data = json.load(file)
    for item in data:
        try:
            get = item['Output']
        except:
            output = evaluate_sentence(item['AttackPrompt'])
            item['Output'] = output
            print(f"modify id={item['id']}")
    print(data)
    json.dump(data,file,indent=2)