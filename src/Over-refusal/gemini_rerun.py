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
    print(resp.json())
    response_content = resp.json()["choices"][0]["message"]["content"]
    return response_content

load_dotenv()
API_KEY = os.getenv("api_key")
URL = "https://api.ohmygpt.com/v1/chat/completions"

with open("/Users/wangjinghan/Documents/GitHub/Enhancing_LLAMA_Against_Language_Jailbreaks/results/implicit_judgment/implicit_contrast/QwQ_32b/fraud_implicit.json",'r') as file:
    data = json.load(file)
    result = evaluate_sentence(data[99]['AttackPrompt'])
    print(result)
    data[99]['Output']

with open("/Users/wangjinghan/Documents/GitHub/Enhancing_LLAMA_Against_Language_Jailbreaks/results/implicit_judgment/implicit_contrast/QwQ_32b/race_implicit.json",'r') as file:
    data = json.load(file)
    result = evaluate_sentence(data[75]['AttackPrompt'])
    print(result)