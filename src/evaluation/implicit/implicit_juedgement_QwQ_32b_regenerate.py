import requests
from dotenv import load_dotenv
import os
import json
import ollama
import re
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

#load_dotenv()
#API_KEY = os.getenv("api_key")
#URL = "https://api.ohmygpt.com/v1/chat/completions"

def filter_thinking_content(text):
    pattern = r'<think>.*?</think>'
    filtered_text = re.sub(pattern, '', text, flags=re.DOTALL)
    return filtered_text

def process(text):
    response = ollama.generate(
        model = 'qwq',
        prompt = text,
        options = {"system": "You are a helpful AI assistant"}
    )
    return response
'''
with open("/home/23099359d/Enhancing_LLAMA_Against_Language_Jailbreaks/results/implicit_judgment/implicit_contrast/QwQ_32b/fraud_implicit.json",'r+') as file:
    data = json.load(file)
    result = process(data[99]['AttackPrompt'])
    print(result)
    data[99]['Output'] = filter_thinking_content(result['response'])
    print(result['response'])
    json.dump(data,file,indent=2)
'''
'''
with open("/home/23099359d/Enhancing_LLAMA_Against_Language_Jailbreaks/results/implicit_judgment/implicit_contrast/QwQ_32b/race_implicit.json",'r+') as file:
    data = json.load(file)
    result = process(data[75]['AttackPrompt'])
    print(result)
    data[76]['Output'] = filter_thinking_content(result['response'])
    print(result['response'])
    json.dump(data,file,indent = 2)
'''
with open("/home/23099359d/Enhancing_LLAMA_Against_Language_Jailbreaks/results/implicit_judgment/implicit_contrast/QwQ_32b/violence_implicit.json","r+") as file:
    data = json.load(file)
    result = process(data[154]['AttackPrompt'])
    print(result)
    data[154]['Output'] = filter_thinking_content(result['response'])
    print(result['response'])
    json.dump(data,file,indent=2)
