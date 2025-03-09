import json
from llamaapi import LlamaAPI

# Initialize the Llama API
llama = LlamaAPI("")

def process_attack_prompt(attack_prompt):
    print("="*100 + "\n")
    print(f"Input: {attack_prompt}\n")  
    
    api_request_json = {
        "model": "llama3.3-70b", 
        "messages": [
            {"role": "system", "content": "Answer the question."},
            {"role": "user", "content": attack_prompt},
        ],
        "stream": False,
        "max_tokens": 1500  # Increase token limit to allow for longer outputs
    }

    try:
        response = llama.run(api_request_json)
        response_data = response.json()
        #print(f"Response data: {json.dumps(response_data, indent=4)}")  # Print full response to check structure
        content = response_data["choices"][0]["message"]["content"]
        print(f"Output:\n{content}\n")  
        return content
    except Exception as e:
        print(f"Error processing prompt: {e}")
        return None


output_data = []  
output_file = "./results.json"

# Open and process the JSON file
with open("./generated_cases.json", "r", encoding="utf-8") as file:
    try:
        # Load JSON content
        data = json.load(file)
        print(f"Successfully loaded JSON file with {len(data)} entries.\n") 
        
        for index, item in enumerate(data, start=1):
            attack_input = item.get("Attack Prompt")
            attack_id = item.get("id")
            
            if attack_input:
                print(f"Processing entry {index} (ID: {attack_id})...")
                output_content = process_attack_prompt(attack_input)
                
                if output_content:
                    # Save the result
                    output_data.append({
                        "id": attack_id,
                        "AttackPrompt": attack_input,
                        "Output": output_content
                    })
            else:
                print(f"Skipping entry {index} due to missing 'Attack Prompt'.")

    except json.JSONDecodeError:
        print("Error: Invalid JSON format.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Write the processed results to an output file
with open(output_file, "w", encoding="utf-8") as output:
    json.dump(output_data, output, ensure_ascii=False, indent=4)

print(f"\nProcessing complete. Results saved to {output_file}.")
