import json
from datasets import load_dataset
from tqdm import tqdm

def format_dataset():
    dataset_name = "OpenLLM-Ro/ro_sft_ultrachat"
    output_filename = "../data/qna-data.json"

    print(f"[STATUS] Streaming started")
    
    try:
        dataset = load_dataset(dataset_name, split="train", streaming=True)
    except Exception as e:
        print(f"Eroare la conexiune/încărcare: {e}")
        return

    formatted_data = []
    

    for row in tqdm(dataset):
        original_messages = row.get('messages') or row.get('conversations')
        
        if not original_messages:
            continue

        new_messages = []
        
        for msg in original_messages:
            role = msg.get('role')
            content = msg.get('content')

            if role == 'human':
                role = 'user'
            elif role in ['gpt', 'bot']:
                role = 'assistant'

            new_messages.append({
                "role": role,
                "content": content
            })

        formatted_data.append({
            "messages": new_messages
        })

    print(f"[STATUS] SAVED {len(formatted_data)} conversations {output_filename}...")
    
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)

    print("✅ Gata! Fișierul a fost generat cu succes.")

if __name__ == "__main__":
    format_dataset()