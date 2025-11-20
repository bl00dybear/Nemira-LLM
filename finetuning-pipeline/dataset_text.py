from torch.utils.data import Dataset
import json


class TextDataset(Dataset):
    def __init__(self,texts,tokenizer,max_len=512):
        self.tokenizer=tokenizer
        self.max_len=max_len
        self.texts=texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )

        #(1, 512) → (512,)
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        # INPUT:  "Când sunt admiterile la FMI ?"
        # TARGET: "sunt admiterile la FMI ? [EOS]"
        # (modelul învață să prezică următorul token)

        labels[attention_mask == 0] = -100
        # Setăm -100 pentru tokenii de padding
        # PyTorch CrossEntropyLoss IGNORĂ automat label-urile cu valoarea -100
        # Astfel modelul nu învață din padding!

        # input_ids:      [150, 2341, 15678, ..., 0,    0,    0   ]
        # attention_mask: [1,   1,    1,     ..., 0,    0,    0   ]
        # labels:         [150, 2341, 15678, ..., -100, -100, -100]
        #                  ↑    ↑     ↑           ↑     ↑     ↑

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# Format
# [
#     {
#         "messages": [
#             {"role": "user", "content": "Când sunt admiterile?"},
#             {"role": "assistant", "content": "Admiterile sunt în iulie."}
#         ]
#     },
#     ...
# ]



def read_text_dataset(text_dataset_path):
    sample_texts = []
    bad_lines = 0
    with open(text_dataset_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:  # skip linii goale
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            text = obj.get("text")
            if isinstance(text, str) and text.strip():
                sample_texts.append(text)
            else:
                bad_lines += 1
    if not sample_texts:
        raise ValueError(f"No valid records found in {text_dataset_path}. Bad lines: {bad_lines}")
    return sample_texts