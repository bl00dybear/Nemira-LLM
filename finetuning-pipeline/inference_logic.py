import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr


def inference_cpu(model,tokenizer):
    if xr.global_ordinal() != 0:
        return
    
    model.eval()
    model = model.cpu()

    with torch.no_grad():
        prompt = "What is artificial intelligence?"
        inputs = tokenizer(prompt, return_tensors="pt")

        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text}")