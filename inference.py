import argparse
import sys
import torch

def load_bitsandbytes_config(mode: str):
    if mode == "none":
        return None
    try:
        from transformers import BitsAndBytesConfig
    except Exception:
        print("[WARN] bitsandbytes/transformers quantization not available. Falling back to FP16.", file=sys.stderr)
        return None
    if mode == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if mode == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    return None

def load_model_and_tokenizer(model_dir: str, quant: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    quant_config = load_bitsandbytes_config(quant)
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = {"device_map": "auto"}
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
    else:
        kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_dir, low_cpu_mem_usage=True, **kwargs)
    model.eval()
    return tok, model

def build_inputs(tokenizer, question: str):
    # Întoarce mereu un dict (BatchEncoding)
    if getattr(tokenizer, "chat_template", None):
        # Rendăm la text și apoi tokenizăm => compatibil cu toate versiunile HF
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            add_generation_prompt=True,
            tokenize=False,
        )
        return tokenizer(rendered, return_tensors="pt")
    return tokenizer(question, return_tensors="pt")

@torch.inference_mode()
def generate_answer(model, tokenizer, question: str, max_new_tokens=256, do_sample=False, temperature=0.7, top_p=0.9):
    inputs = build_inputs(tokenizer, question)

    # Unele versiuni pot întoarce un Tensor; normalizăm la dict
    if isinstance(inputs, torch.Tensor):
        inputs = {"input_ids": inputs}

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        no_repeat_ngram_size=3,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = gen_ids[0, prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=str, default="./model/merged_model", help="Folderul modelului salvat (merged_model)")
    ap.add_argument("--quantization", type=str, choices=["none", "8bit", "4bit"], default="8bit")
    ap.add_argument("--question", type=str, default=None, help="Întrebarea direct din CLI; dacă lipsește, intră în modul interactiv")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--sample", action="store_true", help="Activează sampling (temperature/top-p)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    args = ap.parse_args()

    tok, model = load_model_and_tokenizer(args.model_dir, args.quantization)

    if args.question:
        ans = generate_answer(model, tok, args.question, max_new_tokens=args.max_new_tokens,
                              do_sample=args.sample, temperature=args.temperature, top_p=args.top_p)
        print(ans)
        return

    print("Inferență interactivă. Tastează 'exit' pentru a ieși.\n")
    while True:
        try:
            q = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() in ("exit", "quit"):
            break
        ans = generate_answer(model, tok, q, max_new_tokens=args.max_new_tokens,
                              do_sample=args.sample, temperature=args.temperature, top_p=args.top_p)
        print(ans, flush=True)

if __name__ == "__main__":
    main()