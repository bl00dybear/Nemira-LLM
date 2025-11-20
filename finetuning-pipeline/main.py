import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch.distributed as dist  
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import torch_xla.distributed.parallel_loader as pl
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch_xla.distributed.xla_multiprocessing as xmp 
from pathlib import Path 
import os

from lora_logic import lora_inject
from dataset_text import read_text_dataset,TextDataset
from dataset_qna import read_qa_dataset, InstructionDataset
from train_logic import train
from model_save_logic import save_model,merge_and_save_full_model
from inference_logic import inference_cpu
import logging

class _XlaDeprecationFilter(logging.Filter):
    def filter(self, record):
        m = record.getMessage()
        return ("xrt_world_size() will be removed" not in m
                and "get_ordinal() will be removed" not in m
                and "is_master_ordinal() will be removed" not in m)

logging.getLogger().addFilter(_XlaDeprecationFilter())





def _make_ranked_print():
    orig_print = print
    def ranked_print(*args, **kwargs):
        try:
            rank = xr.global_ordinal()
            world = xr.world_size()
            prefix = f"[worker {rank}/{world}] "
        except Exception:
            prefix = ""
        if args:
            args = (prefix + str(args[0]),) + args[1:]
        else:
            args = (prefix,)
        return orig_print(*args, **kwargs)
    return ranked_print

print = _make_ranked_print()

def _resolve_dataset_paths():
    base_cur = Path(__file__).resolve().parent
    base_sibling = base_cur.parent / "dataset/data"
    text_candidates = [
        base_cur / "text-data.jsonl",
        base_sibling / "text-data.jsonl",
    ]
    qa_candidates = [
        base_cur / "qna-data.json",
        base_sibling / "qna-data.json",
    ]
    def pick(cands):
        for p in cands:
            if p.exists():
                return str(p)
        return None
    text_path = pick(text_candidates)
    qa_path = pick(qa_candidates)
    if text_path is None:
        raise FileNotFoundError(f"Could not find text-data.jsonl in {text_candidates}")
    if qa_path is None:
        raise FileNotFoundError(f"Could not find qna-data.json in {qa_candidates}")
    print(f"[STATUS] Using datasets:\n - text: {text_path}\n - qna : {qa_path}")
    return text_path, qa_path

def _worker_main(index=None):
    if not dist.is_initialized():
        dist.init_process_group('xla', init_method='xla://')

    rank=32
    alpha=32
    per_core_batch = 2
    lr=1e-4
    max_seq_len = 1024
    grad_accum_steps = 4

    text_dataset_path, qa_dataset_path = _resolve_dataset_paths()

    if xm.xla_device_hw(xm.xla_device()) != 'TPU':
        if xr.global_ordinal() == 0:
            print("[!] TPU device not found. Exiting.")
        return

    device = xm.xla_device()
    world_size = xr.world_size()
    ordinal = xr.global_ordinal()
    is_master = (ordinal == 0)
    if is_master:
        print(f"[STATUS] TPU cores: {world_size} | current rank: {ordinal}")

    print("\n[STATUS] Pipeline started.\n")

    tokenizer=AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = max_seq_len

    print("\n[STATUS] Tokenizer initialized.\n")


    model=AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.1",
        dtype=torch.bfloat16,
        device_map=None,
    )
    
    print("\n[STATUS] Model loaded.\n")

    model.config.use_cache = False

    model,lora_layers=lora_inject(model=model,rank=rank,alpha=alpha,device=device)
    
    model=model.to(device)

    ###############################################################################################
    
    print("\n[STATUS] PHASE 1: Training on text completion dataset")

    sample_texts=read_text_dataset(text_dataset_path)

    if is_master:
        print("\n[STATUS] Text processed from file.\n")
        print(f"\n[STATUS] Loaded {len(sample_texts)} samples\n")

    text_dataset=TextDataset(texts=sample_texts,tokenizer=tokenizer,max_len=max_seq_len)
    text_sampler = DistributedSampler(text_dataset, num_replicas=world_size, rank=ordinal, shuffle=True, drop_last=False)
    text_loader = DataLoader(
        text_dataset,
        batch_size=per_core_batch,
        sampler=text_sampler,
        num_workers=0,
        drop_last=False
    )
    text_loader = pl.MpDeviceLoader(text_loader, device, loader_prefetch_size=8)

    if is_master:
        print("\n[STATUS] Dataloader configured\n")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, betas=(0.9, 0.95), weight_decay=0.01)

    if is_master:
        print("\n[STATUS] Starting training process.\n")

    text_sampler.set_epoch(0)

    steps_phase1 = (len(text_loader) // grad_accum_steps) * 2 
    scheduler1 = CosineAnnealingLR(optimizer, T_max=steps_phase1, eta_min=1e-6)

    train(model=model, dataloader=text_loader, optimizer=optimizer, device=device,
          epochs=2, grad_accum_steps=grad_accum_steps,scheduler=scheduler1)

    ###############################################################################################

    print("\n[STATUS] PHASE 2: Training on Q&A instruction dataset")
    qa_data = read_qa_dataset(qa_dataset_path)

    if is_master:
        print(f"[STATUS] Loaded {len(qa_data)} Q&A samples\n")

    qa_dataset = InstructionDataset(data=qa_data, tokenizer=tokenizer,max_len=max_seq_len,min_resp_tokens=16)
    qa_sampler = DistributedSampler(qa_dataset, num_replicas=world_size, rank=ordinal, shuffle=True, drop_last=False)
    qa_loader = DataLoader(
        qa_dataset,
        batch_size=per_core_batch,
        sampler=qa_sampler,
        num_workers=0,
        drop_last=False
    )
    qa_loader = pl.MpDeviceLoader(qa_loader, device, loader_prefetch_size=8)

    print("[STATUS] Starting Q&A training\n")
    qa_sampler.set_epoch(0)

    steps_phase2 = (len(qa_loader) // grad_accum_steps) *1
    scheduler2 = CosineAnnealingLR(optimizer, T_max=steps_phase2, eta_min=1e-6)

    train(model=model, dataloader=qa_loader, optimizer=optimizer, device=device,
          epochs=1, grad_accum_steps=grad_accum_steps, scheduler=scheduler2)
    ###############################################################################################
    

    print("\n[STATUS] Saving model\n")
    save_model(lora_layers=lora_layers, tokenizer=tokenizer)

    print("\n[STATUS] Saving full merged model\n")
    merge_and_save_full_model(model, lora_layers, tokenizer, output_dir="./merged_model")

    if is_master:
        print("\n[STATUS] Inference tests\n")
        inference_cpu(model, tokenizer)
    xm.rendezvous("inference_done")


if __name__ == "__main__":
    procs = int(os.environ.get("PJRT_LOCAL_PROCESS_COUNT", "1"))
    if procs <= 1:
        print("[STATUS] Warning: PJRT_LOCAL_PROCESS_COUNT not set; running single process (1 core).")
    xmp.spawn(_worker_main, args=(), nprocs=procs)

#  PJRT_DEVICE=TPU PJRT_LOCAL_PROCESS_COUNT=8 python3 main.py
