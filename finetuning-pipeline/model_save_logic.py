import torch
import os
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr


def save_model(lora_layers,tokenizer):
    if xr.global_ordinal() == 0:
        lora_state = {name: layer.state_dict() for name, layer in lora_layers.items()}
        torch.save(lora_state, "lora_weights.pt")
        tokenizer.save_pretrained("./model_output")
    xm.rendezvous("save_model_checkpoint")


def merge_and_save_full_model(model, lora_layers, tokenizer, output_dir="./merged_model"):
    if xr.global_ordinal() == 0:
        os.makedirs(output_dir, exist_ok=True)

        print("\n[STATUS] Merging LoRA weights into base model...")

        lora_modules = {}
        for layer_idx in range(len(model.model.layers)):
            layer = model.model.layers[layer_idx]
            for module_name in ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
                if hasattr(layer.self_attn, module_name):
                    module = getattr(layer.self_attn, module_name)

                    lora_name = f"lora_layer_{layer_idx}_{module_name}"
                    if lora_name in lora_layers:
                        lora_modules[(layer_idx, module_name)] = (module, lora_layers[lora_name])

        with torch.no_grad():
            for (layer_idx, module_name), (module, lora_layer) in lora_modules.items():
                A = lora_layer.A.cpu() 
                B = lora_layer.B.cpu()  

                delta_W = (A @ B) * lora_layer.scaling
                
                original_device = module.weight.device
                module_weight_cpu = module.weight.cpu()

                module_weight_cpu.data += delta_W.T.to(module_weight_cpu.dtype)
                
                module.weight.data = module_weight_cpu.to(original_device)


        for lora_name in lora_layers.keys():
            if hasattr(model, lora_name):
                delattr(model, lora_name)

        for layer in model.model.layers:
            for module_name in ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
                if hasattr(layer.self_attn, module_name):
                    module = getattr(layer.self_attn, module_name)
                    if hasattr(module, '__class__'):
                        module.forward = nn.Linear.forward.__get__(module, nn.Linear)

        print("\n[STATUS] Saving merged model...")

        model = model.cpu()
        
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        print(f"\n[STATUS] Full model saved in {output_dir}")
    
    xm.rendezvous("merge_and_save_full_model_checkpoint")
    return model