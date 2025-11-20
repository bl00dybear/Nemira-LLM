import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    def __init__(self, in_feat,out_feat,rank=8,alpha=16,dtype=torch.bfloat16,device=None):
        super().__init__()

        self.rank=rank
        self.alpha=alpha
        self.scaling=alpha/rank
        self.dtype=dtype
        self.device=device
        self.A = nn.Linear(in_feat, rank, bias=False).to(dtype=self.dtype, device=device)
        self.B = nn.Linear(rank, out_feat, bias=False).to(dtype=self.dtype, device=device)

        # W ~ N(0, sqrt(2/n)) - distributia kaiming
        # W ~ U(-bound, bound) unde bound = sqrt(6 / ((1 + a²) * fan_in)) - distributia uniforma kaiming
        # fan_in e nr de inputuri al layerului  Q
        # dezvoltata pentru a evita gradient vanishing si gradient exploding, optimizata pentu ReLU

        nn.init.kaiming_uniform_(tensor=self.A.weight,mode='fan_in',a=5**0.5)
        nn.init.zeros_(self.B.weight)


    # forwardul cu lora functioneaza asa: y = Wx + α/r*B(Ax) (vezi /doc/LoRA.pdf)
    def forward(self, orig_output,tensor):
        # tensor=tensor.to(self.dtype)  nu mai e nevoie deoarece am mutat A si B pe dtype-ul corect la initializare

        # tensor shape: (batch, seq_len, in_feat) or (batch*seq_len, in_feat)
        # A shape: (in_feat, rank)
        # B shape: (rank, out_feat)
        lora_out = self.B(self.A(tensor))*self.scaling
        return orig_output+lora_out
    



def lora_inject(model,rank,alpha,device):
    dtype=next(model.parameters()).dtype

    # loop care da freeze la parametrii originali ai modelului, deoarece nu vrem sa fie antrenati
    for param in model.parameters():
        param.requires_grad = False

    # vezi /doc/mistral7b.md pentru a vedea unde sunt modulele astea, si pe ele adaugam lora deoarece
    # astea sunt layerele care invata. prioritar sunt modulele din self attention si daca ne permit resursele
    # adaugam lora si pe MLP (se poate si fara MLP pentru eficienta computationala, dar e trade-off pe performanta)
    att_modules=["q_proj","v_proj","k_proj","o_proj","gate_proj"]
    mlp_modules=["gate_proj","up_proj","down_proj"]
    lora_layers={}
    layer_index=0

    # /doc/mistral7b.md <-- acolo ai structura modelului, necesara pt a intelege parcurgerea asta de layere si module
    # lora_layers exista deoarece vrem sa salvam weight urile de dupa antrenament si ele se salveaza dictionar
    for layer in model.model.layers:
        for layer_module in att_modules:
            if hasattr(layer.self_attn,layer_module):
                module=getattr(layer.self_attn,layer_module)

                if isinstance(module,nn.Linear):
                    in_feat=module.in_features
                    out_feat=module.out_features

                lora_layer = LoRALayer(in_feat, out_feat, rank=rank, alpha=alpha, dtype=dtype,device=device)
                layer_name = f"lora_att_{layer_index}_{layer_module}"

                setattr(module, "lora_adapter", lora_layer)
                lora_layers[layer_name] = lora_layer
 
                def new_forward(x, orig_forward=module.forward, adapter=lora_layer):
                    return adapter(orig_forward(x), x)
                module.forward = new_forward

        if hasattr(layer, "mlp"):
            for layer_module in mlp_modules:
                if hasattr(layer.mlp, layer_module):
                    module = getattr(layer.mlp, layer_module)
                    if isinstance(module, nn.Linear):
                        in_feat, out_feat = module.in_features, module.out_features
                    lora_layer = LoRALayer(in_feat, out_feat, rank=rank, alpha=alpha, dtype=dtype, device=device)
                    layer_name = f"lora_mlp_{layer_index}_{layer_module}"
                    setattr(module, "lora_adapter", lora_layer)
                    lora_layers[layer_name] = lora_layer

                    def new_forward(x, orig_forward=module.forward, adapter=lora_layer):
                        return adapter(orig_forward(x), x)
                    module.forward = new_forward


        layer_index+=1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nLoRA injected: {len(lora_layers)} adapters")
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model, lora_layers