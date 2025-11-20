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

        # W ~ N(0, sqrt(2/n)) - kaiming distribution
        # W ~ U(-bound, bound) where bound = sqrt(6 / ((1 + a²) * fan_in)) - uniform kaiming distribution
        # fan_in is the number of inputs of the layer Q
        # developed to avoid gradient vanishing and gradient exploding, optimized for ReLU

        nn.init.kaiming_uniform_(tensor=self.A.weight,mode='fan_in',a=5**0.5)
        nn.init.zeros_(self.B.weight)


    # forward with lora works like this: y = Wx + α/r*B(Ax) (see /doc/LoRA_ro.pdf)
    def forward(self, orig_output,tensor):
        # tensor shape: (batch, seq_len, in_feat) or (batch*seq_len, in_feat)
        # A shape: (in_feat, rank)
        # B shape: (rank, out_feat)
        lora_out = self.B(self.A(tensor))*self.scaling
        return orig_output+lora_out
    



def lora_inject(model,rank,alpha,device):
    dtype=next(model.parameters()).dtype

    # loop that freezes the original parameters of the model, because we don't want them to be trained
    for param in model.parameters():
        param.requires_grad = False

    # see /doc/mistral7b.md to see where these modules are, and we add lora to them because
    # these are the layers that learn. priority are the modules from self attention and if we have enough resources
    # we add lora to MLP too (it can work without MLP for computational efficiency, but there's a performance trade-off)
    att_modules=["q_proj","v_proj","k_proj","o_proj","gate_proj"]
    mlp_modules=["gate_proj","up_proj","down_proj"]
    lora_layers={}
    layer_index=0

    # /doc/mistral7b.md <-- there you have the model structure, necessary to understand this traversal of layers and modules
    # lora_layers exists because we want to save the weights after training and they are saved as dictionary
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