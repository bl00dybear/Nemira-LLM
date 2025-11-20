# Mistral 7b

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                          MISTRAL-7B MODEL                               │
│                     (7 Billion Parameters)                              │
└─────────────────────────────────────────────────────────────────────────┘

INPUT: "What is AI?" → Tokenizer → [1234, 5678, 9012, 2]
                                         ↓
                            ┌────────────────────────────┐
                            │   Embedding Layer          │
                            │   vocab_size: 32,000       │
                            │   hidden_dim: 4,096        │
                            └────────────────────────────┘
                                         ↓
    ┌───────────────────────────────────────────────────────────────────────┐
    │                    32 × TRANSFORMER LAYERS                            │
    ├───────────────────────────────────────────────────────────────────────┤
    │                                                                       │
    │  ┌───────────────────────────────────────────────────────────────┐    │
    │  │  Layer 0                                                      │    │
    │  ├───────────────────────────────────────────────────────────────┤    │
    │  │                                                               │    │
    │  │  ┌─────────────────────────────────────────────────────────┐  │    │
    │  │  │  Layer Norm                                             │  │    │
    │  │  └─────────────────────────────────────────────────────────┘  │    │
    │  │                    ↓                                          │    │
    │  │  ┌─────────────────────────────────────────────────────────┐  │    │
    │  │  │  SELF-ATTENTION (Grouped Query Attention)               │  │    │
    │  │  ├─────────────────────────────────────────────────────────┤  │    │
    │  │  │                                                         │  │    │
    │  │  │  Input (4096)                                           │  │    │
    │  │  │     ├──→ q_proj ← LoRA(4096→4096) ──→ Q (32 heads)      │  │    │
    │  │  │     ├──→ k_proj ← LoRA(4096→1024) ──→ K (8 heads)       │  │    │
    │  │  │     └──→ v_proj ← LoRA(4096→1024) ──→ V (8 heads)       │  │    │
    │  │  │                                                         │  │    │
    │  │  │  Q @ K^T / √d → Softmax → × V                           │  │    │
    │  │  │                    ↓                                    │  │    │
    │  │  │  Attention Output → o_proj ← LoRA(4096→4096)            │  │    │
    │  │  │                                                         │  │    │
    │  │  └─────────────────────────────────────────────────────────┘  │    │
    │  │                    ↓                                          │    │
    │  │              Residual Connection (+)                          │    │
    │  │                    ↓                                          │    │
    │  │  ┌────────────────────────────────────────────────────────┐   │    │
    │  │  │  Layer Norm                                            │   │    │
    │  │  └────────────────────────────────────────────────────────┘   │    │
    │  │                    ↓                                          │    │
    │  │  ┌─────────────────────────────────────────────────────────┐  │    │
    │  │  │  MLP (Feed-Forward Network)                             │  │    │
    │  │  ├─────────────────────────────────────────────────────────┤  │    │
    │  │  │                                                         │  │    │
    │  │  │  gate_proj:  4096 → 14,336  (SiLU activation)           │  │    │
    │  │  │  up_proj:    4096 → 14,336                              │  │    │
    │  │  │                   ↓                                     │  │    │
    │  │  │  SiLU(gate) * up  (element-wise)                        │  │    │
    │  │  │                   ↓                                     │  │    │
    │  │  │  down_proj:  14,336 → 4096                              │  │    │
    │  │  │                                                         │  │    │
    │  │  └─────────────────────────────────────────────────────────┘  │    │
    │  │                    ↓                                          │    │
    │  │              Residual Connection (+)                          │    │
    │  │                    ↓                                          │    │
    │  └───────────────────────────────────────────────────────────────┘    │
    │                                                                       │
    │  ┌─────────────────────────────────────────────────────────────────┐  │
    │  │  Layer 1                                                        │  │
    │  │  [Same structure as Layer 0]                                    │  │
    │  └─────────────────────────────────────────────────────────────────┘  │
    │                            ...                                        │
    │  ┌─────────────────────────────────────────────────────────────────┐  │
    │  │  Layer 31                                                       │  │
    │  │  [Same structure as Layer 0]                                    │  │
    │  └─────────────────────────────────────────────────────────────────┘  │
    │                                                                       │
    └───────────────────────────────────────────────────────────────────────┘
                                         ↓
                        ┌────────────────────────────┐
                        │   Final Layer Norm         │
                        └────────────────────────────┘
                                         ↓
                        ┌────────────────────────────┐
                        │   LM Head (Linear)         │
                        │   4096 → 32,000            │
                        │   (vocab projection)       │
                        └────────────────────────────┘
                                         ↓
                              Softmax (logits)
                                         ↓
                        ┌────────────────────────────┐
                        │   Token Probabilities      │
                        │   [0.001, 0.05, 0.94, ...] │
                        └────────────────────────────┘
                                         ↓
                           Next Token: "artificial"

```
## Key specs
```text
┌─────────────────────────────────────────┐                     Layer 0-31 (32 layers)
│  MISTRAL-7B PARAMETERS                  │                       ├─ q_proj ← LoRA adapter (4096 × rank)
├─────────────────────────────────────────┤                       ├─ k_proj ← LoRA adapter (4096 × rank)  
│  Total Parameters:      ~7.24B          │                       ├─ v_proj ← LoRA adapter (4096 × rank)
│  Hidden Dimension:      4,096           │                       └─ o_proj ← LoRA adapter (4096 × rank)
│  Num Layers:            32              │
│  Num Attention Heads:   32 (Q)          │                     Total: 32 layers × 4 modules = 128 LoRA adapters
│  Num KV Heads:          8 (GQA)         │
│  MLP Hidden:            14,336          │                     With rank=32:
│  Vocab Size:            32,000          │                       Trainable params ≈ 32 × 4 × (4096×32 + 32×4096) × 2
│  Context Length:        8,192 tokens    │                                         ≈ 33.5M parameters (~0.46% of total)
│  Sliding Window:        4,096 tokens    │
└─────────────────────────────────────────┘
```

## Locatia fizica in model:
```py
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

# embedding layer 
model.model.embed_tokens  # vocab → Hidden dim (32000 → 4096)

# 32 transformer layers
model.model.layers[0]     # layer 0
model.model.layers[1]     # layer 1
...
model.model.layers[31]    # layer 31

# in each layer
layer = model.model.layers[0]

# 1) SELF-ATTENTION MODULE
layer.self_attn.q_proj    # nn.Linear(4096, 4096) - Query projection
layer.self_attn.k_proj    # nn.Linear(4096, 1024) - Key projection (GQA)
layer.self_attn.v_proj    # nn.Linear(4096, 1024) - Value projection (GQA)
layer.self_attn.o_proj    # nn.Linear(4096, 4096) - Output projection

# 2) MLP (FEED-FORWARD) MODULE
layer.mlp.gate_proj       # nn.Linear(4096, 14336)
layer.mlp.up_proj         # nn.Linear(4096, 14336)
layer.mlp.down_proj       # nn.Linear(14336, 4096)

# 3)LAYER NORMS
layer.input_layernorm     # normalizare before attention
layer.post_attention_layernorm  # normalizare before MLP

# Final layers (la ieșire)
model.model.norm          # final layer norm
model.lm_head             # Linear(4096, 32000) - vocab projection
```