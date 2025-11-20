# Mistral 7B Romanian Fine-Tuning on Google TPU v6e (PyTorch only)

This repository contains the experimental setup and training pipeline for fine-tuning **Mistral 7B** to better understand and generate the Romanian language.


> **Status: Work in Progress (WIP)**
> The training pipeline is fully functional and optimized for TPU v6e-8 using `torch_xla`. However, due to time constraints within the compute allocation window, the model has not yet reached full convergence. This repo serves as a reference implementation for setting up the environment and running the training loop.

## Acknowledgments

**Research supported with Cloud TPUs from Google's TPU Research Cloud (TRC).**

We want to thank the Google TRC team for providing access to the **TPU v6e-8** hardware, which made this research possible.

## Technical Stack
### Finetuning:
* **Hardware:** Google Cloud TPU v6e-8 (Trillium), `europe-west4-a` zone.
* **Framework:** PyTorch 2.5.0 & PyTorch XLA (PJRT runtime).
* **Model:** Mistral 7B (LoRA adaptation).

### Inference:
* **Hardware:** Nvidia RTX 3080 10Gb
* **Framework:**  .
* **Model:** Mistral 7B (quantized LoRA adaptation).

## Repository Structure & Automation

We have automated most of the provisioning and setup process using Bash scripts to ensure reproducibility.

### 1. TPU Provisioning (`utils/gcloud_tpuvm.sh`)

**Script logic:**
1.  Creates the TPU VM (`v6e-8`), spot instance in zone `europe-west4-a`.
2.  Waits for initialization.
3.  SSHs into the machine.
4.  Offers an option to auto-delete the VM upon exit to avoid unnecessary usage of resources.

### 2. Environment Setup (`utils/cloud-clone-private.sh`)
Once inside the VM, this script handles the dependencies. I utilized `git sparse-checkout` to only pull the necessary training folder, keeping the environment clean.

**Script logic:**
1.  Generates SSH keys for GitHub authentication (only for private respository clone).
2.  Initializes a sparse checkout of the repo.
3.  Installs **PyTorch 2.5.0** and **Torch XLA** from the `libtpu` releases (specific version required for v6e).
4.  Installs `transformers` and `datasets`.
5.  Verifies if the XLA devices are visible to PyTorch.

> Note: For public repositories we provid `utils/cloud-clone-public.sh`

## How to Run

### Step 1: Launch the TPU
On your local machine (with Google Cloud SDK installed), run:

```bash
chmod +x gcloud_tpuvm.sh
./gcloud_tpuvm.sh
```
At this point you wait until TPU-VM is initialized and have a shell. 
In TPU-VM terminal:
``` bash
touch cloud-clone.sh
nano cloud-clone.sh 
# here u paste the clone script according to your case
# (cloning a public or private repo)
chmod +x cloud-clone.sh
./cloud-clone.sh
```
### Step 2: Download datasets

Now you have a directory `llm-finetuning` containing last commit of my repository.

For downloading our datasets use: 
```bash
cd llm-finetuning/dataset/scripts
python3 download_qna_dataset.py
python3 download_text_dataset.py
```
This 2 scripts will download 2 romanian datasets, one with raw phrases from `Wikipedia` and one with .the dataset `OpenLLM-Ro/ro_sft_ultrachat`, a conversational dataset. [Datasets format](./docs/dataset_scope.md)


### Step 3: Run finetuning pipeline:
```bash
cd ../../finetuning-pipeline
PJRT_DEVICE=TPU PJRT_LOCAL_PROCESS_COUNT=8 python3 main.py
```


## Technical Highlights

### Custom Low-Rank Adaptation (LoRA) Implementation
**File:** `lora_logic.py`

Instead of utilizing high-level abstraction libraries such as PEFT, this project features a first-principles implementation of LoRA to demonstrate a deep understanding of parameter-efficient fine-tuning mathematics.
* **Matrix Decomposition:** Implements the core update rule $W = W_0 + \frac{\alpha}{r}BA$ directly in PyTorch.
* **Dynamic Injection:** Utilizes Python metaprogramming to dynamically inject adapter layers into specific Mistral 7B target modules (`q_proj`,`v_proj`,`k_proj`,`o_proj`,`gate_proj`,`up_proj`,`down_proj`) at runtime, keeping the base model weights frozen.
* **Initialization Strategy:** Applies Kaiming Uniform initialization to the adapter matrices to ensure gradient stability during the early phases of training.

### PyTorch XLA Optimization for TPU
**File:** `train_logic.py`

The training loop is engineered specifically for the XLA (Accelerated Linear Algebra) compilation paradigm used by Cloud TPUs, distinct from standard GPU execution.
* **Lazy Tensor Execution:** Manually manages execution barriers via `xm.mark_step()` to control the construction and execution of the XLA graph. This optimization minimizes graph recompilations and maximizes throughput.
* **Parallelism:** Implements data parallelism across TPU cores using the PJRT runtime, ensuring efficient synchronization of gradients.
* **Gradient Accumulation:** Simulates larger effective batch sizes to aid model convergence while operating within the High Bandwidth Memory (HBM) constraints of individual TPU cores.

### Training Strategy & Data Pipeline
**File:** `main.py`

* **Multi-Stage Fine-Tuning (Curriculum Learning):** Implemented a dual-phase training regimen. **Phase 1** focuses on Domain Adaptation using raw text completion to align the model with Romanian syntax, followed by **Phase 2** (Instruction Tuning) using Q&A pairs to enforce chat behaviors.
* **High-Throughput Data Loading:** Utilized `torch_xla.distributed.parallel_loader.MpDeviceLoader` with aggressive prefetching (`loader_prefetch_size=8`). This minimizes TPU idle time by asynchronously moving data tensors to the XLA device while the computation runs, solving the CPU-bound bottleneck common in high-performance training.


## Results

As a result, we have successfully established a **fully functional, end-to-end fine-tuning pipeline** on Google's cutting-edge **TPU v6e (Trillium)** architecture.

Key achievements include:
1.  **Infrastructure Validation:** Validated that pure PyTorch workflows (via `torch_xla`) are viable on the latest generation of Cloud TPUs, achieving stable training throughput without relying on high-level frameworks like JAX or TensorFlow.
2.  **Mathematical Verification:** The custom `lora_logic` implementation correctly computes gradients and updates weights across distributed cores, proving that low-level adapter injection works seamlessly within the XLA graph compilation constraint.
3.  **End-to-End Workflow:** The system successfully bridges the gap between cloud-native training (TPU/BFloat16) and consumer-grade inference (GPU/8-bit quantization), producing a deployable model capable of coherent Romanian text generation.

While the model requires extended compute time for full convergence, the **engineering pipeline itself is proven stable, reproducible, and scalable**.

## Future work
#### With the very recent acquisition of two high-quality datasets, the immediate next step is to leverage this validated pipeline for a complete training run to achieve full model convergence.

## Contributors

This research was a collaborative effort between:

* Șincari Sebastian George - [Github](https://github.com/bl00dybear)  [Linkedin](https://www.linkedin.com/in/sebastian-george-sincari/)
* Gheorghe Bogdan Alexandru - [Github](https://github.com/ghrghbogdan)  [Linkedin](https://www.linkedin.com/in/bogdan-alexandru-gheorghe/)