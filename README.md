# Fine-Tuning-LLMs-Using-LoRA-and-QLoRA
## Project Overview
This repository contains two Google Colab notebooks for fine-tuning large language models (LLMs) using:
- LoRA (Low-Rank Adaptation)
- QLoRA (Quantized Low-Rank Adaptation)

Both approaches enable efficient training of LLMs on limited hardware (e.g., Google Colab GPUs).
---
### Notebooks

| Notebook                     | Description                                                          |
| ---------------------------- | -------------------------------------------------------------------- |
| `lora_finetune_colab.ipynb`  | Fine-tunes an LLM using LoRA with full precision or 8-bit weights    |
| `qlora_finetune_colab.ipynb` | Fine-tunes a quantized (4-bit) LLM using QLoRA for memory efficiency |

---

### Requirements

* Google Colab (Pro recommended for longer runtimes and more GPU memory)
* Hugging Face account and token
* Model permission from Hugging Face (e.g., for `meta-llama`, `mistral`, etc.)

---

### Hugging Face token setup

Save your token securely in Colab using:

```python
from huggingface_hub import login
login(token="your_hf_token")
```

Or use Colab’s secret manager for safer handling.

---

### Features covered

* Loading pretrained models with `transformers`
* Model quantization (8-bit for LoRA, 4-bit for QLoRA)
* Parameter-efficient fine-tuning using `PEFT`
* Saving and loading fine-tuned models
* Inference using `AutoPeftModelForCausalLM` (QLoRA) or `AutoModelForCausalLM` (LoRA)

---

### Key differences: LoRA vs QLoRA

| **Aspect**                        | **LoRA (Low-Rank Adaptation)**                                                          | **QLoRA (Quantized LoRA)**                                                                                           |
| --------------------------------- | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Base Model Weights**            | Loaded in **FP16** or **8-bit** (with bitsandbytes)                                     | Loaded in **4-bit NF4 (NormalFloat)** quantized format                                                               |
| **Frozen During Training?**       | **Yes** base weights are **not updated** during training                                | **Yes** — same; quantized base is also **frozen**                                                                  |
| **Why Freeze Base Weights?**      | Saves compute/memory and avoids catastrophic forgetting                                 | Same — especially necessary due to 4-bit quantization                                                                |
| **Where Are Weights Stored?**     | Full-precision model (e.g. 7B) is stored on GPU in **16-bit or 8-bit**                  | Quantized model is stored on GPU in **4-bit NF4**, requiring **\~3–4× less memory**                                  |
| **Precision Format Used**         | FP16 (float16) or optional INT8 (with `bnb`)                                            | NF4 (NormalFloat 4-bit), a custom quantization format introduced by Hugging Face in the QLoRA paper                  |
| **Adapter Location**              | LoRA adapters inserted at specific layers (usually attention/MLP)                       | Same — LoRA adapters are inserted in the **same locations**, only on top of 4-bit base                               |
| **Computation Precision**         | Uses **FP16** for forward/backward passes (efficient on A100/T4)                        | Uses **FP16 or bfloat16 (BF16)** for **matrix computation**, while base is stored in 4-bit                           |
| **Quantization Type**             | Optional INT8 via `bitsandbytes`, not required                                          | **Required 4-bit quantization** via `bnb.QuantLinear` layers + **double quantization** (for better compression)      |
| **Quantization Format (Details)** | INT8, symmetric, round-to-nearest                                                       | NF4 (non-linear float distribution), + double quantization (a second layer of quantizing the quantization constants) |
| **Training Target**               | Only train the **LoRA adapter layers**, nothing else                                    | Same — only adapter layers are trained; base model remains static                                                    |
| **Memory Footprint (7B model)**   | \~12–16 GB (LoRA with 16-bit base); \~10 GB (LoRA + 8-bit)                              | \~5–7 GB with 4-bit quantized base + LoRA adapters (depends on rank and tokenizer)                                   |
| **Model Loading Method**          | `AutoModelForCausalLM.from_pretrained(..., load_in_8bit=True)` or `load_in_16bit=True`  | `AutoModelForCausalLM.from_pretrained(..., load_in_4bit=True, quantization_config=...)`                              |
| **Compatibility with PEFT**       | Fully supported via Hugging Face `peft.LoRAConfig`                                      | Supported via Hugging Face `peft.LoraConfig` + `bnb.4bit` + `transformers>=4.33.0`                                   |
| **Inference Usage**               | Use `AutoModelForCausalLM` for LoRA, `AutoPeftModelForCausalLM` optional                | Must use `AutoPeftModelForCausalLM` for inference if saved as PEFT adapter + 4-bit base                              |
| **Saving the Model**              | Save LoRA adapter only (`save_pretrained()`), or merge with base (`merge_and_unload()`) | Save LoRA adapter and reference to quantized base; can also merge and push to Hugging Face                           |
| **Supported Hardware**            | Colab T4, A100, RTX 3090, 4090 (needs \~12 GB+)                                         | Even runs on 6–8 GB GPUs (e.g., RTX 3060, consumer laptops with GPU, Colab free tier sometimes works)                |
| **Best for**                      | Faster training, quick prototyping, high accuracy                                       | Large-model fine-tuning on low-memory setups; memory-constrained deployment                                          |


---

### Example tasks

* Instruction tuning
* Domain-specific chatbot adaptation
* Dataset examples (can be customized by the user)

---

### How to use

1. Open the notebook in Colab.
2. Connect to GPU runtime.
3. Run cells sequentially.
4. Save your fine-tuned model to Hugging Face or download locally.

