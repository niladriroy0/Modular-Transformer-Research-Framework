# 🔬 Research-Grade Causal Transformer Framework

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

A high-performance, memory-optimized character-level Transformer framework built for rigorous hyperparameter grid searching and causal language modeling. Designed specifically for Kaggle environments, it features deterministic reproducibility, aggressive logging, and efficient dataset slicing.

---

## 🚀 Key Features

* **Causal Masking:** Autoregressive objective using `torch.triu` masks for next-token prediction.
* **Deterministic Execution:** Strict seed-setting across NumPy, PyTorch, and CUDA for 100% reproducible research.
* **Aggressive Grid Search:** Automated `itertools.product` exploration across network depth, width, attention heads, and activation functions.
* **Deep Logging:** Granular batch-level and epoch-level tracking (Loss, Perplexity, LR, Time) exported directly to a tidy CSV format.
* **Memory Optimized:** `CharTokenizer` and `TextDataset` rely on efficient tensor slicing rather than heavy pre-allocation.

---

## 🧠 Architecture Overview

The model employs a standard Decoder-only Transformer architecture, adapted for character-level generation. Below is the interactive data-flow diagram (rendered automatically in Markdown via Mermaid.js).

```mermaid
graph TD
    classDef tensor fill:#f9f,stroke:#333,stroke-width:2px;
    classDef module fill:#bbf,stroke:#333,stroke-width:2px;
    classDef loss fill:#fbb,stroke:#333,stroke-width:2px;

    A[Input Text Stream] --> B(CharTokenizer)
    B --> C[Token IDs: B, Seq_Len]:::tensor
    C --> D(Embedding Layer):::module
    C --> E(Positional Encoding):::module
    
    D --> F((+))
    E --> F
    
    F --> G[Causal Mask generation]:::module
    G --> H{Transformer Encoder Layers <br/> with Causal Mask}:::module
    
    H --> |Depth Scaling| H
    H --> I(LayerNorm):::module
    I --> J(Linear Projection):::module
    
    J --> K[Logits: B, Seq_Len, Vocab]:::tensor
    K --> L(CrossEntropy Loss):::loss
```


