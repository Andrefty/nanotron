---
library_name: nanotron
---

# Mamba

Modeling code for Mamba to use with [Nanotron](https://github.com/huggingface/nanotron/)

## 🚀 Quickstart

```bash
pip install torch==2.1.0
pip install einops
pip install causal-conv1d==1.1.0
pip install mamba-ssm==1.1.4
pip install flash-attn==2.5.0

# Run training
./examples/mamba/train_mamba.sh
```

![mamba](./assets/loss_mamba.png)

> https://wandb.ai/bouteille/test/reports/Mamba-loss--Vmlldzo2OTgwNDM5

## Credits
Credits to the following repositories from which the code was adapted:
- https://github.com/state-spaces/mamba