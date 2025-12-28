# Chinese-to-English NMT

**Goal:** To implement and compare Chinese–English machine translation using RNN and Transformer architectures.
**Scope:** Development of NMT models from scratch (RNN, Transformer), fine-tuning pretrained model (mT5), and conducting comparative architectural analysis.

**Inference Usage:**
RNN:
```bash
python inference.py \
    --checkpoint checkpoints/attn_exp_concat_decay/best_model.pt \
    --model_type rnn
```
Transformer (from scratch):
```bash
python inference.py \
    --checkpoint checkpoints/transformer_layernorm_absolute/best_model.pt \
    --model_type transformer
```
mT5 (fine-tuned from pretrained):
```bash
python inference.py \
    --checkpoint checkpoints/pretrained_mt5-base/best_model.pt \
    --model_type pretrained \
    --pretrained_model_name "path/to/mt5-base"
```