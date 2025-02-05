# Diffusion Policy Test

## Train

```bash
conda activate robodiff
```

Obs Avoid

```bash
python train.py \
--config-dir=. --config-name=lowdim_obsavoid_diffusion_policy_transformer.yaml \
training.seed=42 training.device=cuda:0 \
hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
```
