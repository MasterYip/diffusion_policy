# Diffusion Policy Test

## Train & Eval

```bash
conda activate robodiff
```

**Obs Avoid**

```bash
python train.py \
--config-dir=. --config-name=lowdim_obsavoid_diffusion_policy_transformer.yaml \
training.seed=42 training.device=cuda:0 \
hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
```

Run the evaluation script:

```bash
python eval.py --checkpoint data/obsavoid.ckpt --output_dir data/obsavoid_output --device cuda:0
```

**PushT**

Launch training with seed 42 on GPU 0.

```bash
python train.py \
--config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml \
training.seed=42 training.device=cuda:0 \
hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
###
python train.py \
--config-dir=. --config-name=low_dim_block_pushing_diffusion_policy_cnn.yaml \
training.seed=42 training.device=cuda:0 \
hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
```

Resume training with seed 42 on GPU 0.

```bash
cd ~/Documents/CodeSpace/Python/diffusion_policy
conda activate robodiff
###
python train.py \
--config-dir=. \
--config-name=image_pusht_diffusion_policy_cnn.yaml \
training.seed=42 training.device=cuda:0 \
hydra.run.dir='data/outputs/2024.09.13/10.17.20_train_diffusion_unet_hybrid_pusht_image'
```

Run the evaluation script:

```bash
python eval.py --checkpoint data/0550-test_mean_score=0.969.ckpt --output_dir data/pusht_eval_output --device cuda:0
```
