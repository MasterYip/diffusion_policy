# Diffusion Policy Test

## Train & Eval

```bash
conda activate diffuseloco
```

**Obs Avoid Rolling Diffusion**

```bash
python train.py \
--config-dir=. \
--config-name=lowdim_obsavoid_rolldiff_policy_transformer.yaml \
training.seed=42 training.device=cuda:0 \
hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
```

Continue training

```bash
python train.py \
--config-dir=. \
--config-name=lowdim_obsavoid_rolldiff_policy_transformer.yaml \
training.seed=42 training.device=cuda:0 \
hydra.run.dir='data/outputs/2025.03.27/14.14.53_train_diffusion_transformer_lowdim_obsavoid_lowdim'
```

Eval

Run the evaluation script:

```bash
python eval.py --checkpoint data/rolldiff_obsavoid_acc_0327.ckpt --output_dir data/obsavoid_output --device cuda:0
```

**Obs Avoid**

```bash
python train.py \
--config-dir=. --config-name=lowdim_obsavoid_diffusion_policy_transformer.yaml \
training.seed=42 training.device=cuda:0 \
hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
```

Continue training

```bash
python train.py \
--config-dir=. \
--config-name=lowdim_obsavoid_diffusion_policy_transformer.yaml \
training.seed=42 training.device=cuda:0 \
hydra.run.dir='data/outputs/2025.03.20/10.50.50_train_diffusion_transformer_lowdim_obsavoid_lowdim'
```

> Note: Comment out `val_every` to avoid validation during training.

Run the evaluation script:

```bash
python eval.py --checkpoint data/obsavoid_acc.ckpt --output_dir data/obsavoid_output --device cuda:0
```

Generate Dataset:
Excute `diffusion_policy/scripts/generate_obsavoid.py`

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

### Legged Gym Training

**Generate Dataset**

First, generate a dataset from legged gym environments:

```bash
cd <REPO_ROOT_DIR>
python ./diffusion_policy/diffusion_policy/scripts/legged_gym_dataset_gen.py \
  --output "./diffusion_policy/data/legged_gym/elspider_dataset.zarr" \
  --checkpoints "legged_gym_cmp/legged_gym/logs/flat_elspider_air/exported/policies/policy_1.pt" \
  --task_name "elspider_air_flat" \
  --n_episodes 256 \
  --episode_steps 1000 \
  --num_envs 256 \
  --headless
```

#### **Train Legged Gym Diffusion Policy Transformer**

```bash
python train.py \
--config-dir=. \
--config-name=lowdim_legged_gym_diffusion_policy_transformer.yaml \
training.seed=42 training.device=cuda:0 \
hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
```

Continue training:

```bash
python train.py \
--config-dir=. \
--config-name=lowdim_legged_gym_diffusion_policy_transformer.yaml \
training.seed=42 training.device=cuda:0 \
hydra.run.dir='data/outputs/2025.04.17/17.48.17_train_diffusion_transformer_lowdim_legged_gym_lowdim'
```

**Evaluate Legged Gym Diffusion Policy**

```bash
python eval.py \
--checkpoint data/outputs/2025.04.17/20.45.25_train_diffusion_transformer_lowdim_legged_gym_lowdim/checkpoints/latest.ckpt \
--output_dir data/legged_gym_output \
--device cuda:0 \
--max_steps 5000 \
--num_envs 1 \
```

You can visualize the policy by running without the `--headless` flag:

```bash
python eval.py \
--checkpoint data/best_legged_gym_diffusion_model.ckpt \
--output_dir data/legged_gym_output \
--device cuda:0 \
--visualize
```

#### **Train Legged Gym Rolling Diffusion**

```bash
python train.py \
--config-dir=. \
--config-name=lowdim_legged_gym_rolldiff_policy_transformer.yaml \
training.seed=42 training.device=cuda:0 \
hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
```

Continue training:

```bash
python train.py \
--config-dir=. \
--config-name=lowdim_legged_gym_rolldiff_policy_transformer.yaml \
training.seed=42 training.device=cuda:0 \
hydra.run.dir='data/outputs/2025.04.01/20.19.36_train_diffusion_transformer_lowdim_legged_gym_lowdim'
```

**Evaluate Legged Gym Policy**

```bash
python eval.py \
--checkpoint data/outputs/2025.04.17/17.48.17_train_diffusion_transformer_lowdim_legged_gym_lowdim/checkpoints/latest.ckpt \
--output_dir data/legged_gym_output \
--device cuda:0 \
--max_steps 5000
```

You can visualize the policy by running without the `--headless` flag:

```bash
python eval.py \
--checkpoint data/best_legged_gym_model.ckpt \
--output_dir data/legged_gym_output \
--device cuda:0 \
--visualize
```
