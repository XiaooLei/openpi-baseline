# RSS Deployment

This branch keeps deployment checkpoints as params-only tar files. They include `params`, `assets`, and `_CHECKPOINT_METADATA`; they do not include `train_state`.

## Checkpoint Tar Files

Water MEM42:

```bash
/inspire/qb-ilm/project/gjjproject/public/xl/openpi-baseline/checkpoints/pi05_seal-water-bottle-cap_memory42_phase2_bc_lr5e6/seal_memory42_60_120_water120_phase2_bc_lr5e6/seal_memory42_60_120_water120_phase2_bc_lr5e6_35000_params_only.tar
```

Hanoi no-memory:

```bash
/inspire/qb-ilm/project/gjjproject/public/xl/openpi-baseline/checkpoints/pi05_tower-of-hanoi-game_nomem_phase2_bc_lr5e6/tower_of_hanoi_game_nomem_phase2_bc_lr5e6_20k/tower_of_hanoi_game_nomem_phase2_bc_lr5e6_20k_19999_params_only.tar
```

The same files are also staged under:

```bash
/inspire/hdd/global_user/gongjingjing-25039/xl/openpi-baseline/checkpoints
```

## Extract Checkpoints

Water MEM42:

```bash
cd /inspire/hdd/global_user/gongjingjing-25039/xl/openpi-baseline/checkpoints/pi05_seal-water-bottle-cap_memory42_phase2_bc_lr5e6/seal_memory42_60_120_water120_phase2_bc_lr5e6
tar -xf seal_memory42_60_120_water120_phase2_bc_lr5e6_35000_params_only.tar
```

This creates:

```bash
35000/params
35000/assets
35000/_CHECKPOINT_METADATA
```

Hanoi no-memory:

```bash
cd /inspire/hdd/global_user/gongjingjing-25039/xl/openpi-baseline/checkpoints/pi05_tower-of-hanoi-game_nomem_phase2_bc_lr5e6/tower_of_hanoi_game_nomem_phase2_bc_lr5e6_20k
tar -xf tower_of_hanoi_game_nomem_phase2_bc_lr5e6_20k_19999_params_only.tar
```

This creates:

```bash
19999/params
19999/assets
19999/_CHECKPOINT_METADATA
```

## Serve Water MEM42

```bash
cd /inspire/hdd/global_user/gongjingjing-25039/xl/openpi-baseline
PORT=8000 bash scripts/serve_seal_memory42_35k.sh
```

The default checkpoint is:

```bash
checkpoints/pi05_seal-water-bottle-cap_memory42_phase2_bc_lr5e6/seal_memory42_60_120_water120_phase2_bc_lr5e6/35000
```

Override it with:

```bash
CHECKPOINT_DIR=/path/to/35000 PORT=8000 bash scripts/serve_seal_memory42_35k.sh
```

The client only needs to send the current observation/state. It does not need to send `state_history`; the policy server fills `state_history` from the current-state stream using checkpoint metadata. Start a new websocket connection, or send an observation with `reset`, `episode_start`, or `is_first`, at the beginning of a new episode.

## Serve Hanoi No-Memory

```bash
cd /inspire/hdd/global_user/gongjingjing-25039/xl/openpi-baseline
PORT=8000 bash scripts/serve_tower_nomem_19999.sh
```

The default checkpoint is:

```bash
checkpoints/pi05_tower-of-hanoi-game_nomem_phase2_bc_lr5e6/tower_of_hanoi_game_nomem_phase2_bc_lr5e6_20k/19999
```

Override it with:

```bash
CHECKPOINT_DIR=/path/to/19999 PORT=8000 bash scripts/serve_tower_nomem_19999.sh
```

## Offline Requirements

The server expects the tokenizer to exist locally:

```bash
.cache/openpi/big_vision/paligemma_tokenizer.model
```

Both serve scripts set:

```bash
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

so cluster deployment does not require network access.
