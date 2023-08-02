# EquiformerV2

## OC20 checkpoints

Don't share any code or model weights publicly yet.

### S2EF-All + MD

#### 153M parameters, λₑ=2

- Config: `abhshkdz/equiformer_v2/configs/s2ef/all_md/153M_ec2.yml`
- Checkpoint: `/checkpoint/abhshkdz/open-catalyst-project/checkpoints/2023_06_07_equiformer_v2_shared_checkpoints/eq2_153M_ec2_041404_allmd.pt`
- Val ID 30k: 12.2 meV/A force MAE, 164 meV energy MAE
- Val overall (avg'ed over 4 splits): 14.6 meV/A force MAE, 230 meV energy MAE, 6.29 samples per GPU-sec on V100s

#### 153M parameters, λₑ=4

- Config: `abhshkdz/equiformer_v2/configs/s2ef/all_md/153M_ec4.yml`
- Checkpoint: `/checkpoint/abhshkdz/open-catalyst-project/checkpoints/2023_06_07_equiformer_v2_shared_checkpoints/eq2_153M_ec4_041802_allmd.pt`
- Val ID 30k: 12.6 meV/A force MAE, 159 meV energy MAE
- Val overall (avg'ed over 4 splits): 15.0 meV/A force MAE, 227 meV energy MAE, 6.29 samples per GPU-sec on V100s

#### 31M parameters, λₑ=4

- Config: `abhshkdz/equiformer_v2/configs/s2ef/all_md/31M.yml`
- Checkpoint: `/checkpoint/abhshkdz/open-catalyst-project/checkpoints/2023_06_07_equiformer_v2_shared_checkpoints/eq2_31M_ec4_051701_allmd.pt`
- Val ID 30k: 14.2 meV/A force MAE, 164 meV energy MAE
- Val overall (avg'ed over 4 splits): 16.3 meV/A force MAE, 232 meV energy MAE, 30 samples per GPU-sec on V100s

### S2EF-2M

#### 83M parameters, λₑ=2

- Config: `abhshkdz/equiformer_v2/configs/s2ef/2M/paper/2_9_041701.yml`
- Checkpoint: `/checkpoint/abhshkdz/open-catalyst-project/checkpoints/2023_06_07_equiformer_v2_shared_checkpoints/eq2_83M_2_9_041701_2M.pt`
- Val ID 30k: 16.7 meV/A force MAE, 213 meV energy MAE
- Val overall (avg'ed over 4 splits): 19.4 meV/A force MAE, 278 meV energy MAE, 13.3 samples per GPU-sec on V100s

## Questions

Ping Yi-Lun or Abhishek in case of any questions / issues with running EquiformerV2.
