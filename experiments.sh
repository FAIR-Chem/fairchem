python sbatch.py gres="gpu:1" partition=long time=5:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/sfarinet/sfarinet.yml --optim.batch_size=64 --optim.eval_batch_size=64 --optim.lr_initial=0.0065 --optim.lr_gamma=0.028 --optim.warmup_steps=500 --optim.warmup_factor=0.3 --optim.max_epochs=20 --model.hidden_channels=180 --model.num_interactions=2 --model.num_gaussians=50 --model.cutoff=4.0 --tag_hidden_channels=64 pg_hidden_channels=0 --phys_embeds=False --phys_hidden_channels=32 --model.graph_rewiring='remove-tag-0' --model.energy_head='weighted-av-initial-embebds' --note='valide sota 10k sfarinet'" env=ocp

python sbatch.py gres="gpu:1" partition=long time=5:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/sfarinet/sfarinet.yml --optim.batch_size=64 --optim.eval_batch_size=64 --optim.lr_initial=0.0065 --optim.lr_gamma=0.028 --optim.warmup_steps=500 --optim.warmup_factor=0.3 --optim.max_epochs=20 --model.hidden_channels=180 --model.num_interactions=2 --model.num_gaussians=50 --model.cutoff=4.0 --tag_hidden_channels=64 pg_hidden_channels=0 --phys_embeds=False --phys_hidden_channels=32 --model.graph_rewiring='remove-tag-0' --model.energy_head=False --note='valide sota 10k sfarinet, no E_head'" env=ocp

python sbatch.py gres="gpu:1" partition=long time=5:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/sfarinet/sfarinet.yml --optim.batch_size=64 --optim.eval_batch_size=64 --optim.lr_initial=0.0065 --optim.lr_gamma=0.028 --optim.warmup_steps=500 --optim.warmup_factor=0.3 --optim.max_epochs=20 --model.hidden_channels=256 --model.num_interactions=2 --model.num_gaussians=50 --model.cutoff=4.0 --tag_hidden_channels=64 pg_hidden_channels=0 --phys_embeds=False --phys_hidden_channels=32 --model.graph_rewiring='remove-tag-0' --model.energy_head='weighted-av-initial-embebds' --note='valide sota 10k sfarinet, larger hidden'" env=ocp

python sbatch.py gres="gpu:1" partition=long time=5:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/sfarinet/sfarinet.yml --optim.batch_size=64 --optim.eval_batch_size=64 --optim.lr_initial=0.0065 --optim.lr_gamma=0.028 --optim.warmup_steps=500 --optim.warmup_factor=0.3 --optim.max_epochs=20 --model.hidden_channels=180 --model.num_interactions=2 --model.num_gaussians=50 --model.cutoff=6.0 --tag_hidden_channels=64 pg_hidden_channels=0 --phys_embeds=False --phys_hidden_channels=32 --model.graph_rewiring='remove-tag-0' --model.energy_head='weighted-av-initial-embebds' --note='valide sota 10k sfarinet, validate_cutoff'" env=ocp

# Get SOTA 10k 

# python sbatch.py gres="gpu:1" partition=long time=5:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/sfarinet/sfarinet.yml --optim.batch_size=64 --optim.eval_batch_size=64 --optim.lr_initial=0.005 --optim.lr_gamma=0.006 --optim.warmup_steps=400 --optim.warmup_factor=0.3 --optim.max_epochs=20 --model.hidden_channels=256 --model.num_interactions=2 --model.num_gaussians=50 --model.cutoff=4.0 --tag_hidden_channels=64 pg_hidden_channels=0 --phys_embeds=True --phys_hidden_channels=32 --graph_rewiring='remove-tag-0' --energy_head='weighted-av-initial-embebds' --note='valide sota 10k sfarinet, smaller cutoff'" env=ocp

# python sbatch.py gres="gpu:1" partition=long time=5:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/sfarinet/sfarinet.yml --optim.batch_size=64 --optim.eval_batch_size=64 --optim.lr_initial=0.005 --optim.lr_gamma=0.006 --optim.warmup_steps=400 --optim.warmup_factor=0.3 --optim.max_epochs=20 --model.hidden_channels=320 --model.num_interactions=2 --model.num_gaussians=100 --model.cutoff=§.0 --tag_hidden_channels=64 pg_hidden_channels=0 --phys_embeds=True --phys_hidden_channels=32 --graph_rewiring='remove-tag-0' --energy_head='weighted-av-initial-embebds' --note='valide sota 10k sfarinet, larger hidden/gaussian'" env=ocp

# python sbatch.py gres="gpu:1" partition=long time=5:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/sfarinet/sfarinet.yml --optim.batch_size=64 --optim.eval_batch_size=64 --optim.lr_initial=0.005 --optim.lr_gamma=0.006 --optim.lr_milestones=1500 2000 3000 --optim.warmup_steps=400 --optim.warmup_factor=0.3 --optim.max_epochs=20 --model.hidden_channels=256 --model.num_interactions=2 --model.num_gaussians=50 --model.cutoff=6.0 --tag_hidden_channels=64 pg_hidden_channels=0 --phys_embeds=True --phys_hidden_channels=32 --graph_rewiring='remove-tag-0' --energy_head='weighted-av-initial-embebds' --note='valide sota 10k sfarinet, diff milestones'" env=ocp

# python sbatch.py gres="gpu:1" partition=long time=5:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/sfarinet/sfarinet.yml --optim.batch_size=64 --optim.eval_batch_size=64 --optim.lr_initial=0.005 --optim.lr_gamma=0.006 --optim.warmup_steps=400 --optim.warmup_factor=0.3 --optim.max_epochs=20 --model.hidden_channels=256 --model.num_interactions=2 --model.num_gaussians=50 --model.cutoff=6.0 --tag_hidden_channels=64 pg_hidden_channels=0 --phys_embeds=True --phys_hidden_channels=32 --graph_rewiring='remove-tag-0' --energy_head='weighted-av-initial-embebds' --fa='2D' --note='valide sota 10k sfarinet, fa'" env=ocp

# python sbatch.py gres="gpu:1" partition=long time=5:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/sfarinet/sfarinet.yml --optim.batch_size=64 --optim.eval_batch_size=64 --optim.lr_initial=0.005 --optim.lr_gamma=0.006 --optim.warmup_steps=400 --optim.warmup_factor=0.3 --optim.max_epochs=20 --model.hidden_channels=256 --model.num_interactions=2 --model.num_gaussians=50 --model.cutoff=6.0 --tag_hidden_channels=64 pg_hidden_channels=0 --phys_embeds=True --phys_hidden_channels=32 --graph_rewiring='remove-tag-0' --energy_head='weighted-av-initial-embebds' --fa='3D' --note='valide sota 10k sfarinet, 3D fa'" env=ocp

# # Sfarinet hyperparam tuning

# python sbatch.py gres="gpu:1" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/sfarinet/sfarinet.yml --optim.batch_size=32 --optim.eval_batch_size=32 --optim.lr_initial=0.005 --optim.warmup_steps=400 --optim.warmup_factor=0.2 --optim.max_epochs=20 --model.hidden_channels=256 --model.num_interactions=3 --model.num_gaussians=100 --model.cutoff=6.0 --model.num_filters=128 --fa='2D' --note='2D fa Smaller batch size'" env=ocp

# python sbatch.py gres="gpu:1" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/sfarinet/sfarinet.yml --optim.batch_size=128 --optim.eval_batch_size=128 --optim.lr_initial=0.005 --optim.warmup_steps=400 --optim.warmup_factor=0.2 --optim.max_epochs=20 --model.hidden_channels=256 --model.num_interactions=3 --model.num_gaussians=100 --model.cutoff=6.0 --model.num_filters=128 --fa='2D' --note='2D fa Larger batch'" env=ocp

# python sbatch.py gres="gpu:1" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/sfarinet/sfarinet.yml --optim.batch_size=64 --optim.eval_batch_size=64 --optim.lr_initial=0.001 --optim.warmup_steps=500 --optim.warmup_factor=0.2 --optim.max_epochs=20 --model.hidden_channels=256 --model.num_interactions=3 --model.num_gaussians=100 --model.cutoff=6.0 --model.num_filters=128 --fa='2D' --note='2D fa Smaller lr'" env=ocp

# python sbatch.py gres="gpu:1" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/sfarinet/sfarinet.yml --optim.batch_size=64 --optim.eval_batch_size=64 --optim.lr_initial=0.005 --optim.warmup_steps=400 --optim.warmup_factor=0.2 --optim.max_epochs=20 --model.hidden_channels=128 --model.num_interactions=4 --model.num_gaussians=100 --model.cutoff=6.0 --model.num_filters=128 --fa='2D' --note='2D fa  Smaller hidden, more interactions'" env=ocp

# python sbatch.py gres="gpu:1" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/sfarinet/sfarinet.yml --optim.batch_size=64 --optim.eval_batch_size=64 --optim.lr_initial=0.01 --optim.warmup_steps=500 --optim.warmup_factor=0.2 --optim.max_epochs=20 --model.hidden_channels=256 --model.num_interactions=3 --model.num_gaussians=100 --model.cutoff=6.0 --model.num_filters=128 --fa='2D' --note='2D fa Larger lr'" env=ocp

# # With previous extensions

# python sbatch.py gres="gpu:1" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/sfarinet/sfarinet.yml --optim.batch_size=64 --optim.eval_batch_size=64 --optim.lr_initial=0.005 --optim.warmup_steps=500 --optim.warmup_factor=0.2 --optim.max_epochs=20 --model.hidden_channels=256 --model.num_interactions=3 --model.num_gaussians=100 --model.cutoff=6.0 --model.num_filters=128 --graph_rewiring='remove-tag-0' --note='Tag0'" env=ocp

# python sbatch.py gres="gpu:1" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/sfarinet/sfarinet.yml --optim.batch_size=64 --optim.eval_batch_size=64 --optim.lr_initial=0.005 --optim.warmup_steps=500 --optim.warmup_factor=0.2 --optim.max_epochs=20 --model.hidden_channels=256 --model.num_interactions=3 --model.num_gaussians=100 --model.cutoff=6.0 --model.num_filters=128 --tag_hidden_channels=32 pg_hidden_channels=32 --phys_embeds=True --phys_hidden_channels=32 --note='full embeds'" env=ocp

# python sbatch.py gres="gpu:1" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/sfarinet/sfarinet.yml --optim.batch_size=64 --optim.eval_batch_size=64 --optim.lr_initial=0.005 --optim.warmup_steps=500 --optim.warmup_factor=0.2 --optim.max_epochs=20 --model.hidden_channels=256 --model.num_interactions=3 --model.num_gaussians=100 --model.cutoff=6.0 --model.num_filters=128 --energy_head='pooling' --note='Pooling'" env=ocp

# python sbatch.py gres="gpu:1" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/sfarinet/sfarinet.yml --optim.batch_size=64 --optim.eval_batch_size=64 --optim.lr_initial=0.005 --optim.warmup_steps=500 --optim.warmup_factor=0.2 --optim.max_epochs=20 --model.hidden_channels=256 --model.num_interactions=3 --model.num_gaussians=100 --model.cutoff=6.0 --model.num_filters=128 --energy_head='weighted-av-initial-embebds' --note='Weighted av.'" env=ocp

# python sbatch.py gres="gpu:1" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/sfarinet/sfarinet.yml --optim.batch_size=64 --optim.eval_batch_size=64 --optim.lr_initial=0.005 --optim.warmup_steps=500 --optim.warmup_factor=0.2 --optim.max_epochs=20 --model.hidden_channels=256 --model.num_interactions=3 --model.num_gaussians=100 --model.cutoff=6.0 --model.num_filters=128 --tag_hidden_channels=32 --graph_rewiring='remove-tag-0' --energy_head='weighted-av-initial-embebds' --note='All'" env=ocp

# Other models

# python sbatch.py gres="gpu:1" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/forcenet/new_forcenet.yml --note='Baseline'" env=ocp

# python sbatch.py gres="gpu:4" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/forcenet/new_forcenet.yml --fa='3D' --note='3D rotation'" env=ocp

# python sbatch.py gres="gpu:4" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/forcenet/new_forcenet.yml --fa='2D' --note='2D rotation'" env=ocp

# python sbatch.py gres="gpu:4" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/schnet/new_schnet.yml --fa='3D' --note='3D rotation'" env=ocp

# python sbatch.py gres="gpu:4" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/schnet/new_schnet.yml --fa='2D' --note='2D rotation'" env=ocp

# python sbatch.py gres="gpu:4" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/dimenet_plus_plus/new_dpp.yml --fa='2D' --note='2D rotation'" env=ocp

# python sbatch.py gres="gpu:4" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/sfarinet/sfarinet.yml --fa='2D' --note='2D rotation'" env=ocp

# python sbatch.py gres="gpu:4" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/sfarinet/sfarinet.yml --fa='3D' --note='3D rotation'" env=ocp







