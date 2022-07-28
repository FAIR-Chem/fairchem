python sbatch.py gres="gpu:1" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/schnet/new_schnet.yml --energy_head='diff_pooling' --note='diffpooling'" env=ocp

python sbatch.py gres="gpu:1" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/schnet/new_schnet.yml --energy_head='pooling' --note='pooling'" env=ocp

python sbatch.py gres="gpu:4" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/schnet/new_schnet.yml --energy_head='weighted-av-initial-embeds' --note='w. av. initi embeds'" env=ocp

python sbatch.py gres="gpu:4" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/schnet/new_schnet.yml --energy_head='weighted-av-final-embeds' --note='w. av. initi embeds'" env=ocp


# python sbatch.py gres="gpu:1" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/100k/dimenet_plus_plus/new_dpp.yml --model.graph_rewiring='one-supernode-per-graph' --note='pe + one-supernode-per-graph type'" env=ocp

# python sbatch.py gres="gpu:1" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --energy_head='weighted-av-initial-embeds' --config-yml configs/is2re/10k/schnet/new_schnet.yml --note='weighted av on inital embeds'" env=ocp

# python sbatch.py gres="gpu:1" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/schnet/new_schnet.yml --energy_head='weighted-av-final-embeds' --note='weighted av on final embeds'" env=ocp

# python sbatch.py gres="gpu:1" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/schnet/new_schnet.yml --model.graph_rewiring='remove-tag-0' --note='remove-tag-0'" env=ocp

# python sbatch.py gres="gpu:4" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/schnet/new_schnet.yml --note='Baseline'" env=ocp

# python sbatch.py gres="gpu:4" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --optim.lr_initial=0.0007 --config-yml configs/is2re/all/schnet/new_schnet.yml --note='Baseline'" env=ocp

# python sbatch.py gres="gpu:4" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/schnet/schnet.yml --note='True Baseline'" env=ocp

# python sbatch.py gres="gpu:4" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/forcenet/forcenet.yml --note='True Baseline'" env=ocp


# python sbatch.py gres="gpu:4" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/schnet/new_schnet.yml --model.graph_rewiring='remove-tag-0' --note='remove-tag-0'" env=ocp

# python sbatch.py gres="gpu:4" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/schnet/new_schnet.yml --model.graph_rewiring='one-supernode-per-graph' --note='one-supernode-per-graph'" env=ocp

# python sbatch.py gres="gpu:4" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/schnet/new_schnet.yml --model.graph_rewiring='one-supernode-per-atom-type' --note='one-supernode-per-atom-type'" env=ocp

# python sbatch.py gres="gpu:4" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/schnet/new_schnet.yml --model.graph_rewiring='one-supernode-per-atom-type-dist' --note='one-supernode-per-atom-type-dist'" env=ocp

# python sbatch.py gres="gpu:4" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/forcenet/new_forcenet.yml --model.graph_rewiring='remove-tag-0' --note='remove-tag-0'" env=ocp

# python sbatch.py gres="gpu:4" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/forcenet/new_forcenet.yml --model.graph_rewiring='one-supernode-per-graph' --note='new one-supernode-per-graph'" env=ocp

# python sbatch.py gres="gpu:4" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/forcenet/new_forcenet.yml --model.graph_rewiring='one-supernode-per-atom-type' --note='new  one supernode per atom type'" env=ocp

# python sbatch.py gres="gpu:4" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/forcenet/new_forcenet.yml --model.graph_rewiring='one-supernode-per-atom-type-dist' --note='new  one supernode per atom type dist'" env=ocp

# python sbatch.py gres="gpu:4" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/dimenet_plus_plus/new_dpp.yml --model.graph_rewiring='one-supernode-per-atom-type' --note='new  one supernode per atom type'" env=ocp

# python sbatch.py gres="gpu:4" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/dimenet_plus_plus/new_dpp.yml --model.graph_rewiring='pe + one-supernode-per-graph' --note='new one supernode per graph'" env=ocp
