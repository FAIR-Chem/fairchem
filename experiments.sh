python sbatch.py gres="gpu:4" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/schnet/new_schnet.yml --optim.lr_initial=0.0007 --model.graph_rewiring='one-supernode-per-graph' --note='smaller lr, pe + one-supernode-per-graph type'" env=ocp

python sbatch.py gres="gpu:4" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/forcenet/new_forcenet.yml --optim.lr_initial=0.0003 --model.graph_rewiring='one-supernode-per-graph' --note='smaller lr, pe + one-supernode-per-graph type'" env=ocp

python sbatch.py gres="gpu:4" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/dimenet_plus_plus/new_dpp.yml --optim.lr_initial=0.00008 --model.graph_rewiring='one-supernode-per-graph' --note='smaller lr, pe + one-supernode-per-graph type'" env=ocp

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
