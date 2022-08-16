# python sbatch.py gres="gpu:4" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/forcenet/new_forcenet.yml --energy_head='graclus' --note='graclus'" env=ocp

# python sbatch.py gres="gpu:4" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/forcenet/new_forcenet.yml --energy_head='random' --note='random'" env=ocp

# python sbatch.py gres="gpu:4" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/forcenet/new_forcenet.yml --energy_head='weighted-av-final-embeds' --note='weighted-av-final-embeds'" env=ocp

# python sbatch.py gres="gpu:4" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/forcenet/new_forcenet.yml --energy_head='weighted-av-initial-embeds' --note='weighted-av-initial-embeds'" env=ocp

# python sbatch.py gres="gpu:4" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/forcenet/new_forcenet.yml --energy_head='pooling' --note='pooling'" env=ocp

# python sbatch.py gres="gpu:4" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/forcenet/new_forcenet.yml --note='BASELINE'" env=ocp
