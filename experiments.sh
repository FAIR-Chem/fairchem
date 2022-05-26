


# python sbatch.py gres="gpu:1" partition=long time=1:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/schnet/new_schnet.yml --model.tag_hidden_channels=0 --model.hidden_channels=288 --note='Fixed embed only, large hidden'" env=ocp

# python sbatch.py gres="gpu:1" partition=long time=3:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/schnet/new_schnet.yml --model.tag_hidden_channels=0 --note='Fixed embed only, same hidden'" env=ocp

# python sbatch.py gres="gpu:1" partition=long time=3:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/schnet/new_schnet.yml --model.tag_hidden_channels=32 --model.hidden_channels=320 --note='Fixed embed + tag, large hidden'" env=ocp

# python sbatch.py gres="gpu:1" partition=long time=3:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/schnet/new_schnet.yml --model.tag_hidden_channels=32 --model.hidden_channels=256 --note='Fixed embed + tag, same hidden'" env=ocp
