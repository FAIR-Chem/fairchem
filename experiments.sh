python sbatch.py gres="gpu:1" partition=long time=6:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/schnet/new_schnet.yml --model.fixed_embeds=True --note='Baseline TAG + FIXED + PG embedding'" env=ocp

python sbatch.py gres="gpu:1" partition=long time=6:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/schnet/new_schnet.yml --model.fixed_embeds=False --model.tag_hidden_channels=0 --note='Baseline PG embedding'" env=ocp

python sbatch.py gres="gpu:1" partition=long time=6:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/dimenet_plus_plus/new_dpp.yml --model.fixed_embeds=True --note='Baseline TAG + FIXED + PG embedding'" env=ocp

python sbatch.py gres="gpu:1" partition=long time=6:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/dimenet_plus_plus/new_dpp.yml --model.fixed_embeds=False --model.tag_hidden_channels=0 --model.fixed_embeds=True --note='Baseline PG embedding'" env=ocp

#python sbatch.py gres="gpu:1" partition=long time=7:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/dimenet_plus_plus/new_dpp.yml --model.fixed_embeds=True --model.tag_hidden_channels=0 --note='Baseline FIXED Embedding'" env=ocp

#python sbatch.py gres="gpu:1" partition=long time=7:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/dimenet_plus_plus/new_dpp.yml --model.fixed_embeds=True --note='Baseline TAG + FIXED Embedding'" env=ocp

#python sbatch.py gres="gpu:1" partition=long time=6:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/schnet/new_schnet.yml --model.fixed_embeds=True --model.tag_hidden_channels=0 --note='Baseline FIXED embedding'" env=ocp

#python sbatch.py gres="gpu:1" partition=long time=6:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/schnet/new_schnet.yml --model.fixed_embeds=True --note='Baseline TAG + FIXED embedding'" env=ocp

# python sbatch.py gres="gpu:1" partition=long time=1:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/schnet/new_schnet.yml --model.tag_hidden_channels=0 --model.hidden_channels=288 --note='Fixed embed only, large hidden'" env=ocp

# python sbatch.py gres="gpu:1" partition=long time=3:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/schnet/new_schnet.yml --model.tag_hidden_channels=0 --note='Fixed embed only, same hidden'" env=ocp

# python sbatch.py gres="gpu:1" partition=long time=3:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/schnet/new_schnet.yml --model.tag_hidden_channels=32 --model.hidden_channels=320 --note='Fixed embed + tag, large hidden'" env=ocp

# python sbatch.py gres="gpu:1" partition=long time=3:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/schnet/new_schnet.yml --model.tag_hidden_channels=32 --model.hidden_channels=256 --note='Fixed embed + tag, same hidden'" env=ocp
