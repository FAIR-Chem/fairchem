python sbatch.py gres="gpu:1" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/dimenet_plus_plus/new_dpp.yml --model.graph_rewiring='remove-tag-0' --note='tag0'" env=ocp

python sbatch.py gres="gpu:1" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/forcenet/new_forcenet.yml  --model.graph_rewiring='remove-tag-0' --note='tag0'" env=ocp

python sbatch.py gres="gpu:1" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/schnet/new_schnet.yml --model.graph_rewiring='remove-tag-0' --note='tag0'" env=ocp

python sbatch.py gres="gpu:1" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/dimenet_plus_plus/new_dpp.yml --model.graph_rewiring='one-supernode-per-graph' --note='one supernode per G'" env=ocp

python sbatch.py gres="gpu:1" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/forcenet/new_forcenet.yml  --model.graph_rewiring='one-supernode-per-graph' --note='one supernode per G'" env=ocp

python sbatch.py gres="gpu:1" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/schnet/new_schnet.yml --model.graph_rewiring='one-supernode-per-graph' --note='one supernode per G'" env=ocp

# python sbatch.py gres="gpu:1" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/dimenet_plus_plus/new_dpp.yml --model.graph_rewiring='one-supernode-per-atom-type' --note='Embeds + tag0'" env=ocp

# python sbatch.py gres="gpu:1" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/forcenet/new_forcenet.yml  --model.graph_rewiring='one-supernode-per-atom-type' --note='Embeds + tag0'" env=ocp

# python sbatch.py gres="gpu:1" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/schnet/new_schnet.yml --model.graph_rewiring='one-supernode-per-atom-type' --note='Embeds + tag0'" env=ocp


# python sbatch.py gres="gpu:4" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/dimenet_plus_plus/new_dpp.yml --optim.max_epochs=20 --model.phys_embeds=True --model.tag_hidden_channels=32  --model.pg_hidden_channels=32 --model.graph_rewiring='remove-tag-0' --note='Embeds + tag0'" env=ocp

# python sbatch.py gres="gpu:4" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/forcenet/new_forcenet.yml --model.phys_embeds=True --model.tag_hidden_channels=32  --model.pg_hidden_channels=32 --model.graph_rewiring='remove-tag-0' --note='Embeds + tag0'" env=ocp

# python sbatch.py gres="gpu:4" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/schnet/new_schnet.yml --model.phys_embeds=True --model.tag_hidden_channels=32  --model.pg_hidden_channels=32 --model.graph_rewiring='remove-tag-0' --note='Embeds + tag0'" env=ocp

# python sbatch.py gres="gpu:1" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/schnet/new_schnet.yml --model.phys_embeds=True --model.tag_hidden_channels=32  --model.pg_hidden_channels=32 --model.graph_rewiring='remove-tag-0' --note='Embeds + tag0'" env=ocp

#python sbatch.py gres="gpu:1" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/dimenet_plus_plus/new_dpp.yml --optim.max_epochs=20 --model.phys_embeds=True --model.tag_hidden_channels=32  --model.pg_hidden_channels=32 --model.graph_rewiring='remove-tag-0' --note='Embeds + tag0'" env=ocp

#python sbatch.py gres="gpu:1" partition=long time=24:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/forcenet/new_forcenet.yml --model.phys_embeds=True --model.tag_hidden_channels=32 --model.pg_hidden_channels=32 --model.graph_rewiring='remove-tag-0' --note='Embeds + tag0'" env=ocp

#python sbatch.py gres="gpu:1" partition=long time=6:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/all/schnet/new_schnet.yml --model.phys_hidden_channels=32 --optim.max_epochs=30 --optim.lr_initial=0.0007 --model.phys_embeds=True --model.tag_hidden_channels=32 --model.pg_hidden_channels=32 --model.graph_rewiring='remove-tag-0' --note='Embeds + tag0'" env=ocp

#python sbatch.py gres="gpu:1" partition=long time=7:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/dimenet_plus_plus/new_dpp.yml --model.phys_embeds=True --model.tag_hidden_channels=0 --note='Baseline FIXED Embedding'" env=ocp

#python sbatch.py gres="gpu:1" partition=long time=7:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/dimenet_plus_plus/new_dpp.yml --model.phys_embeds=True --note='Baseline TAG + FIXED Embedding'" env=ocp

#python sbatch.py gres="gpu:1" partition=long time=6:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/schnet/new_schnet.yml --model.phys_embeds=True --model.tag_hidden_channels=0 --note='Baseline FIXED embedding'" env=ocp

#python sbatch.py gres="gpu:1" partition=long time=6:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/schnet/new_schnet.yml --model.phys_embeds=True --note='Baseline TAG + FIXED embedding'" env=ocp

# python sbatch.py gres="gpu:1" partition=long time=1:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/schnet/new_schnet.yml --model.tag_hidden_channels=0 --model.hidden_channels=288 --note='Phys embed only, large hidden'" env=ocp

# python sbatch.py gres="gpu:1" partition=long time=3:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/schnet/new_schnet.yml --model.tag_hidden_channels=0 --note='Phys embed only, same hidden'" env=ocp

# python sbatch.py gres="gpu:1" partition=long time=3:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/schnet/new_schnet.yml --model.tag_hidden_channels=32 --model.hidden_channels=320 --note='Phys embed + tag, large hidden'" env=ocp

# python sbatch.py gres="gpu:1" partition=long time=3:00:00 cpus=4 mem=32GB py_args="--mode train --config-yml configs/is2re/10k/schnet/new_schnet.yml --model.tag_hidden_channels=32 --model.hidden_channels=256 --note='Phys embed + tag, same hidden'" env=ocp
