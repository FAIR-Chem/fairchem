


#python main.py --config-yml configs/ocp_s2ef/dimenet_dist2.yml --seed 1 --num-gpus 8 --distributed --submit
#python main.py --config-yml configs/ocp_s2ef/dimenet_dist2.yml --seed 1 --num-gpus 8 --distributed --submit --amp
python main.py --config-yml configs/ocp_s2ef/dimenet_dist2.yml --identifier dimenet.amp --seed 1 --num-gpus 8 --distributed --submit --amp

#python main.py --config-yml configs/ocp_s2ef/schnet_dist2.yml --seed 1 --num-gpus 8 --distributed --submit
#python main.py --config-yml configs/ocp_s2ef/schnet_dist2.yml --seed 1 --num-gpus 8 --distributed --submit --amp
python main.py --config-yml configs/ocp_s2ef/schnet_dist2.yml --identifier schnet.amp --seed 1 --num-gpus 8 --distributed --submit --amp

