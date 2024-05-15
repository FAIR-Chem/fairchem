# Towards Training Billion Parameter Graph Neural Networks for Atomic Simulations

Anuroop Sriram, Abhishek Das, Brandon M. Wood, Siddharth Goyal, C. Lawrence Zitnick

[[`arXiv:2203.09697`](https://arxiv.org/abs/2203.09697)]


To use graph parallel training, add `--gp-gpus N` to your command line, where N = number of GPUs to split the model over. This flag works for all tasks (`train`, `predict`, `validate` & `run-relaxations`).

As an example, the Gemnet-XL model can be trained using:
```bash
python main.py --mode train --config-yml configs/s2ef/all/gp_gemnet/gp-gemnet-xl.yml \
       --distributed --num-nodes 32 --num-gpus 8 --gp-gpus 4
```
This trains the model on 256 GPUs (32 nodes x 8 GPUs each) with 4-way graph parallelism (i.e. the graph is distributed over 4 GPUs) and 64-way data parallelism (64 == 256 / 4).

The Gemnet-XL model was trained without AMP as it led to unstable training.

## Citing

If you use Graph Parallelism in your work, please consider citing:

```bibtex
@inproceedings{sriram_graphparallel_2022,
  title={{Towards Training Billion Parameter Graph Neural Networks for Atomic Simulations}},
  author={Sriram, Anuroop and Das, Abhishek and Wood, Brandon M. and Goyal, Siddharth and Zitnick, C. Lawrence},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2022}
}
```
