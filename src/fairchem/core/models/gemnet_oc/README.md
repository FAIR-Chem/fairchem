# GemNet-OC: Developing Graph Neural Networks for Large and Diverse Molecular Simulation Datasets

Johannes Gasteiger, Muhammed Shuaibi, Anuroop Sriram, Stephan Günnemann, Zachary Ulissi, C. Lawrence Zitnick, Abhishek Das

[[`arXiv:2204.02782`](https://arxiv.org/abs/2204.02782)]

When running inference with a pretrained GemNet-OC model, make sure that the
`scale_file` path is correct in the config, otherwise predictions will be inaccurate.

| Model | Val ID 30k Force MAE | Val ID 30k Energy MAE | Val ID 30k Force cos | Test metrics | Download |
| ----- | -------------------- | --------------------- | -------------------- | ------------ | -------- |
| gemnet_oc_2M | 0.0225 | 0.2299 | 0.6174 | [S2EF](https://evalai.s3.amazonaws.com/media/submission_files/submission_179229/062c037e-4f1f-49c2-9eeb-8e14681a70ee.json) \| [IS2RE](https://evalai.s3.amazonaws.com/media/submission_files/submission_179296/6688f44f-9d5a-4020-beca-8b804e0212fb.json) \| [IS2RS](https://evalai.s3.amazonaws.com/media/submission_files/submission_179257/0d02a349-0abe-44c0-a65c-29a9df75c886.json) | [config](https://github.com/Open-Catalyst-Project/ocp/blob/main/configs/s2ef/2M/gemnet/gemnet-oc.yml) \| [checkpoint](https://dl.fbaipublicfiles.com/opencatalystproject/models/2022_07/s2ef/gemnet_oc_base_s2ef_2M.pt) \| [scale file](https://github.com/Open-Catalyst-Project/ocp/blob/481f3a5a92dc787384ddae9fe3f50f5d932712fd/configs/s2ef/all/gemnet/scaling_factors/gemnet-oc.pt) |
| gemnet_oc_all | 0.0179 | 0.1668 | 0.6879 | [S2EF](https://evalai.s3.amazonaws.com/media/submission_files/submission_179008/6e731f20-17cf-417e-b0ad-97352be8cc37.json) \| [IS2RE]() \| [IS2RS](https://evalai.s3.amazonaws.com/media/submission_files/submission_160550/72a65a42-1fa9-44c5-8546-9eb691df8d2e.json) | [config](https://github.com/Open-Catalyst-Project/ocp/blob/main/configs/s2ef/all/gemnet/gemnet-oc.yml) \| [checkpoint](https://dl.fbaipublicfiles.com/opencatalystproject/models/2022_07/s2ef/gemnet_oc_base_s2ef_all.pt) \| [scale file](https://github.com/Open-Catalyst-Project/ocp/blob/481f3a5a92dc787384ddae9fe3f50f5d932712fd/configs/s2ef/all/gemnet/scaling_factors/gemnet-oc.pt) |
| gemnet_oc_large_all_md_energy | 0.0178 | 0.1504 | 0.6906 | [S2EF](https://evalai.s3.amazonaws.com/media/submission_files/submission_179143/40940149-6a4a-49a4-a2ce-38486215990f.json) | - |
| gemnet_oc_large_all_md_force | 0.0164 | 0.1665 | 0.7139 | [S2EF](https://evalai.s3.amazonaws.com/media/submission_files/submission_179042/ba160459-0de3-4583-a98b-12102138c61e.json) \| [IS2RS](https://evalai.s3.amazonaws.com/media/submission_files/submission_169243/10bc7c8d-5124-4338-aaf3-04a7d015c4a0.json) | [config](https://github.com/Open-Catalyst-Project/ocp/blob/main/configs/s2ef/all/gemnet/gemnet-oc-large.yml) \| [checkpoint](https://dl.fbaipublicfiles.com/opencatalystproject/models/2022_07/s2ef/gemnet_oc_large_s2ef_all_md.pt) \| [scale file](https://github.com/Open-Catalyst-Project/ocp/blob/481f3a5a92dc787384ddae9fe3f50f5d932712fd/configs/s2ef/all/gemnet/scaling_factors/gemnet-oc-large.pt) |
| gemnet_oc_large_all_md_energy + gemnet_oc_large_all_md_force | - | - | - | [IS2RE](https://evalai.s3.amazonaws.com/media/submission_files/submission_212962/6acc7cf7-e18b-4d6a-9082-b4a114110dbf.json) | - |

## Citing

If you use GemNet-OC in your work, please consider citing:

```bibtex
@article{gasteiger_gemnet_oc_2022,
  title = {{GemNet-OC: Developing Graph Neural Networks for Large and Diverse Molecular Simulation Datasets}},
  author = {Gasteiger, Johannes and Shuaibi, Muhammed and Sriram, Anuroop and G{\"u}nnemann, Stephan and Ulissi, Zachary and Zitnick, C Lawrence and Das, Abhishek},
  journal = {Transactions on Machine Learning Research (TMLR)},
  year = {2022},
}
```
