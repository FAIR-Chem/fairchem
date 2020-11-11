# Pretrained models for OCP

This page summarizes all the pretrained models released as part of the [Open Catalyst Project](https://opencatalystproject.org/). All models were trained using this codebase in October 2020.

* All configurations for these baseline models are available in the [`configs/`](https://github.com/Open-Catalyst-Project/ocp/tree/master/configs) directory.
* All of these models are trained on various splits of the OC20 S2EF / IS2RE datasets. For details, see [https://arxiv.org/abs/2010.09990v1](https://arxiv.org/abs/2010.09990) and https://github.com/Open-Catalyst-Project/ocp/blob/master/DATASET.md.
* All model checkpoints were created using Pytorch 1.6. Please follow steps listed in the [readme](https://github.com/open-catalyst-Project/ocp#installation) to set up your conda environment with correct package versions.

## S2EF baselines: optimized for EFwT



|model	|split	|downloadable link	|val ID force MAE	|val ID EFwT	|
|---	|---	|---	|---	|---	|
|CGCNN	|200k	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/cgcnn_200k.pt	|0.08	|0%	|
|CGCNN	|2M	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/cgcnn_2M.pt	|0.0673	|0.01%	|
|CGCNN	|20M	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/cgcnn_20M.pt	|0.065	|0%	|
|CGCNN	|All	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/cgcnn_all.pt	|0.0684	|0.01%	|
|DimeNet	|200k	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/dimenet_200k.pt	|0.0693	|0.01%	|
|DimeNet	|2M	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/dimenet_2M.pt	|0.0576	|0.02%	|
|DimeNet	|20M	|Coming soon	|	|	|
|DimeNet	|All	|Coming soon	|	|	|
|SchNet	|200k	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/schnet_200k.pt	|0.0743	|0%	|
|SchNet	|2M	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/schnet_2M.pt	|0.0737	|0%	|
|SchNet	|20M	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/schnet_20M.pt	|0.0568	|0.03%	|
|SchNet	|All	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/schnet_all_large.pt	|0.0494	|0.12%	|

## S2EF baselines: optimized for force only

## 

|model	|split	|downloadable link	|val ID force MAE	|
|---	|---	|---	|---	|
|SchNet	|All	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/schnet_all_forceonly.pt	|0.0443	|
|DimeNet++	|All	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/dimenetpp_all_forceonly.pt	|0.0334	|

## IS2RE baselines



|model	|split	|downloadable link	|val ID energy MAE	|
|---	|---	|---	|---	|
|CGCNN	|10k	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/is2re/cgcnn_10k.pt	|1.0479	|
|CGCNN	|100k	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/is2re/cgcnn_100k.pt	|0.7066	|
|CGCNN	|All	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/is2re/cgcnn_all.pt	|0.6048	|
|DimeNet	|10k	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/is2re/dimenet_10k.pt	|1.0117	|
|DimeNet	|100k	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/is2re/dimenet_100k.pt	|0.6658	|
|DimeNet	|All	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/is2re/dimenet_all.pt	|0.5999	|
|SchNet	|10k	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/is2re/schnet_10k.pt	|1.0858	|
|SchNet	|100k	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/is2re/schnet_100k.pt	|0.7266	|
|SchNet	|All	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/is2re/schnet_all.pt	|0.6691	|

The Open Catalyst 2020 (OC20) dataset is licensed under a [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/legalcode).

Please cite the following paper in any research manuscript using the OC20 dataset or pretrained models:


```
@misc{ocp_dataset,
    title={The Open Catalyst 2020 (OC20) Dataset and Community Challenges},
    author={Lowik Chanussot* and Abhishek Das* and Siddharth Goyal* and Thibaut Lavril* and Muhammed Shuaibi* and Morgane Riviere and Kevin Tran and Javier Heras-Domingo and Caleb Ho and Weihua Hu and Aini Palizhati and Anuroop Sriram and Brandon Wood and Junwoong Yoon and Devi Parikh and C. Lawrence Zitnick and Zachary Ulissi},
    year={2020},
    eprint={2010.09990},
    archivePrefix={arXiv}
}
```

