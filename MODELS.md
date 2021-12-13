# Pretrained models for OCP

This page summarizes all the pretrained models released as part of the [Open Catalyst Project](https://opencatalystproject.org/). All models were trained using this codebase.

* All configurations for these models are available in the [`configs/`](https://github.com/Open-Catalyst-Project/ocp/tree/master/configs) directory.
* All of these models are trained on various splits of the OC20 S2EF / IS2RE datasets. For details, see [https://arxiv.org/abs/2010.09990](https://arxiv.org/abs/2010.09990) and https://github.com/Open-Catalyst-Project/ocp/blob/master/DATASET.md.

## S2EF models: optimized for EFwT

|model	|split	|downloadable link	|val ID force MAE	|val ID EFwT	|
|---	|---	|---	|---	|---	|
|CGCNN	|200k	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/cgcnn_200k.pt	|0.08	|0%	|
|CGCNN	|2M	    |https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/cgcnn_2M.pt	|0.0673	|0.01%	|
|CGCNN	|20M	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/cgcnn_20M.pt	|0.065	|0%	|
|CGCNN	|All	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/cgcnn_all.pt	|0.0684	|0.01%	|
|DimeNet	|200k	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/dimenet_200k.pt	|0.0693	|0.01%	|
|DimeNet	|2M	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/dimenet_2M.pt	|0.0576	|0.02%	|
|SchNet	|200k	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/schnet_200k.pt	|0.0743	|0%	|
|SchNet	|2M	    |https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/schnet_2M.pt	|0.0737	|0%	|
|SchNet	|20M	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/schnet_20M.pt	|0.0568	|0.03%	|
|SchNet	|All	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/schnet_all_large.pt	|0.0494	|0.12%	|
|DimeNet++	|200k   |https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_02/s2ef/dimenetpp_200k.pt	|0.0741	|0%	|
|DimeNet++	|2M     |https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_02/s2ef/dimenetpp_2M.pt	|0.0595	|0.01%	|
|DimeNet++	|20M    |https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_02/s2ef/dimenetpp_20M.pt	|0.0511	|0.06%	|
|DimeNet++	|All    |https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_02/s2ef/dimenetpp_all.pt	|0.0444	|0.12%	|
|SpinConv	|2M    |https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_12/s2ef/spinconv_force_centric_2M.pt	|0.0329	|0.18%	|
|SpinConv	|All    |https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_08/s2ef/spinconv_force_centric_all.pt	|0.0267	|1.02%	|
|GemNet-dT	|2M    |https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_12/s2ef/gemnet_t_direct_h512_2M.pt	|0.0257	|1.10%	|
|GemNet-dT	|All    |https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_08/s2ef/gemnet_t_direct_h512_all.pt	|0.0211	|2.21%	|

## S2EF models: optimized for force only

|model	|split	|downloadable link	|val ID force MAE	|
|---	|---	|---	|---	|
|SchNet	|All	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/schnet_all_forceonly.pt	|0.0443	|
|DimeNet++	|All	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/dimenetpp_all_forceonly.pt	|0.0334	|
|DimeNet++-Large	|All	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_02/s2ef/dimenetpp_large_all_forceonly.pt	|0.02825	|
|DimeNet++	|20M+Rattled    |https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_02/s2ef/dimenetpp_20M_rattled_forceonly.pt	|0.0614	|
|DimeNet++	|20M+MD         |https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_02/s2ef/dimenetpp_20M_md_forceonly.pt	|0.0594	|

## IS2RE models

|model	|split	|downloadable link	|val ID energy MAE	|
|---	|---	|---	|---	|
|CGCNN	|10k	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_02/is2re/cgcnn_10k.pt	|0.9881	|
|CGCNN	|100k	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_02/is2re/cgcnn_100k.pt	|0.682	|
|CGCNN	|All	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_02/is2re/cgcnn_all.pt	|0.6199	|
|DimeNet	|10k	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/is2re/dimenet_10k.pt	|1.0117	|
|DimeNet	|100k	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/is2re/dimenet_100k.pt	|0.6658	|
|DimeNet	|All	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/is2re/dimenet_all.pt	|0.5999	|
|SchNet	|10k	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_02/is2re/schnet_10k.pt	|1.059	|
|SchNet	|100k	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_02/is2re/schnet_100k.pt	|0.7137	|
|SchNet	|All	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_02/is2re/schnet_all.pt	|0.6458	|
|DimeNet++	|10k	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_02/is2re/dimenetpp_10k.pt	|0.8837	|
|DimeNet++	|100k	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_02/is2re/dimenetpp_100k.pt	|0.6388	|
|DimeNet++	|All	|https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_02/is2re/dimenetpp_all.pt	|0.5639	|

The Open Catalyst 2020 (OC20) dataset is licensed under a [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/legalcode).

Please cite the following paper in any research manuscript using the OC20 dataset or pretrained models:

```
@article{ocp_dataset,
    author = {Chanussot*, Lowik and Das*, Abhishek and Goyal*, Siddharth and Lavril*, Thibaut and Shuaibi*, Muhammed and Riviere, Morgane and Tran, Kevin and Heras-Domingo, Javier and Ho, Caleb and Hu, Weihua and Palizhati, Aini and Sriram, Anuroop and Wood, Brandon and Yoon, Junwoong and Parikh, Devi and Zitnick, C. Lawrence and Ulissi, Zachary},
    title = {Open Catalyst 2020 (OC20) Dataset and Community Challenges},
    journal = {ACS Catalysis},
    year = {2021},
    doi = {10.1021/acscatal.0c04525},
}
```
