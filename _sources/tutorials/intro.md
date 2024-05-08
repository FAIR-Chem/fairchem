---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

Intro and background on OCP and DFT
----------

## Abstract

The most recent, state of the art machine learned potentials in atomistic simulations are based on graph models that are trained on large (1M+) datasets. These models can be downloaded and used in a wide array of applications ranging from catalysis to materials properties. These pre-trained models can be used on their own, to accelerate DFT calculation, and they can also be used as a starting point to fine-tune new models for specific tasks. In this workshop we will focus on large, graph-based, pre-trained  machine learned models from the Open Catalyst Project (OCP) to showcase how they can be used for these purposes. OCP provides several pre-trained models for a variety of tasks related to atomistic simulations. We will explain what these models are, how they differ, and details of the datasets they are trained from.  We will introduce an Atomic Simulation Environment (ase) calculator that leverages an OCP pre-trained model for typical simulation tasks including adsorbate geometry relaxation, adsorption energy calculations, and reaction energies. We will show how pre-trained models can be fine-tuned on new data sets for new tasks. We will also discuss current limitations of the models and opportunities for future research. Participants will need a laptop with internet capability. 

## Walkthrough video 

<iframe width="560" height="315" src="https://www.youtube.com/embed/0wb0FTa7SV0?si=6gPgj0Pv9jfNT8Go" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

+++

## Introduction

Density functional theory (DFT) has been a mainstay in molecular simulation, but its high computational cost limits the number and size of simulations that are practical. Over the past two decades machine learning has increasingly been used to build surrogate models to supplement DFT. We call these models machine learned potentials (MLP) In the early days, neural networks were trained using the cartesian coordinates of atomistic systems as features with some success. These features lack important physical properties, notably they lack invariance to rotations, translations and permutations, and they are extensive features, which limit them to the specific system being investigated. About 15 years ago, a new set of features called symmetry functions were developed that were intensive, and which had these invariances. These functions enabled substantial progress in MLP, but they had a few important limitations. First, the size of the feature vector scaled quadratically with the number of elements, practically limiting the MLP to 4-5 elements. Second, composition was usually implicit in the functions, which limited the transferrability of the MLP to new systems. Finally, these functions were "hand-crafted", with limited or no adaptability to the systems being explored, thus one needed to use judgement and experience to select them. While progess has been made in mitigating these limitations, a new approach has overtaken these methods.

Today, the state of the art in machine learned potentials uses graph convolutions to generate the feature vectors. In this approach, atomistic systems are represented as graphs where each node is an atom, and the edges connect the nodes (atoms) and roughly represent interactions or bonds between atoms. Then, there are machine learnable convolution functions that operate on the graph to generate feature vectors. These operators can work on pairs, triplets and quadruplets of nodes to compute "messages" that are passed to the central node (atom) and accumulated into the feature vector. This feature generate method can be constructed with all the desired invariances, the functions are machine learnable, and adapt to the systems being studied, and it scales well to high numbers of elements (the current models handle 50+ elements). These kind of MLPs began appearing regularly in the literature around 2016.

Today an MLP consists of three things:

1. A model that takes an atomistic system, generates features and relates those features to some output.
2. A dataset that provides the atomistic systems and the desired output labels. This label could be energy, forces, or other atomistic properties.
3. A checkpoint that stores the trained model for use in predictions.

The [Open Catalyst Project (OCP)](https://github.com/Open-Catalyst-Project) is an umbrella for these machine learned potential models, data sets, and checkpoints from training. 

+++

### Models

OCP provides several [models](../core/models). Each model represents a different approach to featurization, and a different machine learning architecture. The models can be used for different tasks, and you will find different checkpoints associated with different datasets and tasks. 

+++

### Datasets / Tasks

OCP provides several different datasets like [OC20](../core/datasets/oc20) that correspond to different tasks that range from predicting energy and forces from structures to Bader charges, relaxation energies, and others.

+++

### Checkpoints

To use a pre-trained model you need to have [ocp](https://github.com/Open-Catalyst-Project/ocp) installed. Then you need to choose a checkpoint and config file which will be loaded to configure OCP for the predictions you want to make. There are two approaches to running OCP, via scripts in a shell, or using an ASE compatible calculator.

We will focus on the ASE compatible calculator here. To facilitate using the checkpoints, there is a set of [utilities](./ocp-tutorial) for this tutorial. You can list the checkpoints that are readily available here:

```{code-cell} ipython3
from fairchem.core.models.model_registry import available_pretrained_models
print(available_pretrained_models)
```

You can get a checkpoint file with one of the keys listed above like this. The resulting string is the name of the file downloaded, and you use that when creating an OCP calculator later.

```{code-cell} ipython3
from fairchem.core.models.model_registry import model_name_to_local_file

checkpoint_path = model_name_to_local_file('GemNet-OC-S2EFS-OC20+OC22', local_cache='/tmp/ocp_checkpoints/')
checkpoint_path
```

## Goals for this tutorial

This tutorial will start by using OCP in a Jupyter notebook to setup some simple calculations that use OCP to compute energy and forces, for structure optimization, and then an example of fine-tuning a model with new data.

+++

## About the compute environment

`ocpmodels.common.tutorial_utils`  provides `describe_ocp` to output information that might be helpful in debugging.

```{code-cell} ipython3
from fairchem.core.common.tutorial_utils import describe_ocp
describe_ocp()
```

Let's get started! Click here to open the [next notebook](./OCP-introduction).
