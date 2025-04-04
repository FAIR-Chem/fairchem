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

GNNs for Chemistry
----------

The most recent, state of the art machine learned potentials in atomistic simulations are based on graph models that are trained on large (1M+) datasets. These models can be downloaded and used in a wide array of applications ranging from catalysis to materials properties. These pre-trained models can be used on their own, to accelerate DFT calculation, and they can also be used as a starting point to fine-tune new models for specific tasks. 

# Background on DFT and machine learning potentials

Density functional theory (DFT) has been a mainstay in molecular simulation, but its high computational cost limits the number and size of simulations that are practical. Over the past two decades machine learning has increasingly been used to build surrogate models to supplement DFT. We call these models machine learned potentials (MLP) In the early days, neural networks were trained using the cartesian coordinates of atomistic systems as features with some success. These features lack important physical properties, notably they lack invariance to rotations, translations and permutations, and they are extensive features, which limit them to the specific system being investigated. About 15 years ago, a new set of features called symmetry functions were developed that were intensive, and which had these invariances. These functions enabled substantial progress in MLP, but they had a few important limitations. First, the size of the feature vector scaled quadratically with the number of elements, practically limiting the MLP to 4-5 elements. Second, composition was usually implicit in the functions, which limited the transferrability of the MLP to new systems. Finally, these functions were "hand-crafted", with limited or no adaptability to the systems being explored, thus one needed to use judgement and experience to select them. While progess has been made in mitigating these limitations, a new approach has overtaken these methods.

Today, the state of the art in machine learned potentials uses graph convolutions to generate the feature vectors. In this approach, atomistic systems are represented as graphs where each node is an atom, and the edges connect the nodes (atoms) and roughly represent interactions or bonds between atoms. Then, there are machine learnable convolution functions that operate on the graph to generate feature vectors. These operators can work on pairs, triplets and quadruplets of nodes to compute "messages" that are passed to the central node (atom) and accumulated into the feature vector. This feature generate method can be constructed with all the desired invariances, the functions are machine learnable, and adapt to the systems being studied, and it scales well to high numbers of elements (the current models handle 50+ elements). These kind of MLPs began appearing regularly in the literature around 2016.

Today an MLP consists of three things:

1. A model that takes an atomistic system, generates features and relates those features to some output.
2. A dataset that provides the atomistic systems and the desired output labels. This label could be energy, forces, or other atomistic properties.
3. A checkpoint that stores the trained model for use in predictions.


# FAIR Chemistry models

FAIRChem provides a number of GNNs in this repository. Each model represents a different approach to featurization, and a different machine learning architecture. The models can be used for different tasks, and you will find different checkpoints associated with different datasets and tasks. Read the papers for details, but we try to hihglight here the core ideas and advancements from one model the next.  

## equivariant Smooth Energy Network (eSEN)

**Core Idea:** Scaling GNNs to train on hundreds of millions of structures required a number of engineering decisions that led to SOTA models for some tasks, but led to challenges in other tasks. eSEN started with the eSCN network, carefully analyzed which decisions were necessary to build smooth and energy conserving models, and used those learnings to train a new model that is SOTA (as of early 2025) across many domains. 

**Paper:** https://arxiv.org/abs/2502.12147

**Code:** Models available in this repo as a PR!

## Equivariant Transformer V2 (EquiformerV2)

**Core Idea:** We adapted and scaled the Equiformer model to larger datasets using a number of small tweaks/tricks to accelerate training and inference, and incorporating the eSCN convolution operation. This model was also the first shown to be SOTA on OC20 without requiring the underlying structures to be tagged as surface/subsurface atoms, a major improvement in usability. 

**Paper:** https://arxiv.org/abs/2306.12059

**Code:** Models available in this repo!

## Equivariant Spherical Channel Network (eSCN)

**Core Idea:** The SCN network was high performance, but the approach broke equivariance in the resulting models. eSCN enabled equivariance in these models, and introduced an SO(2) convolution operation that allowed the approach to scale to even higher order spherical harmonics. The model was shown to be equivariant in the limit of infinitely find grid for the convolution operation.

**Paper:** https://proceedings.mlr.press/v202/passaro23a.html

**Code:** Models available in this repo!

## Spherical Channel Network (SCN)

**Core Idea:** We developed a message convolution operation, inspired by the vision AI/ML community, that led to more scalable networks and allowed for higher-order spherical harmonics. This model was SOTA on OC20 on release, but introduced some limitations in equivariance addressed later by eSCN. 

**Paper:** https://proceedings.neurips.cc/paper_files/paper/2022/hash/3501bea1ac61fedbaaff2f88e5fa9447-Abstract-Conference.html

**Code:** Models available in this repo!

## GemNet-OC

**Core Idea:** GemNet-OC is a faster and more scalable version of GemNet, a model that incorporated some clever features like triplet/quadruplet information into GNNs, and provided SOTA performance when released on OC20. 

**Paper:** https://arxiv.org/abs/2204.02782

**Code:** Models available in this repo!

