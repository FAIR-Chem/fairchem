# Generative Models

The FAIR chemistry team has released and published four generative models for inorganic materials and molecules. Close collaborators have also released generative models for catalysts. These releases currently live outside of the main FAIR chemistry repo (noted where appropriate). Much of this work has been driven by an incredible group of Meta/FAIR summer PhD interns!

## All-atom Diffusion Transformers (ADiT)

**Core Idea:** Generative models for molecules and generative models for materials tended to be two separate tasks in the AI/ML community, and we developed a transformer-based latent diffusion approach that was able to encode both in the same latent space, leading to synergistic learning. 

**Paper:** https://arxiv.org/abs/2503.03965

**Code:** https://github.com/facebookresearch/all-atom-diffusion-transformer

## FlowLLM

**Core Idea:** We noticed that the Crystal-text-llm did much better at generating compositions than the actual crystal structures, so we took a best-of-both-worlds approach using an LLM to generate compositions we should study, and Flow Matching to generate the actual crystal structures. 

**Paper:** https://arxiv.org/abs/2410.23405

**Code:** https://github.com/facebookresearch/flowmm/

## FlowMM

**Core Idea:** Flow Matching, an emerging methods in the broader AI/ML generative model space, could be used to more quickly and efficiently generate inorganic crystal structures than some prior diffusion-based methods. 

**Paper:** https://openreview.net/forum?id=W4pB7VbzZI

**Code:** https://github.com/facebookresearch/flowmm/

## Crystal-text-llm

**Core Idea:** Building on others in the community who had suggested LLMs could generate molecules or materials as text, we showed that fine-tuned LLaMA models could work quite well for this task, and enable text conditioning.

**Paper:** https://arxiv.org/abs/2402.04379

**Code:** https://github.com/facebookresearch/crystal-text-llm


