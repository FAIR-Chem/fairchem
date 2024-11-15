## Open Catalyst Experiments 2024 (OCx24): Bridging Experiments and Computational Models
![summary figure](co2rr_summary_figure.png)
In this work, we seek to directly bridge the gap between computational descriptors and experimental outcomes for heterogeneous catalysis. We consider two important green chemistries: the hydrogen evolution reaction and the electrochemical reduction of carbon dioxide. To do this, we created a curated dataset of experimental results with materials synthesized and tested in a reproducible manner under industrially relevant conditions. We used this data to build models to directly predict experimental outcomes using computational features. For more information, please read the manuscript [paper](???).

### Datasets
To support this work, we performed X-ray fluorescence (XRF), X-ray diffraction (XRD), and electrochemical testing. Summaries of this data is all publically available in this repository in `fairchem/src/fairchem/applications/ocx/data/experimental_data/`. The computational data is also publicly available at the following link. For details of the contents of these files, please see the Dataset details section below. A csv of the full list of HER candidates has also been included in `fairchem/src/fairchem/applications/ocx/data/`.

|Splits |Size of uncompressed version (in bytes)    | MD5 checksum (download link)   |
|---   |---    |---    |
|Screening data CSV   |1.5G  | [9e75b95bb1a2ae691f07cf630eac3378](https://dl.fbaipublicfiles.com/opencatalystproject/data/ocx24/comp_df_241022.csv)   |


### Citing this work

If you use this codebase in your work, please consider citing:

```bibtex
@article{
}
```

### Dataset details
#### Experimental datasets
Electrochemical testing data:
- `sample id`: An identifier for each experiment
- `source`: The material source. `vsp` = VSParticle spark ablation, `uoft` = University of Toronto chemical reduction
- `batch number`: The material batch number (used for internal tracking)
- `batch date`: The date of preparation
- `composition`: The target composition of the sample
- `post processing id`: The post-synthesis processing conditions (i.e. annealing conditions)
- `rep`: The replicate number
- `total reps`: The total number of replicates
- `reaction`: The reaction type (either CO2R or HER)
- `current density`: The current density the experiment was performed at
- `voltage`: The full cell voltage
- `fe_{}`: The Faradaic efficiency of each {product}

XRD testing data:
- `xrd dir`: the directory of raw XRD data (for internal tracking).
- `sample id`: sample identifier.
- `source`: The material source. `vsp` = VSParticle spark ablation, `uoft` = University of Toronto chemical reduction.
- `batch number`: The material batch number (used for internal tracking).
- `batch date`: The date of preparation.
- `target comp`: The target composition of the sample.
- `xrf comp`: The XRF composition measurement.
- `xrf offset`: The offset between the target and XRF composition.
- `xrf stdev`: The standard devation across three replicate XRF measurements.
- `allsolutions_cifids`: a list of lists. Each entry corresponds to a solution of matched XRD bulk id(s).
- `allsolutions_formulas`: a list of lists. Each entry corresponds to a solution of matched XRD composition(s).
- `allsolutions_weights`: a list of lists. The weights of the component phases.
- `allsolutions_rwp`: a list of the goodness of fit RWP from Rietveld refinement for each solution.
- `total no of solutions`: integer, the length of all of the allsolutions entries

`solutions_target` corresponds to the subset of solutions with XRF within twice the standard deviation of the composition and xrd matches as a major phase (>70 wt.%)
q-score is a measure of closeness to the best solution. The actual expression is included in the supplementary materials of the manuscript.

- `solutions_target_cifids`: solution cifids list
- `solutions_target_qrankings`: the q-scores of all subset solutions
- `solutions_target_qscore`: the best q-score
- `solutions_target_formulas`: the compositions of all subset solutions
- `solutions_target_rwp`: the RWP of all subset solutions
- `solutions_target_wt`: The weight of the target phase for all subset solutions
- `solutions_target_rankings`: The ranking of the subset of solutions 


#### Computational Dataset
To build models capable of predicting experimental outcomes, we screened 19,406 materials as potential catalyst candidates by calculating C, H, OH, CO, COH, and COCOH adsorption energies on their surfaces up to Miller index 2. This data is available publicly in the file below.

The file contains the following columns where each row corresponds to a unique surface:
- `bulk_id`: A material identifier which corresponds to the bulk material structure. They begin with `mp-`, `oqmd-`, and `dcgat-` indicating the bulk material database source (Materials Project, Open Quantum Materials Database, and Alexandria, respectively).
- `slab_millers`: The miller indices of the surface
- `slab_shift`: The shift where the slab was cut (there may be multiple slabs for unique miller indices)
- `top`: True if the top of the slab, otherwise False. Most slabs are asymmetric meaning the top and bottom are unique. This along with `slab_shift` and `slab_millers` uniquely defines the surface.
- `slab_comp`: The composition of the material as a string (e.g. "Pt-0.5-Au-0.5")
- `{}_ml_Eads`: The five lowest energies by ML for {} adsorbate. N may be less than five if there were fewer than five valid ML structures
- `{}_ml_dft_Eads`: The DFT SPs on the five lowest energy structures by ML for {} adsorbate. N may be less than five if there were fewer ML structures or DFT SPs did not converge.
- `{}_min_ml_Eads`: The minimum adsorption energy by ML for {} adsorbate
- `{}_min_sp_e`: The energies used in modeling for {} adsorbate. The minimum adsorption energy on the surface from DFT SPs (and the AdsorbML pipeline)
- `n_bulk_elements`: The number of bulk elements. One corresponds to unary, two to binary and so on.
- `slab_identifier`:  A combination of the bulk_id and all information to uniquely identify a slab which makes a unique surface identifier.
- `cleavage_energy`: The energy associated with cleaving the slab from the bulk. We only calculated this for compositions experimentally tested, so most values are nans.
- `facet_fraction_on_wulff_not_normalized`: Using the cleavage energy, this column contains the proportion of surface area on the Wulff shape for each facet where available. For one material, this may not sum to 1. We calculated cleavage energies up to Miller 3, but only adsorption energies up to 2. There were also instances where the adsorption energy was otherwise incomplete for a given surface. This gives instances where surfaces that may appear on Wulff are not represented.