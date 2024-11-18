## Open Catalyst Experiments 2024 (OCx24) Datasets

### Experimental dataset
We include several experimental files relevant for analysis and understanding the data.

|Name | Description |
|---   |---    |
|[Raw Experimental Data](data/experimental_data/ExpDataDump_241113.csv)   | Raw experimental testing data for all samples. |
|[Clean Experimental Data](data/experimental_data/ExpDataDump_241113_clean.csv) | Experimental testing data excluding problematic samples (missing XRF, missing computational) directly used for downstream predictions. |
|[XRD Data](data/experimental_data/XRDDataDump-241113.csv) | XRD information for experimental samples. |
|[XRF Data](data/experimental_data/XRFDataDump-241113.csv) | XRF information for experimental samples. |

#### Dataset details
##### Experimental Data
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

##### XRD Data
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

##### Experimental Metadata
We provide additional metadata corresponding to experimental samples. While this is not used for the analysis, it does provide more thorough details on the experimental samples.
|Name | Description |
|---   |---    |
|[Batch Info](data/experimental_data/supporting_data/batches_info.csv)   |  |
|[Materials ID](data/experimental_data/supporting_data/materials_id-241113.csv) |  |
|[Materials Postprocessing ID](data/experimental_data/supporting_data/materials_postprocessing_id-241113.csv) |  |


### Computational dataset
To build models capable of predicting experimental outcomes, we screened 19,406 materials as potential catalyst candidates by calculating C, H, OH, CO, COH, and COCOH adsorption energies on their surfaces up to Miller index 2. This data is available publicly in the file below.

|Splits |Size of uncompressed version (in bytes)    | MD5 checksum (download link)   |
|---   |---    |---    |
|Computational screening data |1.5G  | [9e75b95bb1a2ae691f07cf630eac3378](https://dl.fbaipublicfiles.com/opencatalystproject/data/ocx24/comp_df_241022.csv)   |

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

### Processed dataset

As detailed in the manuscript, the raw experimental dataset included some aggregation steps to combine similar samples to be used as experimental targets for the predictive models. Here we provide the processed datasets that were directly used for the results in the manuscript.
Samples are considered matched if q-score > 70 and Rwp < 40.
Users are free to explore alternative aggregation beyond what was performed here. We provide these processed dataframes to directly reproduce results in the manuscript and simplify processing steps for users wanting to expand on the features provided here.

|Name | Description |
|---   |---    |
|[All CO2RR](data/processed_data/CO2R_40_70_all.csv)   | CO2RR experimental and computational data corresponding to all samples (XRD matched and not matched). |
|[Matched CO2RR](data/processed_data/CO2R_40_70_matched.csv)   | CO2RR experimental and computational data corresponding to only XRD matched samples. |
|[All HER](data/processed_data/HER_40_70_all.csv)   | HER experimental and computational data corresponding to all samples (XRD matched and not matched). |
|[Matched HER](data/processed_data/HER_40_70_all.csv)   | HER experimental and computational data corresponding to only XRD matched samples. |
