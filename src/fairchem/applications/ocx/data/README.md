## Open Catalyst Experiments 2024 (OCx24) Datasets

### Experimental dataset
We include several experimental files relevant for analysis and understanding the data.

|Name | Description |
|---   |---    |
|[Raw Experimental Data](experimental_data/ExpDataDump_241113.csv)   | Raw experimental testing data for all samples. |
|[Clean Experimental Data](experimental_data/ExpDataDump_241113_clean.csv) | Experimental testing data excluding problematic samples (missing XRF, missing computational) directly used for downstream predictions. |
|[XRD Data](experimental_data/XRDDataDump-241113.csv) | XRD information for experimental samples. |
|[XRF Data](experimental_data/XRFDataDump-241113.csv) | XRF information for experimental samples. |

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
|[Batch Info](experimental_data/supporting_data/batches_info.csv)   | Provides some information about each of the synthesis batches |
|[Materials ID](experimental_data/supporting_data/materials_id-241113.csv) | Material ids and their important components for all samples |
|[Materials Postprocessing ID](experimental_data/supporting_data/materials_postprocessing_id-241113.csv) |  Important information about post-synthesis processing|
|[Synthesis Parameters](experimental_data/supporting_data/vsp_synthesis_params.csv)| Important information about the synthesis parameters used in the VSParticle syntheses |


### Computational dataset
#### Adsorption energy descriptors
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

#### Other computational data files
A few miscellanous files can be found in the [computational data folder](computational_data/). Their contents are as follows:
1. `her_candidates.csv`: a complete list of the potential HER catalysts found in this work
2. `cod_matches_lookup.pkl`: a lookup dictionary for COD materials. In some cases, the XRD data has matched to structures taken from COD. This file allows the lookup of a corresponding material in the computational pipeline if available.

### Processed dataset

As detailed in the manuscript, the raw experimental dataset included some aggregation steps to combine similar samples to be used as experimental targets for the predictive models. Here we provide the processed datasets that were directly used for the results in the manuscript.
Samples are considered matched if q-score > 70 and Rwp < 40.
Users are free to explore alternative aggregation beyond what was performed here. We provide these processed dataframes to directly reproduce results in the manuscript and simplify processing steps for users wanting to expand on the features provided here.

|Name | Description |
|---   |---    |
|[All CO2RR](processed_data/CO2R_40_70_all.csv)   | CO2RR experimental and computational data corresponding to all samples (XRD matched and not matched). |
|[Matched CO2RR](processed_data/CO2R_40_70_matched.csv)   | CO2RR experimental and computational data corresponding to only XRD matched samples. |
|[All HER](processed_data/HER_40_70_all.csv)   | HER experimental and computational data corresponding to all samples (XRD matched and not matched). |
|[Matched HER](processed_data/HER_40_70_matched.csv)   | HER experimental and computational data corresponding to only XRD matched samples. |

### Interactive XRD html files
|File | Size of compressed version (in bytes) |Size of uncompressed version (in bytes)    | MD5 checksum (download link)   |
|---   |---    |---    |---   |
|XRD interactive html files |  19.6G |  64.4G | [fdf37085325d663194a3ffffeb462c36](https://dl.fbaipublicfiles.com/opencatalystproject/data/ocx24/XRDData_241116.tar.gz)   | 

To support community interaction with our XRD data, we have included interactive html files that can help users derive insights about the XRD spectra. The file is a tarred directory where each folder contains subfolders and files related to XRD and XRF analysis. The contents are organized as follows:

Top level per Experimental XRD folder
* `raw`: contains the original and processed XRD data.
    * Raw file subfolder: files are in `.xy` format, representing the raw data acquired from XRD equipment
    * Processed XRD data subfolder: includes snapshots of the data after each processing step: background removal, substrate peak removal, and normalization
* `analysis`: contains data and files used for further analysis and dashboard creation
    * HTML file subfolder: contains all HTML files used to create the dashboards
    * XRF reasurement results: Multiple `xrf-<substrate>.csv` files containing XRF measurement results for samples on different substartes (e.g., wafer or gde).
    * Composition data: `composition.csv` file that combines XRF and XRD information
    * XRD data is csv format: `xrd.csv` file containing all XRD data in CSV format.
    * XRD analysis reports: Contains `xrd_report_raw` and `xrd_report_analysis` which are  XRD analysis results as obtained from the XRD analysis pipeline and XRD matches and evaluation metrics, respectively
    * Log file: `log.txt` which contains detailed information about the calculations conducted by the XRD analysis pipeline, step by step    
* `composition_analysis.html`: (VERY USEFUL) A dashboard page providing insights into the composition analysis 
* `xrd_dashboard.html`: (VERY USEFUL) A dashboard page displaying XRD analysis results and visualizations
* `xrd_renaming_sheet.csv`: A file used to archive name changes made to XRD files
 
This structure ensures that all raw data, processed data, analysis results, and dashboards are organized and easily accessible for further research and evaluation.
