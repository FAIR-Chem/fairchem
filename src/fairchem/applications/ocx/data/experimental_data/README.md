## File overview

### Primary files
|Name | Description |
|---   |---    |
|`ExpDataDump_YYMMDD.csv`  | Raw experimental testing data for all samples. |
|`ExpDataDump_YYMMDD_clean.csv` | Experimental testing data excluding problematic samples (missing XRF, missing computational) directly used for downstream predictions. |
|`XRDDataDump-YYMMDD.csv` | XRD information for experimental samples. |
|`XRFDataDump-YYMMDD.csv` | XRF information for experimental samples. |


### Supporting Data
A couple extra files that can be found in the [supporting data folder](supporting_data/)

1. `batches_info.csv`: This file provides information about synthesized batches of materials. It includes a "source" column, which contains a code name indicating the method of material synthesis: "uoft" for chemical reduction and "vsp" for spark ablation. The file also records the date of synthesis and lists the elements used as precursors for alloying.
2. `materials_id-241113.csv`: This file contains Sample IDs, which include information such as the batch code name, the target composition of the synthesized material, and the post-processing ID. The post-processing ID links to the materials_postprocessing_id-241113.csv file, providing details on the exact temperature and annealing procedure used for processing the samples after synthesis.
3. `materials_postprocessing_id-241113.csv`: This file provides detailed information on the post-processing procedures applied to the synthesized materials. It includes data on the specific temperatures and annealing procedures used.
4. `vsp_synthesis_params.csv`: This file contains detailed information about the synthesis parameters used for spark ablation. It includes data on the number of metal rods used in synthesis, the composition of the rods, their diameter, the applied voltage (kV), the applied current (mA), and the speed of printing (in micrometers per second).
