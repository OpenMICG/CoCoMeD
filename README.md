# Consistency Conditioned Memory Augmented Dynamic Diagnosis Model for Medical Visual Question Answering
Implementation of the CoCoMeD model.

## Data Preparation
- DME
  - Download [DME](https://zenodo.org/record/6784358) dataset 
  - Place the zip file in any location and unzip it
  - Put it under your data path
- C-SLAKE
  - Download [C-SLAKE](https://github.com/OpenMICG/CSLAKE)dataset
  - Put it under your data path

## Setup
    conda create -n CoCoMeD python=3.9
    conda activate CoCoMeD
    pip install -r requirements.txt
  >**Note**: after cloning the repository, create a new environment named CoCoMeD with Python 3.9, activate it and then install the 
> required packages.


## Configuration file
In the folder `config/idrid_regions/single/` you can find different configuration files that correspond to different scenarios, as shown in Table 1 of our paper. More specifically, you can find the following configuration files:

<p align="center">


| Config file                             | DATASET | Consistency method                                                                                                                                                                     |
|-----------------------------------------|---------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| default_baseline.yaml                   | DME     | None                                                                                                                                                                                   |
| default_squint.yaml                     | DME     | [SQuINT](https://openaccess.thecvf.com/content_CVPR_2020/papers/Selvaraju_SQuINTing_at_VQA_Models_Introspecting_VQA_Models_With_Sub-Questions_CVPR_2020_paper.pdf) by Selvaraju et al. |
| default_consistency.yaml                | DME     | [CPQA](https://github.com/sergiotasconmorales/consistency_vqa) by Sergio Tascon-Morales et al.                                                                                         |
| default_CoCoMeD.yaml                    | DME     | None                                                                                                                                                                                   |
| default_CoCoMeD_consistency.yaml        | DME     | ours                                                                                                                                                                                   |
| default_CoCoMeD_CSLAKE_consistency.yaml | C-SLAKE | ours                                                                                                                                                                                   |

</p>

In order to use a configuration file to train, you must first change the fields `path_img`, `path_qa` and `path_masks` to match the path to the downloaded data `<path_data>`. Please note that with these configuration files you should obtain results that are similar to the ones reported in our paper. However, since we reported the average for 10 runs of each model, your results may deviate. 


## Training
To train a model just run the following command:

    train.py --path_config <path_config>

Example:
    
    train.py --path_config config/idrid_regions/single/default_baseline.yaml

After training, the `logs` folder, as defined in the YAML file, will contain the results of the training. This includes the model weights for the best and last epoch, as well as the answers produced by the model for each epoch. Additionally, a JSON file named `logbook` will be generated, which contains the information from the config file and the values of the metrics (loss and performance) for each epoch.

## Inference for test set
In order to do inference on the test set, use the following command:

    inference.py --path_config <path_config>

The inference results are stored in the `logs` folder, as defined in the config file, in the sub-folder answers. In total 6 answer files are generated, as follows:

<p align="center">

| File name      | Meaning |
| ----------- | ----------- |
| answers_epoch_0.pt     | best model on test set      |
| answers_epoch_2000.pt   | best model on val set     |
| answers_epoch_1000.pt   | best model on train set        |
| answers_epoch_1000.pt   | best model on train set        |
| answers_epoch_2001.pt   | last model on val set        |
| answers_epoch_1001.pt   | last model on train set        |

</p>

Each of these files contains a matrix with two columns, the first one representing the question ID, and the second one corresponding to the answer provided by the model. The answer is an integer. To convert from integer to the textual answer, a dictionary is given in `<path_data>/processed/map_index_answer.pickle`

## Inference for a single sample
The following command allows you to do inference on a single sample using a previously trained model (as specified by the config file in `<path_config>`):

    inference_single.py --path_config <path_config> --path_image <path_image> --path_mask <path_mask> --question <question>



## Plotting metrics and learning curves
To plot learning curves and accuracy, use the following command after having trained and done inference:

    plotter.py --path_config <path_config>

The resulting plots are stored in the `logs` folder. 


## Computing consistency

After running the inference script, you can compute the consistency using:

    compute_consistency.py --path_config <path_config>

By default, this only computes the consistency C1 (see paper). To compute the consistency C2 as well, set the parameter `q3_too` to True when calling the function `compute_consistency`in the script `compute_consistency.py`.

<br />
<br />



## Acknowledgement
The implementation of CoCoMeD relies on [MVQA-CPQA](https://github.com/sergiotasconmorales/consistency_vqa). We use PyTorch as our deep learning framework. 
We thank the original authors for their work and open source code.
