# DUNES
![Title Image](https://github.com/katesanders9/dai/blob/master/title-im.png?raw=true)

This repository contains code and data for DUNES (Dataset of UNcertain Event Scenes), a collection of 12,000 event-based ambiguous images extracted from videos. All images are annotated with ground truth values and a test set is annotated with 10,800 human uncertainty judgments. The dataset is detailed in the paper **Ambiguous Images With Human Judgments for Robust Visual Event Classification** ([PDF linked here](https://openreview.net/forum?id=6Hl7XoPNAVX)). In the paper, it is illustrated how DUNES can be used to explore human uncertainty quantification, train robust models, and directly evaluate models and model calibration techniques. The repository includes dataset information, a dataset loader, and code used for experiments in the paper.

## Overview
### Code Directory
```
dunes
│   README.md    # Repository documentation
|   setup.py     # Script to install repository code
│   
└───dataset
│   │   
|   └───data
|   |   |   dataset.csv            # Video IDs, frame numbers, ground truth labels
|   |   |   datasheet.pdf          # Dataset documentation
|   |   |   huj_annotations.csv    # Task variant, event prompt, video ID, frame number, human judgments
|   |
|   └───data_scripts
|       |   extract_images.py      # Script to extract dataset images from downloaded videos
|       |   load_yt.py             # Script to download YouTube videos used in DUNES
|       |   load_ucf.sh            # Script to download UCF videos used in DUNES
|       |   requirements.txt       # Requirements needed for running data scripts (included in setup.py)
│
└───experiments
    |   indices.json    # Dictionary of data indices used in paper experiments
    │   setup_alt.py    # Script to install code if using PyTorch >= 1.12.0 train.py filters
    |
    └───models
    |   |   __init__.py            # Init file
    |   |   mc_model.py            # Monte-Carlo dropout version of rn_model.py
    |   |   rn_model.py            # Basic ResNet-based classification model
    |
    └───tools
        |   train.py               # Script to train and evaluate a model for Section 5.1 experiments
        |   train_uq.py            # Script to train and evaluate a model for Section 5.3 experiments
    
```

## Installation

It is recommended that you set up a virtual environment for installation. All code was run using Python 3.7.11. 

Run `python setup.py develop` to install the required packages and DUNES code.

## Dataset


## Experiments
### Section 5.1
The experiments in this section can be run with the command `python tools/train.py`. At the top of the script there is a list of pathname variables that must be assigned to proper paths to the downloaded images, train/test indices (stored as a json), and image labels (also should be stored as a json). Example files for both of these are provided in the `indices` folder. To select specific augmentation setups to evaluate, the list of augmentation filters can be modified at the end of the script.

Note: Some of the augmentation filters used for this experiment require PyTorch >= 1.12.0. An alternate `setup.py` file (`setup_alt.py`) is provided for running these experiments. It is recommended to use a separate environment for this.

### Section 5.2
Experiments were run using the respective codebases for the three situation recognition models that were evaluated. Their GitHub repositories are listed here:

- **Grounded Situation Recognition**  [code](https://github.com/allenai/swig) | [paper](https://arxiv.org/abs/2003.12058)
- **Grounded Situation Recognition with Transformers**  [code](https://github.com/jhcho99/gsrtr) | [paper](https://arxiv.org/abs/2111.10135)
- **Collaborative Transformers for Grounded Situation Recognition**  [code](https://github.com/jhcho99/CoFormer) | [paper](https://arxiv.org/abs/2203.16518)

### Section 5.3
The experiments in this section can be run with the command `python tools/train_uq.py`. As with the script for Section 5.1, there is a set of paths at the top of the script that must be filled in. Again, to select specific model calibration techniques to evaluate, the list of methods at the bottom of the script can be modified.

## Acknowledgements
Our code builds on the following research repositories:
- [Grounded Situation Recognition](https://github.com/allenai/swig)
- [Grounded Situation Recognition with Transformers](https://github.com/jhcho99/gsrtr)
- [Collaborative Transformers for Grounded Situation Recognition](https://github.com/jhcho99/CoFormer)
- [Focal Calibration](https://github.com/torrvision/focal_calibration)
- [Being Bayesian about Categorical Probability](https://github.com/tjoo512/belief-matching-framework)

## Citation
If you find this dataset or code useful in your research, please consider citing the paper:
```
@article{sandersambiguous,
  title={Ambiguous Images With Human Judgments for Robust Visual Event Classification},
  author={Sanders, Kate and Kriz, Reno and Liu, Anqi and Van Durme, Benjamin}
}
```
