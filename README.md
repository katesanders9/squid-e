# 🦑EventSQUID

This repository contains code and data for EventSQUID (Event Scenes with Quantitative Uncertainty Information Dataset), a collection of 12,000 event-based ambiguous images extracted from videos. All images are annotated with ground truth values and a test set is annotated with 10,800 human uncertainty judgments. The dataset is detailed in the paper **Ambiguous Images With Human Judgments for Robust Visual Event Classification** ([PDF here](https://openreview.net/forum?id=6Hl7XoPNAVX)). The paper illustrates how EventSQUID can be used to explore human uncertainty quantification, train robust models, and directly evaluate models and model calibration techniques. The repository includes dataset information, a dataset loader, and code used for experiments in the paper.

![Title Image](title-im.png?raw=true)

## Overview
### Code Directory
```
eventsquid
|   title-im.png   # Title page figure
│   README.md      # Repository documentation
|   setup.py       # Script to install repository code
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

Run `python setup.py develop` to install the required packages and EventSQUID code.

## Dataset
### Dataset statistics
- 12,000 images
- 20 distinct event types
- 2,000 online videos used
- 1,800 human-labeled test set images
- 6 test set event types
- 10,800 test set human uncertainty judgments

### Event types
|Baseball  |Basketball   |Birthday Parties   |Cooking   |COVID Tests   |
|:---:|:---:|:---:|:---:|:---:|
|__Cricket__   |__Fires__   |__Fishing__   |__Gardening__   |__Graduation Ceremonies__   |
|__Hiking__   |__Hurricanes__   |__Medical Procedures__   |__Music Concerts__   |__Parades__   |
|__Protests__ |__Soccer/Football__   |__Tennis__   |__Tsunamis__   |__Weddings__   |

Event types included in the human-annotated test set are *Birthday Parties*, *COVID Tests*, *Medical Procedures*, *Parades*, *Protests*, and *Weddings*.

### Details
Our dataset explicitly consists of two parts: (1) Necessary information for downloading and extracting the images used for the dataset and (2) grount truth and human judgment annotations for each image. Each image is identified by a video ID and a frame number, since we do not own or have legal rights to the images/videos used for this dataset and cannot distribute them directly. We provide the URLs for all the videos, frame numbers corresponding to the images extracted from each video (both located in the `dataset/data/dataset.csv` file), and scripts to download these videos from the URLs and extract the appropriate frames (located in `data/data_scripts`).

The human judgment annotations (located in `dataset/data/huj_annotations.csv`) consist of the video ID and frame number, event prompt given to annotators, and three integer values within [0,100] measuring an individual annotators' confidence that the image depicts the event type prompt. All images in the test set have at least 3 annotation scores where the event prompt is the image's ground truth event type.

A datasheet following the format introduced by [Gebru et al.](https://arxiv.org/abs/1803.09010) is included (`dataset/data/datasheet.pdf`) that covers information regarding dataset motivation, collection, and intended use. **We also encourage researchers to review the limitations and ethical considerations regarding this dataset discussed in [Section 6 of the paper](https://openreview.net/forum?id=6Hl7XoPNAVX).**

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
