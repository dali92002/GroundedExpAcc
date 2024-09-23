# Grounded Explanation Accuracy (GEA) Calculation

This project implements a method to calculate the Grounded Explanation Accuracy (GEA) using the Pix2Struct model for document-based visual question answering (DocVQA). The GEA metric evaluates the model's performance in understanding and justifying its answers based on specific regions within an image.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

To get started, clone this repository and install the necessary dependencies.

```bash
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt
```
## Usage

This script allows you to evaluate GEA for a set of image samples using a pre-trained Pix2Struct model. The model takes in a document image, questions, and evidence bounding boxes and calculates the ANLS score for both visible and occluded image regions.

To use the script run the following

```bash
python gea.py --set_path examples/example.json
```

There are also other command-Line Arguments:

- ```--model_name```: (Optional) the name of the model to be used (default: "pix2struct").
- ```--set_path```: Path to the JSON dataset containing the image paths, questions, answers, and evidence bounding boxes.
- ```plot_auc```: (Optional) chose whether to plot the AUC of both show and hide images or no.

The script will output the GEA which represent the Grounded Explanation Accuracy. The maximum accuracy is 1. 