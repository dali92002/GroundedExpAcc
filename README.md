# Grounded Explanation Accuracy (GEA) Calculation

This project implements a method to calculate the Grounded Explanation Accuracy (GEA) using the Pix2Struct model for document-based visual question answering (DocVQA). The GEA metric evaluates the model's performance in understanding and justifying its answers based on specific regions within an image.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

To get started, clone this repository and install the necessary dependencies.

```bash
pip install -r requirements.txt
```
## Data
The input data format is indicated in the following JSON. For this metric, the `"questionId"`, image path (`"image"`), `"question"`, `"answers"` and `"evidence"` are necessary. These are set as following, noting that the other fields are optional and the whole structure can be modified according to the needs.
```json
{
    "data": [
        {
            "questionId": 16404,
            "question": "What is the title of William M. Coleman ?",
            "image": "examples/fryn0081_p8.jpg",
            "answers": [
                "Sr. R&D Chemist"
            ],
            "evidence": {
                "positions": [[0.0843, 0.5203, 0.1508, 0.1897], [0.0855, 0.4330, 0.1810, 0.2176]]
            },
            ...
        },
        ...
    ]
}
```
## Annotated Data (for small test)
Few examples of SPDocVQA were anotated, download the data from [here](https://drive.google.com/file/d/1DadDauqHt0N1a5rlTi6csSsnrZEP-DEe/view?usp=sharing)

## Usage

This script allows you to evaluate GEA for a set of image samples using a pre-trained Pix2Struct model. The model takes in a document image, questions, and evidence bounding boxes and calculates the ANLS score for both visible and occluded image regions.

To use the script run the following, you can change args for different settings.

```bash
python gea.py --data_path PATH/TO/DATA --set_name ann_pilot.json --batch_size 22 --method perturb --model_name pix2struct
```

There are also other command-Line Arguments:

- `--model_name`: (Optional) the name of the model to be used (default: "pix2struct" and you can use also "gpt", you have to specify your API_CREDENTIALS_PATH in this case and which GPT version you want to use), check the beginnig of the file [simple_gpt.py](simple_gpt.py).
- `--method`: (Optional) which method or box manupilation to use `shrink` or `perturb`:
  -  `shrink`: shrinking the box gradually
  - `perturb`: masking some parts of the box randomely and gradually ugment the masked area.
- `--set_name`: Path to the JSON file containing the image names, questions, answers, and evidence bounding boxes, etc. 
- `--plot_auc`: (Optional) Set to `True` to generate plots showing the AUC (Area Under the Curve) for the visible and occluded regions. Default is `False`.

The script will output the GEA which represent the Grounded Explanation Accuracy. The maximum accuracy is `1`. 

If `plot_auc` is enabled, the script will also generate a plot comparing the AUC curves for the visible (shown) and occluded (hidden) regions. The plot is saved as `gea_plot_ID.png` in the `examples/` directory.
