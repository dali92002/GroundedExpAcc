import os
import json
import random
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from args import create_args
from utilts import calculate_gea_accuracy

SEED = 0
GRID_SIZE = 3  # 3x3 grid
COUNTED = 1 # number of samples to be counted for this test (for demonstration), set -1 for full dataset.

def load_model_and_processor(model_name, device):
    """
    Load the pre-trained model and processor based on the model name.
    
    Args:
        model_name (str): Name of the model to load.
        device (str): Device to load the model on ('cpu' or 'cuda').
        
    Returns:
        Tuple[model, processor]: Loaded model and processor or None if model not supported.
    """
    if model_name == "pix2struct":
        model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-docvqa-base").to(device)
        processor = Pix2StructProcessor.from_pretrained("google/pix2struct-docvqa-base")
        return model, processor
    elif model_name == "gpt":
        return None, None
    else:
        raise ValueError("Unsupported model. Choose either 'pix2struct' or 'gpt'.")

def save_results(result, show_anls, hide_anls, method, question_id):
    """
    Save GEA results to a JSON file.
    
    Args:
        result (float): GEA accuracy score.
        show_anls (list): ANLS scores for visible regions.
        hide_anls (list): ANLS scores for hidden regions.
        method (str): Perturbation method.
        question_id (str): Identifier for the question.
    """
    result_path = Path("results/sample_res")
    result_path.mkdir(parents=True, exist_ok=True)
    res_json = {"gea": result, "show": show_anls, "hide": hide_anls}
    with open(result_path / f"gea_{method}_id_{question_id}.json", "w") as f:
        json.dump(res_json, f, indent=4)

def plot_auc_curve(show_anls, hide_anls, question_id):
    """
    Plot and save the AUC curve for ANLS scores.
    
    Args:
        show_anls (list): ANLS scores for visible regions.
        hide_anls (list): ANLS scores for hidden regions.
        question_id (str): Identifier for the question.
    """
    plot_path = Path("results/plots")
    plot_path.mkdir(parents=True, exist_ok=True)
    plt.plot(show_anls, label="Show")
    plt.plot(hide_anls, label="Hide")
    plt.legend()
    plt.savefig(plot_path / f"gea_auc_{question_id}.png")
    plt.close()

def main():
    args = create_args()
    random.seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, processor = load_model_and_processor(args.model_name, device)
    data_path = Path(args.data_path)
    test_set = json.load(open(data_path / args.set_name, "r"))
    total_accuracy = 0
    if COUNTED == -1:
        sample_count = len(test_set["data"])
    else:
        sample_count = min(len(test_set["data"]), COUNTED)  # For demonstration, limiting to n samples

    for sample in tqdm(test_set["data"][:sample_count]):
        img_path = data_path / "imgs" / sample["image"]
        question_id = str(sample["questionId"])
        
        # Load image and scale boxes to image dimensions
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Image at {img_path} could not be loaded.")
            continue
        
        height, width, _ = image.shape
        boxes = np.array(sample["evidence"]["positions"]) * np.array([width, width, height, height])

        # Calculate GEA accuracy
        accuracy, show_anls, hide_anls = calculate_gea_accuracy(
            args.model_name, model, processor, image, boxes, sample["question"], 
            sample["answers"], question_id, args.batch_size, GRID_SIZE, method=args.method
        )
        total_accuracy += accuracy
        
        # Save results
        save_results(accuracy, show_anls, hide_anls, args.method, question_id)
        
        # Plot AUC curve if specified
        if args.plot_auc:
            plot_auc_curve(show_anls, hide_anls, question_id)

    average_accuracy = total_accuracy / sample_count
    print(f"Average GEA: {average_accuracy}")

if __name__ == "__main__":
    main()
