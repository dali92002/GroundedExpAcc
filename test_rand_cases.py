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
from create_test_cases import random_box_cases
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
        Tuple[model, processor]: Loaded model and processor, or None if model not supported.
    """
    if model_name == "pix2struct":
        model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-docvqa-base").to(device)
        processor = Pix2StructProcessor.from_pretrained("google/pix2struct-docvqa-base")
        return model, processor
    elif model_name == "gpt":
        return None, None
    else:
        raise ValueError("Unsupported model. Choose either 'pix2struct' or 'gpt'.")

def save_case_results(result, show_anls, hide_anls, method, question_id, case):
    """
    Save GEA results for a specific case to a JSON file.
    
    Args:
        result (float): GEA accuracy score.
        show_anls (list): ANLS scores for visible regions.
        hide_anls (list): ANLS scores for hidden regions.
        method (str): Perturbation method.
        question_id (str): Identifier for the question.
        case (str): Type of case (e.g., 'gt', 'extra_random', etc.).
    """
    result_path = Path("results/sample_res")
    result_path.mkdir(parents=True, exist_ok=True)
    res_json = {"gea": result, "show": show_anls, "hide": hide_anls}
    with open(result_path / f"gea_{method}_id_{question_id}_case_{case}.json", "w") as f:
        json.dump(res_json, f, indent=4)

def plot_auc_curve(show_anls, hide_anls, question_id, case):
    """
    Plot and save the AUC curve for ANLS scores for a specific case.
    
    Args:
        show_anls (list): ANLS scores for visible regions.
        hide_anls (list): ANLS scores for hidden regions.
        question_id (str): Identifier for the question.
        case (str): Type of case (e.g., 'gt', 'extra_random', etc.).
    """
    plot_path = Path("results/plots")
    plot_path.mkdir(parents=True, exist_ok=True)
    plt.plot(show_anls, label="Show")
    plt.plot(hide_anls, label="Hide")
    plt.legend()
    plt.savefig(plot_path / f"gea_auc_{question_id}_case_{case}.png")
    plt.close()

def calculate_and_store_results(data_path, test_set, model_name, model, processor, device, method, batch_size, grid_size, plot_auc):
    """
    Calculate GEA accuracy across all samples and cases in the test set, then save and plot results.

    Args:
        test_set (dict): Loaded test set JSON.
        model_name (str): Name of the model being used.
        model (Model): Model object or None if model is not used.
        processor (Processor): Processor object or None if model is not used.
        device (str): Device to run calculations on.
        method (str): Perturbation method.
        batch_size (int): Batch size for processing.
        grid_size (int): Size of the grid for perturbation.
        plot_auc (bool): Flag to plot AUC curves.
    """
    all_result = {"gt": 0, "extra_random": 0, "extra_offset": 0, "extra_uniform": 0, "extra_full_page": 0, "less": 0}
    sample_count = min(len(test_set["data"]), COUNTED)  # For demonstration, limiting to n samples

    for sample in tqdm(test_set["data"][:sample_count]):
        img_path = data_path / "imgs" / sample["image"]
        question_id = str(sample["questionId"])
        image = cv2.imread(str(img_path))
        
        if image is None:
            print(f"Image at {img_path} could not be loaded.")
            continue

        height, width, _ = image.shape
        question = sample["question"]
        answers = sample["answers"]

        possible_cases, box_cases = random_box_cases(sample["evidence"]["positions"])

        for case, boxes in zip(possible_cases, box_cases):
            # Scale boxes to image dimensions
            scaled_boxes = np.array([[box[0] * width, box[1] * width, box[2] * height, box[3] * height] for box in boxes])

            # Calculate GEA accuracy
            result, show_anls, hide_anls = calculate_gea_accuracy(
                model_name, model, processor, image, scaled_boxes, question, answers, 
                question_id, batch_size, grid_size, method=method
            )
            all_result[case] += result
            
            # Save case results
            save_case_results(result, show_anls, hide_anls, method, question_id, case)

            # Plot AUC curve if specified
            if plot_auc:
                plot_auc_curve(show_anls, hide_anls, question_id, case)
    
    # Average results
    all_result = {k: v / sample_count for k, v in all_result.items()}
    
    # Save overall results
    results_path = Path("results")
    results_path.mkdir(parents=True, exist_ok=True)
    with open(results_path / f"gea_{method}_model_{model_name}_grid_{grid_size}.json", "w") as f:
        json.dump(all_result, f, indent=4)
    
    print("Final Results:", all_result)

def main():
    args = create_args()
    random.seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = Path(args.data_path)

    model, processor = load_model_and_processor(args.model_name, device)
    test_set = json.load(open(Path(args.data_path) / args.set_name, "r"))

    calculate_and_store_results(
        data_path, test_set, args.model_name, model, processor, device, 
        args.method, args.batch_size, GRID_SIZE, args.plot_auc
    )

if __name__ == "__main__":
    main()