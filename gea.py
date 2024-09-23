from PIL import Image
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from anls import anls_score
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2
from create_perturbation import create_show_hide
import random
import json
import argparse

def predict_pix2struct(model, processor, img, question, answers):
    """
    Predicts the answer to a question based on an image using the Pix2Struct model.

    Args:
        model: The pre-trained Pix2Struct model.
        processor: The processor for handling image and text inputs.
        img (np.array): The input image in BGR format.
        question (str): The question related to the image.
        answers (list): List of ground truth answers for evaluation.

    Returns:
        tuple: Contains the predicted answer and the ANLS score.
    """
    # Convert the image from BGR to RGB format
    image_rgb = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Process the image and question for model input
    inputs = processor(images=image_rgb, text=question, return_tensors="pt").to(device)

    # Generate predictions using the model
    predictions = model.generate(**inputs)

    # Decode the predicted output
    predicted_answer = processor.decode(predictions[0], skip_special_tokens=True)
    
    # Calculate the ANLS score
    anls_score_value = anls_score(prediction=predicted_answer, gold_labels=answers, threshold=0.5)

    return predicted_answer, anls_score_value

def calculate_gea_accuracy(model, processor, image, boxes, question, answers):
    """
    Calculates the Grounded Explanation Accuracy (GEA) based on the model's predictions 
    for both shown and hidden regions in the image.

    Args:
        model: The pre-trained Pix2Struct model.
        processor: The processor for handling image and text inputs.
        img_path (str): Path to the input image.
        boxes (np.array): Array of bounding boxes.
        question (str): The question related to the image.
        answers (list): List of ground truth answers for evaluation.

    Returns:
        tuple: Contains the average difference in ANLS scores, and lists of ANLS scores 
               for show and hide images.
    """
    
    # Initialize lists to hold ANLS scores for shown and hidden images
    show_anls_scores = []
    hide_anls_scores = []

    # Create show and hide images based on the provided boxes
    img_show, img_hide = create_show_hide(image, boxes.copy())

    # Predict ANLS scores for the show and hide images
    _, show_anls = predict_pix2struct(model, processor, img_show, question, answers)
    _, hide_anls = predict_pix2struct(model, processor, img_hide, question, answers)

    # Calculate the base difference in ANLS scores
    base_diff = show_anls - hide_anls
    show_anls_scores.append(show_anls)
    hide_anls_scores.append(hide_anls)

    sum_diff = 0

    # Iterate through 1 to 10 perturbations (according to shrinkings) and calculate ANLS scores
    for i in tqdm(range(1, 11)): # we opt to shrink 10 times, but you can change this number
        shrink_factor = i / 10
        img_show, img_hide = create_show_hide(image, boxes.copy(), shrink_factor)
        
        # Get ANLS scores for the perturbed images
        _, show_anls = predict_pix2struct(model, processor, img_show, question, answers)
        _, hide_anls = predict_pix2struct(model, processor, img_hide, question, answers)
        
        # Accumulate the differences
        sum_diff += hide_anls - show_anls
        show_anls_scores.append(show_anls)
        hide_anls_scores.append(hide_anls)

    # Calculate the average difference in ANLS scores
    average_diff = (base_diff + sum_diff / 10) / 2

    return average_diff, show_anls_scores, hide_anls_scores

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Calculate Grounded Explanation Accuracy (GEA)')
    parser.add_argument('--model_name', type=str, default="pix2struct", help='Name of the model to use') # will be used with including more models
    parser.add_argument('--set_path', type=str, default="examples/example.json", help='Path to the JSON dataset to test')
    args = parser.parse_args()

    
    random.seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load the pre-trained model and processor
    model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-docvqa-base").to(device)
    processor = Pix2StructProcessor.from_pretrained("google/pix2struct-docvqa-base")

    test_set = json.load(open(args.set_path, "r"))
    for sample in  test_set["data"]:

        img_path = sample["image"]
        boxes = sample["evidence"]["positions"]
        question = sample["question"]
        answers = sample["answers"]

        # Denormalize boxes and stack them into a numpy array
        image = cv2.imread(img_path)
        height, width, _ = image.shape
        boxes = np.array([[box[0] * width, box[1] * width, box[2] * height, box[3] * height] for box in boxes])
        
        # Calculate the GEA accuracy
        result, all_show_anls, all_hide_anls = calculate_gea_accuracy(model, processor, image, boxes, question, answers)
        

        # Print the results
        print("GEA: ", result)
        print("Show ANLS Scores: ", all_show_anls)
        print("Hide ANLS Scores: ", all_hide_anls)

        # Plot the ANLS scores for show and hide images
        plt.plot(all_show_anls, label="Show")
        plt.plot(all_hide_anls, label="Hide")
        plt.legend()

        # Save the plot to a file
        plt.savefig(f"examples/gea_plot.png")
        plt.close()
