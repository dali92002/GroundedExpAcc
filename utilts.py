from PIL import Image
from anls import anls_score
from docvqa_metrics import Evaluator
import cv2
from create_perturbation import create_show_hide, create_show_hide_box_perturb
from simple_gpt import predict_gpt

def predict_batch_p2s(model, processor, data_one_case, batch_size):
    """ Predicts the answers for a batch of images and questions using the Pix2Struct
        model.
    """


    result = {"ids": [], "show_anls": [], "hide_anls": []}
    evaluator = Evaluator(case_sensitive=False)

    for i in range(0, len(data_one_case["ids"]), batch_size):
        ids = data_one_case["ids"][i:i+batch_size]
        imgs = data_one_case["imgs"][i:i+batch_size]
        questions = data_one_case["questions"][i:i+batch_size]
        answers = data_one_case["answers"][i:i+batch_size]

        imgs_rgb = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in imgs]
        # Process the image and question for model input
        inputs = processor(images=imgs_rgb, text=questions, return_tensors="pt").to(model.device)
        # Generate predictions using the model
        predictions = model.generate(**inputs)
        # Decode the predicted output
        predicted_answers = processor.batch_decode(predictions, skip_special_tokens=True)

        metrics = evaluator.get_metrics(answers, predicted_answers)

        result["ids"].extend(ids)
        result["show_anls"].extend(metrics["anls"][::2])
        result["hide_anls"].extend(metrics["anls"][1::2])

    return result

def predict_batch_gpt(data_one_case):
    """ Predicts the answers for a batch of images and questions using the GPT model.
    """
    
    imgs = data_one_case["imgs"]
    questions = data_one_case["questions"]
    answers = data_one_case["answers"]
    ids = data_one_case["ids"]

    result = {"ids": [], "show_anls": [], "hide_anls": []}

    for id, img, q, a in zip(ids, imgs, questions, answers):
        # Predict the answer using the GPT model
        predicted_answer = predict_gpt(img, q)
        # Calculate the ANLS score
        anls_score_value = anls_score(prediction=predicted_answer, gold_labels=a, threshold=0.5)
        result["ids"].append(id)
        if "show" in id:
            result["show_anls"].append(anls_score_value)
        else:
            result["hide_anls"].append(anls_score_value)

    return result

def calculate_gea_accuracy(model_name, model, processor, image, boxes, question, answers, id, batch_size, grid_size, method="shrink"):
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
    assert method in ["shrink", "perturb"], "Method must be either 'shrink' or 'perturb'"

    sum_diff = 0
    iterations = 10

    data_one_case = {"ids": [], "imgs": [], "questions": [], "answers": []}

    data_one_case["questions"] = [question] * (iterations + 1) * 2
    data_one_case["answers"] = [answers] * (iterations + 1) * 2


    # Iterate through 1 to 10 perturbations (according to shrinkings) and calculate ANLS scores
    for i in range(0, iterations + 1): # we opt to shrink 10 times, but you can change this number
        factor = i / 10
        if method == "shrink":
            img_show, img_hide = create_show_hide(image, boxes.copy(), shrink_factor=factor)
        elif method == "perturb":
            img_show, img_hide = create_show_hide_box_perturb(image, boxes.copy(), rand_factor=factor, grid_size=grid_size)
        
        # # to save images
        # cv2.imwrite(f"examples/data_hide_show_batch/{id}_show_{i}.jpg", img_show)
        # cv2.imwrite(f"examples/data_hide_show_batch/{id}_hide_{i}.jpg", img_hide)

        data_one_case["imgs"].append(img_show)
        data_one_case["imgs"].append(img_hide)
        data_one_case["ids"].append(id + f"_show_{i}")
        data_one_case["ids"].append(id + f"_hide_{i}")
        
    # Get ANLS scores for the perturbed images
    if model_name == "pix2struct":
        result = predict_batch_p2s(model, processor, data_one_case, batch_size)
    elif model_name == "gpt":
        result = predict_batch_gpt(data_one_case) 

    base_diff = result["show_anls"][0] - result["hide_anls"][0]

    for i in range(1, iterations + 1):
        show_anls = result["show_anls"][i]
        hide_anls = result["hide_anls"][i]
        sum_diff += hide_anls - show_anls

    # Calculate the average difference in ANLS scores
    average_diff = (base_diff + sum_diff / iterations) / 2

    return average_diff, result["show_anls"], result["hide_anls"]