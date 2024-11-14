import argparse

# create args
def create_args():
    parser = argparse.ArgumentParser(description='Calculate Grounded Explanation Accuracy (GEA)')
    parser.add_argument('--model_name', type=str, default="pix2struct", help='Name of the model to use (gpt or pix2struct)') # will be used with including more models
    parser.add_argument('--set_path', type=str, default="ann_pilot.json", help='Path to the JSON dataset to test')
    parser.add_argument('--data_path', type=str, default="./data/", help='Path to the data folder')
    parser.add_argument('--plot_auc', type=bool, default=False, help='Plot the AUC curve')
    parser.add_argument('--method', type=str, default="perturb", help='Method to use for perturbation (shrink or perturb)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing images')
    parser.add_argument('--save_imgs', type=bool, default=False, help='Save the images for each perturbation')
    args = parser.parse_args()
    return args