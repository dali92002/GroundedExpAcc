import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

def apply_mask(image, mask, fill_value):
    """
    Applies a mask to an image. Pixels where the mask equals 1 are replaced with the original image's pixel values, 
    and pixels where the mask equals 0 are replaced with the fill_value.

    Args:
        image (np.array): The original image.
        mask (np.array): The binary mask where 1 indicates the regions to keep.
        fill_value (np.array): The fill value to apply to regions where the mask is 0.

    Returns:
        np.array: The resulting masked image.
    """
    return image * mask + fill_value * (1 - mask)
    

def create_random_evidences(image, boxes, max_num_perturbations=5):
    """
    Adds random perturbation boxes as evidence regions to the input image. 
    These randomly generated boxes are appended to the original boxes list.

    Args:
        image (np.array): The original image.
        boxes (np.array): Ground truth boxes of shape (N, 4) with each box represented as (x_min, x_max, y_min, y_max).
        max_num_perturbations (int): The maximum number of random boxes to add. Default is 5.

    Returns:
        np.array: Updated array of boxes including the random perturbation boxes.
    """
    for _ in range(random.randint(1, max_num_perturbations)):
        x_min = random.randint(0, image.shape[1] - 100)
        x_max = random.randint(x_min + 1, image.shape[1])
        y_min = random.randint(0, image.shape[0] - 100)
        y_max = random.randint(y_min + 1, image.shape[0])

        boxes = np.vstack([boxes, [x_min, x_max, y_min, y_max]])
    return boxes

def create_img_show(image, boxes):
    """
    Creates a version of the image where the regions inside the boxes are highlighted (preserved) 
    and additional random perturbations (evidences) are added. The rest of the image is filled with the mean color.

    Args:
        image (np.array): The original image.
        boxes (np.array): Ground truth boxes to preserve, of shape (N, 4) where each box is (x_min, x_max, y_min, y_max).

    Returns:
        np.array: The resulting image with showing only the pixels inside the boxes (gt and randmely added) and
                 hiding the rest.
    """
    # Calculate the mean color of the original image
    mean_color = np.mean(image, axis=(0, 1), keepdims=True).astype(np.uint8)

    # Create a mask for the boxes
    mask = np.zeros_like(image, dtype=np.uint8)

    # Add more boxes
    boxes = create_random_evidences(image, boxes)
    for box in boxes:
        x_min, x_max, y_min, y_max = map(int, box)
        mask[y_min:y_max, x_min:x_max] = 1

    # Apply the mask
    return apply_mask(image, mask, mean_color)

def create_img_hide(image, boxes):
    """
    Creates an image where the regions inside the boxes are hidden by filling them with the mean color.

    Args:
        image (np.array): The original image.
        boxes (np.array): Array of bounding boxes, where each box is (x_min, x_max, y_min, y_max).

    Returns:
        np.array: The image with boxes hidden.
    """
    # Calculate the mean color of the original image
    mean_color = np.mean(image, axis=(0, 1), keepdims=True).astype(np.uint8)

    # Create a mask for the boxes
    mask = np.ones_like(image, dtype=np.uint8)
    for box in boxes:
        x_min, x_max, y_min, y_max = map(int, box)
        mask[y_min:y_max, x_min:x_max] = 0

    # Apply the mask
    return apply_mask(image, mask, mean_color)

def create_show_hide(image, boxes, shrink_factor=0.0):
    """
    Creates two versions of an image: one showing the regions inside the boxes, and the other hiding those regions.
    The boxes are shrunk symmetrically from both the bottom-left and upper-right based on the shrink factor.

    Args:
        image (np.array): The original image.
        boxes (np.array): Array of bounding boxes, where each box is (x_min, x_max, y_min, y_max).
        shrink_factor (float): Factor to shrink the boxes symmetrically. Shrinking is applied equally to all sides.

    Returns:
        tuple: (show_image, hide_image) where:
            - show_image is the image with the boxes highlighted and perturbations added.
            - hide_image is the image with the boxes hidden.
    """
    # Calculate the shrink amounts
    x_shrink = (boxes[:, 1] - boxes[:, 0]) * shrink_factor / 2  # Shrink by half from both sides along x-axis
    y_shrink = (boxes[:, 3] - boxes[:, 2]) * shrink_factor / 2  # Shrink by half from both sides along y-axis

    # Adjust the boxes symmetrically
    boxes[:, 0] += x_shrink  # Increase x_min (move right)
    boxes[:, 1] -= x_shrink  # Decrease x_max (move left)
    boxes[:, 2] += y_shrink  # Increase y_min (move down)
    boxes[:, 3] -= y_shrink  # Decrease y_max (move up)

    # Generate the "show" and "hide" images
    img_show = create_img_show(image, boxes)
    img_hide = create_img_hide(image, boxes)

    return img_show, img_hide
def random_box_perturb(box, rand_factor=0.0, grid_size=10):
    """
    Randomly perturbs a box by dividing it into a grid and setting random grid cells to zero.

    Args:
        box (np.array): The bounding box to perturb.
        rand_factor (float): The factor of random perturbations to apply.
        grid_size (int): The size of the grid to divide the box into.
    Returns:
        np.array: The mask with random perturbations applied.
    """

    # divide box into 100 by 100 grid
    grid_x = np.linspace(box[0], box[1], grid_size) - box[0]
    grid_y = np.linspace(box[2], box[3], grid_size) - box[2]
    
    grid_x = grid_x.astype(int)
    grid_y = grid_y.astype(int)

    mask = np.ones((box[3] - box[2], box[1] - box[0], 3))

    # randomely perturb rand factor of the grid to be zeros
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            if random.randint(0,9) < rand_factor*10:
                mask[grid_y[i]:grid_y[i+1], grid_x[j]:grid_x[j+1]] = 0
    return mask


def create_show_hide_box_perturb(image, boxes, rand_factor=0.0):
    """
    Creates two versions of an image: one showing the regions inside the boxes, and the other hiding those regions.
    The boxes are perturbed by setting random grid cells to zero based on the random factor.
    
    
    Args:
        image (np.array): The original image.
        boxes (np.array): Array of bounding boxes, where each box is (x_min, x_max, y_min, y_max).
        rand_factor (float): The factor of random perturbations to apply.
    Returns:
        tuple: (show_image, hide_image) where:
            - show_image is the image with the boxes highlighted and perturbations added.
            - hide_image is the image with the boxes hidden.
    """
    # Calculate the mean color of the original image
    mean_color = np.mean(image, axis=(0, 1), keepdims=True).astype(np.uint8)

    # Create a mask for the boxes
    mask_show = np.zeros_like(image, dtype=np.uint8)
    mask_hide = np.ones_like(image, dtype=np.uint8)

    # # Add more boxes
    # boxes = create_random_evidences(image, boxes)

    for box in boxes:
        box = box.astype(int)
        show_box_mask = random_box_perturb(box, rand_factor)
        hide_box_mask = 1 - show_box_mask

        mask_show[box[2]:box[3], box[0]:box[1]] = show_box_mask
        mask_hide[box[2]:box[3], box[0]:box[1]] = hide_box_mask

    # apply mask hide and show
    image_show = apply_mask(image, mask_show, mean_color)
    image_hide = apply_mask(image, mask_hide, mean_color)

    return image_show, image_hide
    


if __name__ == "__main__":

    test = "perturb"
    factor = 0.8

    # Load an example image and bounding boxes
    img_path = "examples/fryn0081_p8.jpg"
    boxes = np.array([[143., 882., 330., 415.],
                    [145., 734., 396., 476.]])
    image = cv2.imread(img_path)

    if test == "shrinking":
        img_show, img_hide = create_show_hide(image, boxes, factor)

    elif test=="perturb":
        img_show, img_hide = create_show_hide_box_perturb(image, boxes, factor)

    # Display the original, show, and hide images
    plt.figure(figsize=(10, 10))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
    plt.title("Show Image")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(img_hide, cv2.COLOR_BGR2RGB))
    plt.title("Hide Image")
    plt.axis("off")

    plt.savefig(f"examples/new_show_hide_{test}_{factor}.png")