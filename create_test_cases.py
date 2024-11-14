import numpy as np
import cv2

def random_box_cases(boxes):
    """
    return different boxes for the cases, gt, extra random, extra offset, extra uniform
    extra full page, less"""

    possible_cases = ["gt", "extra_random", "extra_offset", "extra_uniform", "extra_full_page", "less"]
    gt, extra_random, extra_offset, extra_uniform, extra_full_page, less = [], [], [], [], [], []
    for box in boxes:
        gt.append(box)

        # bigger random boxes
        rand_box = np.random.rand(4) * max(box[1]-box[0], box[3]-box[2])
        rand_box[0] *= -1
        rand_box[2] *= -1
        extra_random.append(box + rand_box )
        
        # bigger random boxes offset (keep the gt in top left)
        rand_xy = np.random.rand(2) * (1 - max(box[1]-box[0], box[3]-box[2]))
        extra_offset.append(box + np.array([0.0, rand_xy[0], 0.0, rand_xy[1]]))
        
        # bigger keeping the gt in center 
        rand_xy = np.random.rand(1) * (max(box[1]-box[0], box[3]-box[2]))
        extra_uniform.append(box + np.array([-1*rand_xy[0], rand_xy[0], -1*rand_xy[0], rand_xy[0]]))
        
        # full page
        extra_full_page.append([0., 1.0, 0., 1.0])

        # smaller boxes
        rand_box_x = np.random.rand(2) * (box[1]-box[0])
        rand_box_y = np.random.rand(2) * (box[3]-box[2])
        rand_box_x[1] *= -1
        rand_box_y[1] *= -1
        rand_box = np.concatenate([rand_box_x, rand_box_y])
        less.append(box + rand_box)

    for i in range(len(gt)):
        gt[i] = np.clip(gt[i], 0, 1)
        extra_random[i] = np.clip(extra_random[i], 0, 1)
        extra_offset[i] = np.clip(extra_offset[i], 0, 1)
        extra_uniform[i] = np.clip(extra_uniform[i], 0, 1)
        extra_full_page[i] = np.clip(extra_full_page[i], 0, 1)
        less[i] = np.clip(less[i], 0, 1)
    # make sure the boxes of [x1, x2, y1, y2] are respecting x1 < x2 and y1 < y2
    for i in range(len(gt)):
        if gt[i][0] > gt[i][1]:
            gt[i][0], gt[i][1] = gt[i][1], gt[i][0]
        if gt[i][2] > gt[i][3]:
            gt[i][2], gt[i][3] = gt[i][3], gt[i][2]
        if extra_random[i][0] > extra_random[i][1]:
            extra_random[i][0], extra_random[i][1] = extra_random[i][1], extra_random[i][0]
        if extra_random[i][2] > extra_random[i][3]:
            extra_random[i][2], extra_random[i][3] = extra_random[i][3], extra_random[i][2]
        if extra_offset[i][0] > extra_offset[i][1]:
            extra_offset[i][0], extra_offset[i][1] = extra_offset[i][1], extra_offset[i][0]
        if extra_offset[i][2] > extra_offset[i][3]:
            extra_offset[i][2], extra_offset[i][3] = extra_offset[i][3], extra_offset[i][2]
        if extra_uniform[i][0] > extra_uniform[i][1]:
            extra_uniform[i][0], extra_uniform[i][1] = extra_uniform[i][1], extra_uniform[i][0]
        if extra_uniform[i][2] > extra_uniform[i][3]:
            extra_uniform[i][2], extra_uniform[i][3] = extra_uniform[i][3], extra_uniform[i][2]
        if extra_full_page[i][0] > extra_full_page[i][1]:
            extra_full_page[i][0], extra_full_page[i][1] = extra_full_page[i][1], extra_full_page[i][0]
        if extra_full_page[i][2] > extra_full_page[i][3]:
            extra_full_page[i][2], extra_full_page[i][3] = extra_full_page[i][3], extra_full_page[i][2]
        if less[i][0] > less[i][1]:
            less[i][1] = less[i][0] + 0.02
        if less[i][2] > less[i][3]:
            less[i][3] = less[i][2] + 0.02
    return possible_cases, (gt, extra_random, extra_offset, extra_uniform, extra_full_page, less)



# do a test
if __name__ == "__main__":
    img_path = "./data/imgs/kqbf0227_p0.jpg"

    image = cv2.imread(img_path)
    height, width = image.shape[:2]

    norm_boxes = np.array([[
                        0.1363,
                        0.3991,
                        0.4252,
                        0.4496
                    ],
                    [
                        0.1839,
                        0.2839,
                        0.503,
                        0.5209
                    ],
                    [
                        0.3975,
                        0.4597,
                        0.5043,
                        0.5231
                    ]])
    

    possible_cases, box_cases = random_box_cases(norm_boxes)

    print(possible_cases)

    for i, norm_boxes in enumerate(box_cases):
        boxes = norm_boxes * np.array([width, width, height, height])
        image = cv2.imread(img_path)
        # draw the boxes
        for box in boxes:
            x1, x2, y1, y2 = box
            rand_color = np.random.randint(0, 255, 3)
            image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), [int(r) for r in rand_color], 2)
        # save the image
        cv2.imwrite(f"./results/imgs/test_{possible_cases[i]}.jpg", image)
