from ultralytics.utils.plotting import Annotator, colors
import numpy as np
import cv2
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)


def read_txt_file(file_path):
    try:
        with open(file_path, 'r') as file:
            contents = file.read()
        return contents
    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def box2xyxy(box, dim):
    X = dim[0]
    Y = dim[1]
    width = box[2]
    height = box[3]
    xyxy = np.array([box[0] - width / 2, box[1] - height / 2,
                    box[0] + width / 2, box[1] + height / 2])
    scale = np.array([X, Y, X, Y])
    a = xyxy * scale
    return a.astype(int)


def kpts_conversion(kpts, dim):
    X = dim[0]
    Y = dim[1]
    kpts_array = np.array(kpts).reshape((-1, 3))
    kpts_new = np.zeros_like((kpts_array))
    kpts_new[:, 0] = X * kpts_array[:, 0]
    kpts_new[:, 1] = Y * kpts_array[:, 1]
    kpts_new[:, 2] = kpts_array[:, 2]
    return kpts_new


# image_name = "000000000785.jpg"
# label_name = "000000000785.txt"

# image_dir = "./datasets/coco-multi-person/images/val2017"
# label_dir = "./datasets/coco-multi-person/labels/val2017"

image_name = "Image_0.jpg"
label_name = "Image_0.txt"

image_dir = "./datasets/green_onion/images"
label_dir = "./datasets/green_onion/labels"

img = cv2.imread(os.path.join(ROOT_DIR, image_dir, image_name))
(Y, X, _) = img.shape

file_path = os.path.join(ROOT_DIR, label_dir, label_name)
contents = read_txt_file(file_path)
content_list = contents.split(" ")
content_list = [float(value) for value in content_list]
category = int(content_list[0])

annotator = Annotator(img, line_width=2)

# # object detection visualization
# box = content_list[1:5]
# xyxy = box2xyxy(box, (X, Y))
# annotator.box_label(xyxy, "human")

# cv2.imshow("Image Window", annotator.im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# keypoints visualization
keypoints = content_list[5:5 + 17 * 3]
keypoints = kpts_conversion(keypoints, (X, Y))

annotator.kpts(keypoints, img.shape)
cv2.imshow("Image Window", annotator.im)
cv2.waitKey(0)
cv2.destroyAllWindows()
