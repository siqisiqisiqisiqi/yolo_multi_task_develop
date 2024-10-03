from ultralytics.utils.plotting import Annotator, colors
from ultralytics.data.utils import polygon2mask
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
            contents = file.readlines()
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


def polygons_conversion(polygons, dim):
    X = dim[0]
    Y = dim[1]
    polygons_new = np.zeros_like((polygons))
    polygons_new[:, 0] = X * polygons[:, 0]
    polygons_new[:, 1] = Y * polygons[:, 1]
    return [polygons_new.reshape((1,-1))]


def mask_visualization(mask, color, img):
    mask = (mask * 255).astype(np.uint8)
    colored_mask = np.zeros_like(img)
    colored_mask[:, :] = color  # Green mask
    colored_mask = cv2.bitwise_and(colored_mask, colored_mask, mask=mask * 255)
    # mask_all +=colored_mask
    img = cv2.addWeighted(img, 1.0, colored_mask, 0.3, 0)
    return img

def main():

    image_name = "Image_0.jpg"
    label_name = "Image_0.txt"

    image_dir = "./datasets/green_onion/images"
    label_dir = "./datasets/green_onion/labels"

    img = cv2.imread(os.path.join(ROOT_DIR, image_dir, image_name))
    (Y, X, _) = img.shape

    file_path = os.path.join(ROOT_DIR, label_dir, label_name)
    contents = read_txt_file(file_path)

    annotator = Annotator(img, line_width=2)

    for i, content in enumerate(contents):
        color = colors.palette[i % len(colors.palette)]

        content_list = content.replace('\n', ' ').split(" ")
        content_list = [float(value) for value in content_list[:-1]]
        category = int(content_list[0])

        # # object detection visualization
        box = content_list[1:5]
        xyxy = box2xyxy(box, (X, Y))
        annotator.box_label(xyxy, f"green onion{i}", color=color)

        # keypoints visualization
        keypoints = content_list[5:5 + 3 * 7]
        keypoints = kpts_conversion(keypoints, (X, Y))
        annotator.kpts(keypoints, img.shape)

        # segmentation visualization
        polygons = np.array(content_list[5 + 3 * 7:]).reshape((-1, 2))
        polygons = polygons_conversion(polygons, (X, Y))
        mask = polygon2mask((Y, X), polygons)
        annotator.im = mask_visualization(mask, color, annotator.im)

        cv2.imshow("Image Window", annotator.im)
        cv2.waitKey(0)

        # visiualize one instance
        # break


    cv2.destroyAllWindows()


# def polygon2mask(imgsz, polygons, color=1, downsample_ratio=1):
#     """
#     Convert a list of polygons to a binary mask of the specified image size.

#     Args:
#         imgsz (tuple): The size of the image as (height, width).
#         polygons (list[np.ndarray]): A list of polygons. Each polygon is an array with shape [N, M], where
#                                      N is the number of polygons, and M is the number of points such that M % 2 = 0.
#         color (int, optional): The color value to fill in the polygons on the mask. Defaults to 1.
#         downsample_ratio (int, optional): Factor by which to downsample the mask. Defaults to 1.

#     Returns:
#         (np.ndarray): A binary mask of the specified image size with the polygons filled in.
#     """
#     pass
if __name__ == "__main__":
    main()
