from ultralytics import YOLO
import cv2
import numpy as np


def size_interpolation(img, image_shape):
    """resize the image to the desired shape

    Parameters
    ----------
    img : ndarray
        original image needs to be shaped
    image_shape : tuple
        goal image shape
    Returns
    -------
    ndarray
        resized image
    """
    instack = img.transpose((1, 2, 0))

    mask = cv2.resize(instack, (image_shape[1], image_shape[0]),
                      interpolation=cv2.INTER_NEAREST)

    if instack.shape[-1] == 1:
        mask = np.expand_dims(mask, axis=-1)

    mask = mask.transpose((2, 0, 1))
    return mask


model = YOLO("./weights/best.pt")

# results = model("./green_onion.jpg",)
# results = model("./Image_19.jpg",imgsz=1080)
results = model("./green_onion.jpg", conf=0.55, imgsz=1088)

# for r in results:
#     im_array = r.plot()
#     cv2.imshow("image", im_array)
#     cv2.waitKey()

# self visualization
colors = [
    (0, 255, 0),     # Green
    (255, 0, 0),     # Blue
    (255, 0, 0),
    (0, 0, 255),     # Red
    (0, 0, 255),
    (255, 255, 0),   # Cyan
    (255, 255, 0),
    (255, 0, 255),   # Magenta
    (255, 0, 255),
    (0, 255, 255),   # Yellow
    (128, 0, 128),   # Purple
    (255, 165, 0),   # Orange
    (0, 128, 128),   # Teal
    (128, 128, 0),   # Olive
    (75, 0, 130),    # Indigo
    (238, 130, 238),  # Violet
    (255, 192, 203),  # Pink
    (139, 69, 19),   # Brown
    (192, 192, 192),  # Silver
]

img = cv2.imread("./green_onion.jpg")

# segmentation visualization
mask_result = results[0].masks.data.cpu().detach().numpy()
mask_result = size_interpolation(mask_result, img.shape)
mask_all = np.zeros_like(img)
for i, mask in enumerate(mask_result):
    #     cv2.imshow("test", mask * 255)
    #     cv2.waitKey(0)
    mask = (mask * 255).astype(np.uint8)
    colored_mask = np.zeros_like(img)
    colored_mask[:, :] = colors[i % len(colors)]  # Green mask
    colored_mask = cv2.bitwise_and(colored_mask, colored_mask, mask=mask * 255)
    mask_all += colored_mask

img = cv2.addWeighted(img, 0.9, mask_all, 0.3, 0)


# # key point visualization
keypoints = results[0].keypoints.data.detach().cpu().numpy().astype(np.int32)
num_object = keypoints.shape[0]
for keypoint in keypoints:
    keypoint = keypoint[~np.all(keypoint == 0, axis=1)]
    num_point = keypoint.shape[0]
    for i in range(num_point):
        point = keypoint[i][:2]

        cv2.circle(img, point, 5, colors[i], -1)
        # cv2.putText(img, str(i + 1), (point[0] + 10, point[1]),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i % len(colors)], 2)
        if i < 6:
            if i + 1 == 5:
                cv2.line(img, keypoint[2][:2], keypoint[i + 1][:2], colors[5], thickness=2)
            else:
                cv2.line(img, keypoint[i][:2], keypoint[i + 1][:2], colors[5], thickness=2)


cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
