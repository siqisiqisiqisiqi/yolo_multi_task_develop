from tqdm import tqdm
import numpy as np
from ultralytics.utils.ops import segments2boxes
from ultralytics.utils import ops
import argparse
import os
import glob
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
PARENT_DIR = os.path.dirname(ROOT_DIR)


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0, 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou, interArea


def match_pose_to_segment(seg_line, pose_lines):
    seg_parts = [x.split() for x in seg_line.strip().splitlines() if len(x)]
    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2)
                for x in seg_parts]  # (cls, xy1...)
    seg_bbox = segments2boxes(segments)[0]

    best_match = None
    min_bbox_diff = float("inf")

    lb = [x.split() for x in pose_lines if len(x)]

    for i, pose_bbox in enumerate([np.array(x[1:5], dtype=np.float32) for x in lb]):
        bbox_diff = sum(abs(seg_bbox[i] - pose_bbox[i]) for i in range(4))
        if bbox_diff < min_bbox_diff:
            min_bbox_diff = bbox_diff
            best_match = pose_lines[i]

    return best_match


def match_segment_to_pose(pose_line, seg_lines):
    """match the pose label and seg label

    Parameters
    ----------
    pose_line : String
        pose label
    seg_lines : String
        seg label

    Returns
    -------
    _type_
        best match
    """
    best_match = None
    max_bbox_inter = 0
    bbox = None

    pose_bbox = np.array(pose_line.split()[1:5], dtype=np.float32)
    pose_bbox = ops.xywh2xyxy(pose_bbox)

    for i, seg_line in enumerate(seg_lines):
        seg_parts = [x.split() for x in seg_line.strip().splitlines() if len(x)]
        segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2)
                    for x in seg_parts]  # (cls, xy1...)
        seg_bbox_xywh = segments2boxes(segments)[0]
        seg_bbox = ops.xywh2xyxy(seg_bbox_xywh)

        # bbox_diff = sum(abs(seg_bbox[i] - pose_bbox[i]) for i in range(4))
        iou, inter = bb_intersection_over_union(list(seg_bbox), list(pose_bbox))
        if iou > max_bbox_inter:
            max_bbox_inter = iou
            best_match = seg_lines[i]
            bbox = seg_bbox_xywh

    return best_match, bbox


def merge_annotations(seg_path, pose_path, output_base_path):
    for subdir, _, _ in os.walk(seg_path):
        relative_path = os.path.relpath(subdir, seg_path)
        # box_subdir = os.path.join(box_path, relative_path)
        pose_subdir = os.path.join(pose_path, relative_path)
        output_subdir = os.path.join(output_base_path, relative_path)

        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        seg_files = glob.glob(os.path.join(subdir, "Image_*.txt"))

        if not seg_files:
            continue

        for seg_file in tqdm(seg_files, desc=f"Processing {subdir} labels", unit="file"):
            pose_file = os.path.join(pose_subdir, os.path.basename(seg_file))
            # box_file = os.path.join(box_subdir, os.path.basename(seg_file))
            output_file = os.path.join(
                output_subdir, os.path.basename(seg_file))

            if os.path.exists(pose_file):
                with open(seg_file, "r") as seg, open(pose_file, "r") as pose, open(output_file, "w") as out:
                    seg_lines = seg.readlines()
                    pose_lines = pose.readlines()

                    for pose_line in pose_lines:
                        pose_class_index = pose_line.strip().split()[0]
                        if pose_class_index == "0":  # Process only if class index is 0
                            best_match, bbox = match_segment_to_pose(
                                pose_line, seg_lines
                            )
                            if best_match:
                                seg_parts = best_match.strip().split()
                                pose_parts = pose_line.strip().split()
                                bbox = list(bbox)
                                merged_line = (
                                    pose_parts[0] +
                                    " " +
                                    " ".join(map(str, bbox)) +
                                    " " +
                                    " ".join(pose_parts[5:]) +
                                    " " +
                                    " ".join(seg_parts[1:]) +
                                    "\n"
                                )
                                out.write(merged_line)
                        else:
                            # Write segmentation line without pose points
                            continue



def main(keypoint_dataset, segmentation_dataset, output_dataset):
    merge_annotations(segmentation_dataset, keypoint_dataset, output_dataset)


if __name__ == "__main__":
    datasets_dir = os.path.join(ROOT_DIR, "datasets", "green_onion_improved")
    parser = argparse.ArgumentParser(
        description="Merge detection, keypoint and segmentation datasets.")

    # parser.add_argument("-box", "--detection_dataset", default=os.path.join(datasets_dir, "box"),
    #                     required=False, help="Path to the detection dataset")
    parser.add_argument("-kpt", "--keypoint_dataset", default=os.path.join(datasets_dir, "kpt"),
                        required=False, help="Path to the keypoint dataset")
    parser.add_argument("-seg", "--segmentation_dataset", default=os.path.join(datasets_dir, "seg"),
                        required=False, help="Path to the segmentation dataset")
    parser.add_argument("-o", "--output_dataset", default=os.path.join(datasets_dir, "output"),
                        required=False, help="Path to the output dataset")

    args = parser.parse_args()

    main(args.keypoint_dataset, args.segmentation_dataset, args.output_dataset)

    # print("Merging datasets completed.")
