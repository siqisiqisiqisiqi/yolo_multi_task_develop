from tqdm import tqdm
import numpy as np
from ultralytics.utils.ops import segments2boxes
import argparse
import os
import glob
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
PARENT_DIR = os.path.dirname(ROOT_DIR)


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
    best_match = None
    min_bbox_diff = float("inf")

    pose_bbox = np.array(pose_line.split()[1:5], dtype=np.float32)

    for i, seg_line in enumerate(seg_lines):
        seg_parts = [x.split() for x in seg_line.strip().splitlines() if len(x)]
        segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2)
                    for x in seg_parts]  # (cls, xy1...)
        seg_bbox = segments2boxes(segments)[0]

        bbox_diff = sum(abs(seg_bbox[i] - pose_bbox[i]) for i in range(4))
        if bbox_diff < min_bbox_diff:
            min_bbox_diff = bbox_diff
            best_match = seg_lines[i]

    return best_match


def merge_annotations(box_path, seg_path, pose_path, output_base_path):
    for subdir, _, _ in os.walk(seg_path):
        relative_path = os.path.relpath(subdir, seg_path)
        box_subdir = os.path.join(box_path, relative_path)
        pose_subdir = os.path.join(pose_path, relative_path)
        output_subdir = os.path.join(output_base_path, relative_path)

        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        seg_files = glob.glob(os.path.join(subdir, "Image_*.txt"))

        if not seg_files:
            continue

        for seg_file in tqdm(seg_files, desc=f"Processing {subdir} labels", unit="file"):
            pose_file = os.path.join(pose_subdir, os.path.basename(seg_file))
            box_file = os.path.join(box_subdir, os.path.basename(seg_file))
            output_file = os.path.join(
                output_subdir, os.path.basename(seg_file))

            if os.path.exists(pose_file) and os.path.exists(box_file):
                with open(seg_file, "r") as seg, open(box_file, "r") as box, open(pose_file, "r") as pose, open(output_file, "w") as out:
                    box_lines = box.readlines()
                    seg_lines = seg.readlines()
                    pose_lines = pose.readlines()

                    # for seg_line in seg_lines:
                    #     seg_class_index = seg_line.strip().split()[0]
                    #     if seg_class_index == "0":  # Process only if class index is 0
                    #         best_match = match_pose_to_segment(
                    #             seg_line, pose_lines)
                    #         if best_match:
                    #             pose_parts = best_match.strip().split()
                    #             seg_parts = seg_line.strip().split()
                    #             merged_line = (
                    #                 pose_parts[0] +
                    #                 " " +
                    #                 " ".join(pose_parts[1:]) +
                    #                 " " +
                    #                 " ".join(seg_parts[1:]) +
                    #                 "\n"
                    #             )
                    #             # out.write(merged_line)
                    #     else:
                    #         # Write segmentation line without pose points
                    #         continue
                    #         # out.write(seg_line)

                    for pose_line in pose_lines:
                        pose_class_index = pose_line.strip().split()[0]
                        if pose_class_index == "0":  # Process only if class index is 0
                            best_match = match_segment_to_pose(
                                pose_line, seg_lines
                            )
                            if best_match:
                                seg_parts = best_match.strip().split()
                                pose_parts = pose_line.strip().split()
                                merged_line = (
                                    pose_parts[0] +
                                    " " +
                                    " ".join(pose_parts[1:]) +
                                    " " +
                                    " ".join(seg_parts[1:]) +
                                    "\n"
                                )
                                out.write(merged_line)
                        else:
                            # Write segmentation line without pose points
                            continue
                            # out.write(seg_line)


def main(box_dataset, keypoint_dataset, segmentation_dataset, output_dataset):
    merge_annotations(box_dataset, segmentation_dataset, keypoint_dataset, output_dataset)


if __name__ == "__main__":
    datasets_dir = os.path.join(ROOT_DIR, "datasets", "green_onion")
    parser = argparse.ArgumentParser(
        description="Merge detection, keypoint and segmentation datasets.")

    parser.add_argument("-box", "--detection_dataset", default=os.path.join(datasets_dir, "box"),
                        required=False, help="Path to the detection dataset")
    parser.add_argument("-kpt", "--keypoint_dataset", default=os.path.join(datasets_dir, "kpt"),
                        required=False, help="Path to the keypoint dataset")
    parser.add_argument("-seg", "--segmentation_dataset", default=os.path.join(datasets_dir, "seg"),
                        required=False, help="Path to the segmentation dataset")
    parser.add_argument("-o", "--output_dataset", default=os.path.join(datasets_dir, "output"),
                        required=False, help="Path to the output dataset")

    args = parser.parse_args()

    main(args.detection_dataset, args.keypoint_dataset, args.segmentation_dataset, args.output_dataset)

    # print("Merging datasets completed.")
