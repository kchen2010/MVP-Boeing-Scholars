import itertools

import numpy as np

from utils import mask_boundary


def iou(predicted_mask, ground_truth_mask):
    """
    Calculate Intersection over Union (IoU) for a single image segmentation mask.

    Args:
        predicted_mask: Array of (x, y) pixel coordinate tuples for predicted mask
        ground_truth_mask: Array of (x, y) pixel coordinate tuples for ground truth mask

    Returns:
        float: IoU score for the image
    """
    # Convert pixel coordinate lists to sets of tuples for efficient intersection/union
    pred_set = set(tuple(coord) for coord in predicted_mask)
    gt_set = set(tuple(coord) for coord in ground_truth_mask)

    # Calculate intersection and union
    intersection = len(pred_set & gt_set)
    union = len(pred_set | gt_set)

    # Handle edge case where both masks are empty
    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union


def iou_video(predicted_masks, ground_truth_masks):
    """
    Calculate Intersection over Union (IoU) for video segmentation masks.

    Args:
        predicted_masks: 2D array where outer array represents frames and inner arrays
                        contain (x, y) pixel coordinate tuples for predicted mask
        ground_truth_masks: 2D array where outer array represents frames and inner arrays
                           contain (x, y) pixel coordinate tuples for ground truth mask

    Returns:
        float: Average IoU across all frames
    """
    if len(predicted_masks) != len(ground_truth_masks):
        raise ValueError(
            "Predicted and ground truth masks must have the same number of frames"
        )

    if len(predicted_masks) == 0:
        return 0.0

    iou_scores = []

    for pred_frame, gt_frame in zip(predicted_masks, ground_truth_masks):
        frame_iou = iou(pred_frame, gt_frame)
        iou_scores.append(frame_iou)

    # Return average IoU across all frames
    return np.mean(iou_scores)


def dice(predicted_mask, ground_truth_mask):
    """
    Calculate Dice coefficient for a single image segmentation mask.

    Args:
        predicted_mask: Array of (x, y) pixel coordinate tuples for predicted mask
        ground_truth_mask: Array of (x, y) pixel coordinate tuples for ground truth mask

    Returns:
        float: Dice coefficient for the image
    """
    # Convert pixel coordinate lists to sets of tuples for efficient intersection/union
    pred_set = set(tuple(coord) for coord in predicted_mask)
    gt_set = set(tuple(coord) for coord in ground_truth_mask)

    # Calculate intersection and union
    intersection = len(pred_set & gt_set)
    union = len(pred_set | gt_set)

    # Handle edge case where both masks are empty
    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return 2 * intersection / (len(pred_set) + len(gt_set))


def dice_video(predicted_masks, ground_truth_masks):
    """
    Calculate Dice coefficient for a video segmentation.

    Args:
        predicted_masks: 2D array where outer array represents frames and inner arrays
                         contain (x, y) pixel coordinate tuples for predicted mask
        ground_truth_masks: 2D array where outer array represents frames and inner arrays
                            contain (x, y) pixel coordinate tuples for ground truth mask

    Returns:
        float: Average Dice coefficient across all frames
    """
    if len(predicted_masks) != len(ground_truth_masks):
        raise ValueError(
            "Predicted and ground truth masks must have the same number of frames"
        )

    if len(predicted_masks) == 0:
        return 0.0

    dice_scores = []

    for pred_frame, gt_frame in zip(predicted_masks, ground_truth_masks):
        frame_dice = dice(pred_frame, gt_frame)
        dice_scores.append(frame_dice)

    # Return average Dice coefficient across all frames
    return np.mean(dice_scores)


def boundary_f_score(predicted_mask, ground_truth_mask, tolerance=2):
    """
    Calculate boundary F-score for a video segmentation.

    Args:
        predicted_mask: (N, 2) numpy array containing (x, y) pixel coordinate tuples for predicted mask
        ground_truth_mask: (N, 2) numpy array containing (x, y) pixel coordinate tuples for ground truth mask
        tolerance: Maximum taxicab distance for a predicted boundary pixel to be considered a match

    Returns:
        float: Boundary F-score for the two masks
    """
    pred_bound = set(map(tuple, mask_boundary(predicted_mask)))
    gt_bound = set(map(tuple, mask_boundary(ground_truth_mask)))

    # Handle edge cases where one or both boundaries are empty
    if len(pred_bound) == 0 and len(gt_bound) == 0:
        return 1.0
    if len(pred_bound) == 0 or len(gt_bound) == 0:
        return 0.0

    def is_within_tolerance(pixel, target_set, tol):
        """Check if pixel is within taxicab distance of any pixel in target_set."""
        px, py = pixel
        for dx in range(-tol, tol + 1):
            for dy in range(-tol + abs(dx), tol - abs(dx) + 1):
                if (px + dx, py + dy) in target_set:
                    return True
        return False

    # Calculate precision: fraction of predicted boundary pixels that match ground truth
    true_positives_precision = sum(
        1 for p in pred_bound if is_within_tolerance(p, gt_bound, tolerance)
    )
    precision = true_positives_precision / len(pred_bound)

    # Calculate recall: fraction of ground truth boundary pixels that match prediction
    true_positives_recall = sum(
        1 for p in gt_bound if is_within_tolerance(p, pred_bound, tolerance)
    )
    recall = true_positives_recall / len(gt_bound)

    # Calculate F-score
    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def boundary_f_score_video(predicted_masks, ground_truth_masks, tolerance=2):
    """
    Calculate boundary F-score for video segmentation.

    Args:
        predicted_masks: 2D array where outer array represents frames and inner arrays
                         contain (x, y) pixel coordinate tuples for predicted mask
        ground_truth_masks: 2D array where outer array represents frames and inner arrays
                            contain (x, y) pixel coordinate tuples for ground truth mask
        tolerance: Maximum taxicab distance for a predicted boundary pixel to be considered a match

    Returns:
        float: Average boundary F-score across all frames
    """
    if len(predicted_masks) != len(ground_truth_masks):
        raise ValueError(
            "Predicted and ground truth masks must have the same number of frames"
        )

    if len(predicted_masks) == 0:
        return 0.0

    bf_scores = []
    for pred_frame, gt_frame in zip(predicted_masks, ground_truth_masks):
        frame_bf = boundary_f_score(pred_frame, gt_frame, tolerance)
        bf_scores.append(frame_bf)

    return np.mean(bf_scores)


def jitter(predicted_masks):
    """Calculate the Jitter metric of a sequence of masks.

    Args:
        predicted_masks: 2D array where outer array represents frames and inner arrays
                contain (x, y) pixel coordinate tuples for predicted mask

    Returns:
        float: Jitter metric value
    """
    self_ious = [
        iou(mask1, mask2) for mask1, mask2 in itertools.pairwise(predicted_masks)
    ]
    return 1 - np.mean(self_ious)
