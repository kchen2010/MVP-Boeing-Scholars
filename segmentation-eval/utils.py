import numpy as np
from PIL import Image


def mask_img_to_pixel_lists(mask_img: np.ndarray) -> list[np.ndarray]:
    """
    Convert an image with multiple masks (represented as different non-0 color
    values) to a list of numpy arrays of pixel coordinates, where each array is a mask.

    Args:
        mask_img (np.ndarray): An image containing masks.

    Returns:
        List[np.ndarray]: A list of numpy arrays with shape (N, 2) containing pixel coordinates.
    """
    return [
        np.column_stack(np.where(mask_img == color))
        for color in np.unique(mask_img)[1:]  # ignore black
    ]


def flatten_pixel_lists(pixel_lists: list[np.ndarray]) -> np.ndarray:
    """
    Flatten a list of numpy arrays of pixel coordinates into a single numpy array.

    Args:
        pixel_lists (list[np.ndarray]): A list of numpy arrays with shape (N, 2) containing pixel coordinates.

    Returns:
        np.ndarray: A numpy array with shape (N, 2) containing pixel coordinates.
    """
    return np.concatenate(pixel_lists, axis=0)


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    """
    Find the boundary pixels of a mask image.

    Args:
        mask (np.ndarray): A numpy array with shape (N, 2) containing pixel coordinates of the mask.

    Returns:
        np.ndarray: A numpy array with shape (N, 2) containing pixel coordinates of the boundary.
    """
    pixels = set(map(tuple, mask))
    boundary = [
        np.array([p[0], p[1]])
        for p in pixels
        if any(
            (p[0] - dx, p[1] - dy) not in pixels
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]
        )
    ]
    return np.array(boundary)
