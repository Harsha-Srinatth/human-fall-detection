"""Geometry-based fall detection using bounding box aspect ratio."""


def get_box_size(x1, y1, x2, y2):
    """Return width and height of a bounding box."""
    return x2 - x1, y2 - y1


def is_possible_fall(x1, y1, x2, y2):
    """
    Detect a possible fall from box shape.

    Standing: height > width
    Lying:    width > height  -> possible fall
    """
    width, height = get_box_size(x1, y1, x2, y2)
    return width > height


def get_status(x1, y1, x2, y2):
    """Return status label and whether a possible fall was detected."""
    if is_possible_fall(x1, y1, x2, y2):
        return "Possible Fall", True
    return "Standing", False
