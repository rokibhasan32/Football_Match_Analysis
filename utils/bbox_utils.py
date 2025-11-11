import numpy as np

def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    # Check for NaN values
    if any(np.isnan(coord) for coord in [x1, y1, x2, y2]):
        return None
    return int((x1+x2)/2), int((y1+y2)/2)

def get_bbox_width(bbox):
    # Check for NaN values
    if any(np.isnan(coord) for coord in bbox):
        return 0
    return bbox[2]-bbox[0]

def measure_distance(p1, p2):
    # Check if points are valid
    if p1 is None or p2 is None:
        return float('inf')
    if any(np.isnan(coord) for coord in p1 + p2):
        return float('inf')
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def measure_xy_distance(p1, p2):
    # Check if points are valid
    if p1 is None or p2 is None:
        return 0, 0
    if any(np.isnan(coord) for coord in p1 + p2):
        return 0, 0
    return p1[0]-p2[0], p1[1]-p2[1]

def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    # Check for NaN values
    if any(np.isnan(coord) for coord in [x1, y1, x2, y2]):
        return None
    return int((x1+x2)/2), int(y2)