def compute_bbox_centroid(bbox):
    left, top, right, bottom = bbox
    cx = int((left + right) / 2)
    cy = int((top + bottom) / 2)
    return cx, cy

def compute_bbox_horizontal_span(bbox):
    _, _, right, _ = bbox
    return right - bbox[0]

def calculate_euclidean_distance(pt_a, pt_b):
    dx = pt_a[0] - pt_b[0]
    dy = pt_a[1] - pt_b[1]
    dist = (dx**2 + dy**2)**0.5
    return dist

def calculate_vector_offset(pt_a, pt_b):
    offset_x = pt_a[0] - pt_b[0]
    offset_y = pt_a[1] - pt_b[1]
    return offset_x, offset_y

def get_base_position(bbox):
    left, _, right, bottom = bbox
    base_x = int((left + right) / 2)
    base_y = int(bottom)
    return base_x, base_y