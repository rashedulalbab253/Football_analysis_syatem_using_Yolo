import numpy as np 
import cv2

class CoordinateTransformer():
    def __init__(self):
        field_width = 68
        field_length = 23.32

        self.source_coords = np.array([[110, 1035], 
                               [265, 275], 
                               [910, 260], 
                               [1640, 915]])
        
        self.dest_coords = np.array([
            [0, field_width],
            [0, 0],
            [field_length, 0],
            [field_length, field_width]
        ])

        self.source_coords = self.source_coords.astype(np.float32)
        self.dest_coords = self.dest_coords.astype(np.float32)

        self.homography_matrix = cv2.getPerspectiveTransform(self.source_coords, self.dest_coords)

    def transform_point(self, point):
        pt = (int(point[0]), int(point[1]))
        is_within_bounds = cv2.pointPolygonTest(self.source_coords, pt, False) >= 0 
        if not is_within_bounds:
            return None

        reshaped = point.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped, self.homography_matrix)
        return transformed.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, detection_tracks):
        for entity_type, entity_frames in detection_tracks.items():
            for idx, frame_data in enumerate(entity_frames):
                for oid, obj_info in frame_data.items():
                    adjusted_pos = obj_info['position_adjusted']
                    pos_array = np.array(adjusted_pos)
                    transformed_pos = self.transform_point(pos_array)
                    if transformed_pos is not None:
                        transformed_pos = transformed_pos.squeeze().tolist()
                    detection_tracks[entity_type][idx][oid]['position_transformed'] = transformed_pos