import pickle
import cv2
import numpy as np
import os
import sys 
sys.path.append('../')
from utils import calculate_euclidean_distance, calculate_vector_offset

class OpticalFlowAnalyzer():
    def __init__(self, initial_frame):
        self.distance_threshold = 5

        self.flow_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
        roi_mask = np.zeros_like(gray)
        roi_mask[:, 0:20] = 1
        roi_mask[:, 900:1050] = 1

        self.corner_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=roi_mask
        )

    def add_adjust_positions_to_tracks(self, detection_tracks, motion_data):
        for entity_type, entity_frames in detection_tracks.items():
            for idx, frame_data in enumerate(entity_frames):
                for oid, obj_info in frame_data.items():
                    pos = obj_info['position']
                    motion = motion_data[idx]
                    adjusted_pos = (pos[0]-motion[0], pos[1]-motion[1])
                    detection_tracks[entity_type][idx][oid]['position_adjusted'] = adjusted_pos
                    


    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        # Load from cache if available
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as cache:
                return pickle.load(cache)

        motion_sequence = [[0, 0]]*len(frames)

        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        prev_corners = cv2.goodFeaturesToTrack(prev_gray, **self.corner_params)

        for idx in range(1, len(frames)):
            curr_gray = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2GRAY)
            curr_corners, _, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_corners, None, **self.flow_params)

            max_disp = 0
            disp_x, disp_y = 0, 0

            for curr, prev in zip(curr_corners, prev_corners):
                curr_pt = curr.ravel()
                prev_pt = prev.ravel()

                disp = calculate_euclidean_distance(curr_pt, prev_pt)
                if disp > max_disp:
                    max_disp = disp
                    disp_x, disp_y = calculate_vector_offset(prev_pt, curr_pt)
            
            if max_disp > self.distance_threshold:
                motion_sequence[idx] = [disp_x, disp_y]
                prev_corners = cv2.goodFeaturesToTrack(curr_gray, **self.corner_params)

            prev_gray = curr_gray.copy()
        
        if stub_path is not None:
            with open(stub_path, 'wb') as cache:
                pickle.dump(motion_sequence, cache)

        return motion_sequence
    
    def draw_camera_movement(self, frames, motion_data):
        output_frames = []

        for idx, frame in enumerate(frames):
            frame = frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            blend = 0.6
            cv2.addWeighted(overlay, blend, frame, 1-blend, 0, frame)

            disp_x, disp_y = motion_data[idx]
            frame = cv2.putText(frame, f"Camera Movement X: {disp_x:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            frame = cv2.putText(frame, f"Camera Movement Y: {disp_y:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            output_frames.append(frame) 

        return output_frames