import cv2
import sys 
sys.path.append('../')
from utils import calculate_euclidean_distance, get_base_position

class MotionMetricsCalculator():
    def __init__(self):
        self.step_interval = 5
        self.fps = 24
    
    def add_speed_and_distance_to_tracks(self, detection_tracks):
        cumulative_distance = {}

        for entity_type, entity_frames in detection_tracks.items():
            if entity_type == "ball" or entity_type == "referees":
                continue 
            total_frames = len(entity_frames)
            for idx in range(0, total_frames, self.step_interval):
                end_idx = min(idx + self.step_interval, total_frames - 1)

                for oid, _ in entity_frames[idx].items():
                    if oid not in entity_frames[end_idx]:
                        continue

                    start_pos = entity_frames[idx][oid]['position_transformed']
                    end_pos = entity_frames[end_idx][oid]['position_transformed']

                    if start_pos is None or end_pos is None:
                        continue
                    
                    disp_dist = calculate_euclidean_distance(start_pos, end_pos)
                    time_delta = (end_idx - idx) / self.fps
                    velocity_mps = disp_dist / time_delta
                    velocity_kmph = velocity_mps * 3.6

                    if entity_type not in cumulative_distance:
                        cumulative_distance[entity_type] = {}
                    
                    if oid not in cumulative_distance[entity_type]:
                        cumulative_distance[entity_type][oid] = 0
                    
                    cumulative_distance[entity_type][oid] += disp_dist

                    for batch_idx in range(idx, end_idx):
                        if oid not in detection_tracks[entity_type][batch_idx]:
                            continue
                        detection_tracks[entity_type][batch_idx][oid]['speed'] = velocity_kmph
                        detection_tracks[entity_type][batch_idx][oid]['distance'] = cumulative_distance[entity_type][oid]
    
    def draw_speed_and_distance(self, frames, detection_tracks):
        annotated_frames = []
        for idx, frame in enumerate(frames):
            for entity_type, entity_frames in detection_tracks.items():
                if entity_type == "ball" or entity_type == "referees":
                    continue 
                for _, obj_data in entity_frames[idx].items():
                   if "speed" in obj_data:
                       vel = obj_data.get('speed', None)
                       dist = obj_data.get('distance', None)
                       if vel is None or dist is None:
                           continue
                       
                       bbox = obj_data['bbox']
                       pos = get_base_position(bbox)
                       pos = list(pos)
                       pos[1] += 40

                       pos = tuple(map(int, pos))
                       cv2.putText(frame, f"{vel:.2f} km/h", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                       cv2.putText(frame, f"{dist:.2f} m", (pos[0], pos[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            annotated_frames.append(frame)
        
        return annotated_frames