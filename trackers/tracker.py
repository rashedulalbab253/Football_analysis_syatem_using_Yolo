from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
sys.path.append('../')
from utils import compute_bbox_centroid, compute_bbox_horizontal_span, get_base_position

class ObjectTracker:
    def __init__(self, weights_path):
        self.detector = YOLO(weights_path) 
        self.motion_tracker = sv.ByteTrack()

    def add_position_to_tracks(self, detection_tracks):
        for entity_type, entity_frames in detection_tracks.items():
            for idx, frame_data in enumerate(entity_frames):
                for oid, obj_info in frame_data.items():
                    bbox_coords = obj_info['bbox']
                    if entity_type == 'ball':
                        computed_pos = compute_bbox_centroid(bbox_coords)
                    else:
                        computed_pos = get_base_position(bbox_coords)
                    detection_tracks[entity_type][idx][oid]['position'] = computed_pos

    def interpolate_ball_positions(self, ball_data):
        ball_coords = [x.get(1, {}).get('bbox', []) for x in ball_data]
        df_coords = pd.DataFrame(ball_coords, columns=['x1', 'y1', 'x2', 'y2'])

        # Fill missing values using interpolation
        df_coords = df_coords.interpolate()
        df_coords = df_coords.bfill()

        reconstructed_data = [{1: {"bbox": row}} for row in df_coords.to_numpy().tolist()]

        return reconstructed_data

    def detect_frames(self, frames):
        processing_batch = 20 
        all_detections = [] 
        for start_idx in range(0, len(frames), processing_batch):
            batch = self.detector.predict(frames[start_idx:start_idx+processing_batch], conf=0.1)
            all_detections += batch
        return all_detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as cache:
                detection_tracks = pickle.load(cache)
            return detection_tracks

        detections = self.detect_frames(frames)

        detection_tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for idx, detection in enumerate(detections):
            class_mapping = detection.names
            inverse_mapping = {v: k for k, v in class_mapping.items()}

            # Convert to supervision format
            sv_detection = sv.Detections.from_ultralytics(detection)

            # Reclassify goalkeeper as player
            for obj_idx, obj_class in enumerate(sv_detection.class_id):
                if class_mapping[obj_class] == "goalkeeper":
                    sv_detection.class_id[obj_idx] = inverse_mapping["player"]

            # Update tracking
            tracked_detections = self.motion_tracker.update_with_detections(sv_detection)

            detection_tracks["players"].append({})
            detection_tracks["referees"].append({})
            detection_tracks["ball"].append({})

            for detection_obj in tracked_detections:
                bbox = detection_obj[0].tolist()
                class_id = detection_obj[3]
                id_num = detection_obj[4]

                if class_id == inverse_mapping['player']:
                    detection_tracks["players"][idx][id_num] = {"bbox": bbox}
                
                if class_id == inverse_mapping['referee']:
                    detection_tracks["referees"][idx][id_num] = {"bbox": bbox}
            
            for sv_obj in sv_detection:
                bbox = sv_obj[0].tolist()
                class_id = sv_obj[3]

                if class_id == inverse_mapping['ball']:
                    detection_tracks["ball"][idx][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as cache:
                pickle.dump(detection_tracks, cache)

        return detection_tracks
    
    def render_circle_marker(self, img, bbox, rgb_color, id_label=None):
        bottom_y = int(bbox[3])
        center_x, _ = compute_bbox_centroid(bbox)
        span = compute_bbox_horizontal_span(bbox)

        cv2.ellipse(
            img,
            center=(center_x, bottom_y),
            axes=(int(span), int(0.35*span)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=rgb_color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rect_w = 40
        rect_h = 20
        rect_x1 = center_x - rect_w//2
        rect_x2 = center_x + rect_w//2
        rect_y1 = (bottom_y - rect_h//2) + 15
        rect_y2 = (bottom_y + rect_h//2) + 15

        if id_label is not None:
            cv2.rectangle(img,
                          (int(rect_x1), int(rect_y1)),
                          (int(rect_x2), int(rect_y2)),
                          rgb_color,
                          cv2.FILLED)
            
            text_x = rect_x1 + 12
            if id_label > 99:
                text_x -= 10
            
            cv2.putText(
                img,
                f"{id_label}",
                (int(text_x), int(rect_y1+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return img

    def render_polygon_marker(self, img, bbox, rgb_color):
        top_y = int(bbox[1])
        center_x, _ = compute_bbox_centroid(bbox)

        polygon_coords = np.array([
            [center_x, top_y],
            [center_x-10, top_y-20],
            [center_x+10, top_y-20],
        ])
        cv2.drawContours(img, [polygon_coords], 0, rgb_color, cv2.FILLED)
        cv2.drawContours(img, [polygon_coords], 0, (0, 0, 0), 2)

        return img

    def render_possession_stats(self, img, current_frame, possession_array):
        # Create semi-transparent overlay
        temp_overlay = img.copy()
        cv2.rectangle(temp_overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        opacity = 0.4
        cv2.addWeighted(temp_overlay, opacity, img, 1 - opacity, 0, img)

        current_possession = possession_array[:current_frame+1]
        # Compute cumulative possession time
        team_a_frames = current_possession[current_possession==1].shape[0]
        team_b_frames = current_possession[current_possession==2].shape[0]
        team_a_pct = team_a_frames/(team_a_frames+team_b_frames)
        team_b_pct = team_b_frames/(team_a_frames+team_b_frames)

        cv2.putText(img, f"Team 1 Ball Control: {team_a_pct*100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(img, f"Team 2 Ball Control: {team_b_pct*100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return img

    def draw_annotations(self, video_frames, detection_tracks, possession_array):
        rendered_frames = []
        for frame_idx, frame in enumerate(video_frames):
            frame = frame.copy()

            players = detection_tracks["players"][frame_idx]
            balls = detection_tracks["ball"][frame_idx]
            refs = detection_tracks["referees"][frame_idx]

            # Render players
            for id_num, player_data in players.items():
                squad_color = player_data.get("team_color", (0, 0, 255))
                frame = self.render_circle_marker(frame, player_data["bbox"], squad_color, id_num)

                if player_data.get('has_ball', False):
                    frame = self.render_polygon_marker(frame, player_data["bbox"], (0, 0, 255))

            # Render officials
            for _, ref_data in refs.items():
                frame = self.render_circle_marker(frame, ref_data["bbox"], (0, 255, 255))
            
            # Render ball
            for _, ball_data in balls.items():
                frame = self.render_polygon_marker(frame, ball_data["bbox"], (0, 255, 0))

            # Render possession stats
            frame = self.render_possession_stats(frame, frame_idx, possession_array)

            rendered_frames.append(frame)

        return rendered_frames