from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator


def execute_pipeline():
    # Load input footage
    frames_sequence = read_video('input_videos/08fd33_4.mp4')

    # Instantiate and configure tracker
    obj_tracker = Tracker('models/best.pt')

    detection_tracks = obj_tracker.get_object_tracks(frames_sequence,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    # Calculate and store object center positions
    obj_tracker.add_position_to_tracks(detection_tracks)

    # Initialize camera motion processor
    cam_estimator = CameraMovementEstimator(frames_sequence[0])
    cam_motion_data = cam_estimator.get_camera_movement(frames_sequence,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    cam_estimator.add_adjust_positions_to_tracks(detection_tracks,cam_motion_data)


    # Configure perspective transformer
    perspective_transform = ViewTransformer()
    perspective_transform.add_transformed_position_to_tracks(detection_tracks)

    # Smooth and complete ball trajectory
    detection_tracks["ball"] = obj_tracker.interpolate_ball_positions(detection_tracks["ball"])

    # Compute velocity and displacement metrics
    motion_estimator = SpeedAndDistance_Estimator()
    motion_estimator.add_speed_and_distance_to_tracks(detection_tracks)

    # Associate players with their respective teams
    squad_assigner = TeamAssigner()
    squad_assigner.assign_team_color(frames_sequence[0], 
                                    detection_tracks['players'][0])
    
    for idx, players_in_frame in enumerate(detection_tracks['players']):
        for pid, info in players_in_frame.items():
            squad_id = squad_assigner.get_player_team(frames_sequence[idx],   
                                                 info['bbox'],
                                                 pid)
            detection_tracks['players'][idx][pid]['team'] = squad_id 
            detection_tracks['players'][idx][pid]['team_color'] = squad_assigner.team_colors[squad_id]

    
    # Determine ball possession
    possession_tracker = PlayerBallAssigner()
    possession_control= []
    for idx, players_in_frame in enumerate(detection_tracks['players']):
        ball_box = detection_tracks['ball'][idx][1]['bbox']
        possessing_player = possession_tracker.assign_ball_to_player(players_in_frame, ball_box)

        if possessing_player != -1:
            detection_tracks['players'][idx][possessing_player]['has_ball'] = True
            possession_control.append(detection_tracks['players'][idx][possessing_player]['team'])
        else:
            possession_control.append(possession_control[-1])
    possession_control= np.array(possession_control)


    # Render visual output
    ## Render player and object positions
    annotated_frames = obj_tracker.draw_annotations(frames_sequence, detection_tracks, possession_control)

    ## Render camera movement overlay
    annotated_frames = cam_estimator.draw_camera_movement(annotated_frames, cam_motion_data)

    ## Render motion metrics
    motion_estimator.draw_speed_and_distance(annotated_frames, detection_tracks)

    # Write processed video to file
    save_video(annotated_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    execute_pipeline()