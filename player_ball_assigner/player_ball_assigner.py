import sys 
sys.path.append('../')
from utils import compute_bbox_centroid, calculate_euclidean_distance

class BallPossessionDetector():
    def __init__(self):
        self.possession_threshold = 70
    
    def match_ball_to_nearest_player(self, all_players, ball_bbox):
        ball_center = compute_bbox_centroid(ball_bbox)

        closest_distance = float('inf')
        closest_player = -1

        for pid, p_data in all_players.items():
            p_bbox = p_data['bbox']

            left_foot_dist = calculate_euclidean_distance((p_bbox[0], p_bbox[-1]), ball_center)
            right_foot_dist = calculate_euclidean_distance((p_bbox[2], p_bbox[-1]), ball_center)
            proximity = min(left_foot_dist, right_foot_dist)

            if proximity < self.possession_threshold:
                if proximity < closest_distance:
                    closest_distance = proximity
                    closest_player = pid

        return closest_player