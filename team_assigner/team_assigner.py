from sklearn.cluster import KMeans

class SquadClassifier:
    def __init__(self):
        self.team_colors = {}
        self.pid_to_squad = {}
    
    def construct_clustering_algorithm(self, img_data):
        # Flatten image to 2D feature space
        flat_img = img_data.reshape(-1, 3)

        # Apply K-means partitioning into 2 groups
        km = KMeans(n_clusters=2, init="k-means++", n_init=1)
        km.fit(flat_img)

        return km

    def extract_dominant_player_color(self, frame, bbox):
        cropped = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        upper_region = cropped[0:int(cropped.shape[0]/2), :]

        # Compute color clustering
        km = self.construct_clustering_algorithm(upper_region)

        # Retrieve cluster assignments
        assignments = km.labels_

        # Reshape back to image dimensions
        labeled_img = assignments.reshape(upper_region.shape[0], upper_region.shape[1])

        # Identify background cluster
        corners = [labeled_img[0,0], labeled_img[0,-1], labeled_img[-1,0], labeled_img[-1,-1]]
        bg_cluster = max(set(corners), key=corners.count)
        fg_cluster = 1 - bg_cluster

        dominant_shade = km.cluster_centers_[fg_cluster]

        return dominant_shade


    def initialize_team_colors(self, frame, initial_players):
        
        player_shades = []
        for _, detect in initial_players.items():
            b_coords = detect["bbox"]
            shade = self.extract_dominant_player_color(frame, b_coords)
            player_shades.append(shade)
        
        km = KMeans(n_clusters=2, init="k-means++", n_init=10)
        km.fit(player_shades)

        self.kmeans = km

        self.team_colors[1] = km.cluster_centers_[0]
        self.team_colors[2] = km.cluster_centers_[1]


    def identify_player_squad(self, frame, player_box, player_id):
        if player_id in self.pid_to_squad:
            return self.pid_to_squad[player_id]

        shade = self.extract_dominant_player_color(frame, player_box)

        squad = self.kmeans.predict(shade.reshape(1, -1))[0]
        squad += 1

        if player_id == 91:
            squad = 1

        self.pid_to_squad[player_id] = squad

        return squad
