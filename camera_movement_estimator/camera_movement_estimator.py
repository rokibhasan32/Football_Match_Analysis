import pickle
import cv2
import numpy as np
import os
import sys 
sys.path.append('../')
from utils.bbox_utils import measure_distance, measure_xy_distance

class CameraMovementEstimator():
    def __init__(self, frame):
        self.minimum_distance = 5

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Create a better mask for feature detection
        mask_features = np.ones_like(first_frame_grayscale) * 255
        # Focus on areas more likely to have stable features (sidelines, field markings)
        h, w = first_frame_grayscale.shape
        mask_features[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)] = 0  # Exclude center field
        
        self.features = dict(
            maxCorners=200,  # Increased number of features
            qualityLevel=0.1,  # Lower quality threshold
            minDistance=10,
            blockSize=7,
            mask=mask_features
        )

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    if 'position' in track_info:
                        position = track_info['position']
                        if frame_num < len(camera_movement_per_frame):
                            camera_movement = camera_movement_per_frame[frame_num]
                            position_adjusted = (position[0]-camera_movement[0], position[1]-camera_movement[1])
                            tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        # Read the stub 
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            try:
                with open(stub_path, 'rb') as f:
                    camera_movement = pickle.load(f)
                print(f"Loaded camera movement from stub: {len(camera_movement)} frames")
                return camera_movement
            except Exception as e:
                print(f"Error loading camera movement stub: {e}. Regenerating...")

        camera_movement = [[0, 0]] * len(frames)

        if len(frames) == 0:
            return camera_movement

        # Initialize with first frame
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)
        
        if old_features is None:
            print("No features found in first frame, using default Shi-Tomasi parameters")
            # Fallback to default parameters
            self.features = dict(
                maxCorners=100,
                qualityLevel=0.01,  # Very low quality threshold
                minDistance=5,
                blockSize=7,
                mask=None  # No mask
            )
            old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        if old_features is None:
            print("Warning: Still no features detected. Camera movement will be zero.")
            return camera_movement

        print(f"Found {len(old_features)} features for camera movement tracking")

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            
            # Check if we have features to track
            if old_features is None or len(old_features) == 0:
                # Re-detect features
                old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)
                if old_features is None:
                    camera_movement[frame_num] = [0, 0]
                    old_gray = frame_gray.copy()
                    continue

            try:
                new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                    old_gray, frame_gray, old_features, None, **self.lk_params
                )
                
                # Check if tracking was successful
                if new_features is None:
                    camera_movement[frame_num] = [0, 0]
                    old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
                    old_gray = frame_gray.copy()
                    continue

                # Filter only good points
                good_new = new_features[status == 1]
                good_old = old_features[status == 1]

                if len(good_new) == 0:
                    camera_movement[frame_num] = [0, 0]
                    old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
                    old_gray = frame_gray.copy()
                    continue

                max_distance = 0
                camera_movement_x, camera_movement_y = 0, 0

                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    new_features_point = new.ravel()
                    old_features_point = old.ravel()

                    distance = measure_distance(new_features_point, old_features_point)
                    if distance > max_distance:
                        max_distance = distance
                        camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point)

                if max_distance > self.minimum_distance:
                    camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                    # Re-detect features periodically
                    if frame_num % 30 == 0:  # Every 30 frames
                        old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
                else:
                    camera_movement[frame_num] = [0, 0]
                    old_features = good_new.reshape(-1, 1, 2)

                old_gray = frame_gray.copy()

            except Exception as e:
                print(f"Error in optical flow at frame {frame_num}: {e}")
                camera_movement[frame_num] = [0, 0]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
                old_gray = frame_gray.copy()
                continue

        if stub_path is not None:
            try:
                os.makedirs(os.path.dirname(stub_path), exist_ok=True)
                with open(stub_path, 'wb') as f:
                    pickle.dump(camera_movement, f)
                print(f"Saved camera movement to stub: {len(camera_movement)} frames")
            except Exception as e:
                print(f"Error saving camera movement stub: {e}")

        return camera_movement

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            if frame_num >= len(camera_movement_per_frame):
                break

            frame = frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame, f"Camera Movement X: {x_movement:.2f}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            frame = cv2.putText(frame, f"Camera Movement Y: {y_movement:.2f}", (10, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            output_frames.append(frame)

        return output_frames