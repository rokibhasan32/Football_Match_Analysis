from utils import read_video, save_video
from trackers.tracker import Tracker
import cv2
import numpy as np
from team_assigner.team_assigner import TeamAssigner
from player_ball_assigner.player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator
from view_transformer.view_transformer import ViewTransformer
from speed_and_distance_estimator.speed_and_distance_estimator import SpeedAndDistance_Estimator
import os


def process_video_in_batches(input_path, output_path, batch_size=50):
    """Process video in batches to avoid memory issues"""
    
    print(f"Processing video in batches of {batch_size} frames...")
    
    # Initialize components
    tracker = Tracker('models/best.pt')
    team_assigner = TeamAssigner()
    player_assigner = PlayerBallAssigner()
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    
    # Read first frame to initialize camera movement estimator
    cap = cv2.VideoCapture(input_path)
    ret, first_frame = cap.read()
    if not ret:
        print("Error reading first frame")
        return
    
    camera_movement_estimator = CameraMovementEstimator(first_frame)
    view_transformer = ViewTransformer()
    
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")
    
    # Process video batch by batch
    all_output_frames = []
    frame_offset = 0
    
    while frame_offset < total_frames:
        print(f"Processing batch: frames {frame_offset} to {min(frame_offset + batch_size, total_frames)}")
        
        # Read batch of frames
        batch_frames = []
        for i in range(batch_size):
            ret, frame = cap.read()
            if not ret:
                break
            batch_frames.append(frame)
        
        if not batch_frames:
            break
            
        # Process this batch
        output_batch = process_batch(
            batch_frames, frame_offset, tracker, team_assigner, player_assigner,
            camera_movement_estimator, view_transformer, speed_and_distance_estimator
        )
        
        all_output_frames.extend(output_batch)
        frame_offset += len(batch_frames)
        
        # Clear memory
        del batch_frames
        del output_batch
        
    cap.release()
    
    # Save final video
    print(f"Saving output video with {len(all_output_frames)} frames...")
    save_video(all_output_frames, output_path)
    print("Video processing completed successfully!")


def process_batch(batch_frames, frame_offset, tracker, team_assigner, player_assigner,
                  camera_movement_estimator, view_transformer, speed_and_distance_estimator):
    """Process a single batch of frames"""
    
    # Get tracks for this batch
    tracks = tracker.get_object_tracks(
        batch_frames,
        read_from_stub=False,
        stub_path=None  # Don't use stubs for batch processing
    )
    
    # Ensure tracks have same number of frames
    num_frames = len(batch_frames)
    for track_type in ['players', 'referees', 'ball']:
        current_len = len(tracks[track_type])
        if current_len != num_frames:
            if current_len < num_frames:
                tracks[track_type].extend([{}] * (num_frames - current_len))
            else:
                tracks[track_type] = tracks[track_type][:num_frames]
    
    # Add positions to tracks
    tracker.add_position_to_tracks(tracks)
    
    # Camera movement estimation for this batch
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        batch_frames,
        read_from_stub=False,
        stub_path=None
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    
    # View transformation
    view_transformer.add_transformed_position_to_tracks(tracks)
    
    # Interpolate ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    # Speed and distance estimation
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    
    # Team assignment (only for first batch or when needed)
    if frame_offset == 0:
        team_assigner.assign_team_color(batch_frames[0], tracks['players'][0])
    
    # Assign teams to players
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                batch_frames[frame_num],
                track['bbox'],
                player_id
            )
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    
    # Ball assignment with NaN checking
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        # Safely get ball bbox
        ball_bbox = None
        if (1 in tracks['ball'][frame_num] and 
            'bbox' in tracks['ball'][frame_num][1] and
            len(tracks['ball'][frame_num][1]['bbox']) == 4):
            ball_bbox = tracks['ball'][frame_num][1]['bbox']
            # Check for NaN values
            import numpy as np
            if any(np.isnan(coord) for coord in ball_bbox):
                ball_bbox = None
        
        if ball_bbox is not None:
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
            if assigned_player != -1:
                tracks['players'][frame_num][assigned_player]['has_ball'] = True
                team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
            else:
                team_ball_control.append(team_ball_control[-1] if team_ball_control else 1)
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 1)
    
    team_ball_control = np.array(team_ball_control)
    
    # Draw annotations
    output_frames = tracker.draw_annotations(batch_frames, tracks, team_ball_control)
    
    # Draw camera movement
    output_frames = camera_movement_estimator.draw_camera_movement(output_frames, camera_movement_per_frame)
    
    # Draw speed and distance
    output_frames = speed_and_distance_estimator.draw_speed_and_distance(output_frames, tracks)
    
    return output_frames


def main():
    input_path = 'input_videos/Data-1.mp4'
    output_path = 'output_videos/output_video.avi'
    
    # Create output directory if it doesn't exist
    os.makedirs('output_videos', exist_ok=True)
    
    # Process video in batches to avoid memory issues
    process_video_in_batches(input_path, output_path, batch_size=50)


if __name__ == '__main__':
    main()