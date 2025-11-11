import cv2
import numpy as np
import torch
import torch.nn.functional as F
import sys
import os

sys.path.append('../')

class XAIAnalyzer:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
    def analyze_tactical_patterns(self, tracks, frame_num):
        """
        Analyze tactical patterns using tracking data
        """
        try:
            players = tracks["players"][frame_num]
            ball_data = tracks["ball"][frame_num]
            
            analysis = {
                "formation": self._detect_formation(players),
                "pressure_zones": self._detect_pressure_zones(players, ball_data),
                "passing_lanes": self._detect_passing_lanes(players, ball_data),
                "defensive_line": self._detect_defensive_line(players)
            }
            return analysis
        except Exception as e:
            return {
                "formation": "Analysis failed",
                "pressure_zones": "Analysis failed", 
                "passing_lanes": "Analysis failed",
                "defensive_line": "Analysis failed"
            }
    
    def _detect_formation(self, players):
        """Detect team formation"""
        if not players:
            return "No players detected"
            
        team_positions = {1: [], 2: []}
        for player_id, player in players.items():
            if 'position' in player and 'team' in player:
                team_positions[player['team']].append(player['position'])
        
        formations = []
        for team, positions in team_positions.items():
            if len(positions) >= 5:
                x_coords = [pos[0] for pos in positions]
                formation = self._classify_formation(x_coords)
                formations.append(f"Team {team}: {formation}")
        
        return " vs ".join(formations) if formations else "Formation unclear"
    
    def _classify_formation(self, x_coords):
        """Classify formation based on player distribution"""
        if len(x_coords) < 5:
            return "Unknown"
            
        sorted_x = sorted(x_coords)
        spread = max(x_coords) - min(x_coords)
        
        if spread < 0.4:
            return "Compact"
        elif spread > 0.7:
            return "Wide"
        else:
            return "Balanced"
    
    def _detect_pressure_zones(self, players, ball_data):
        """Detect high-pressure zones"""
        if not players or not ball_data or 1 not in ball_data:
            return "No pressure data"
        
        ball_pos = ball_data[1].get('position', None)
        if not ball_pos:
            return "No ball position"
        
        ball_x, ball_y = ball_pos
        pressure_count = 0
        
        for player_id, player in players.items():
            if 'position' in player:
                player_x, player_y = player['position']
                distance = ((player_x - ball_x)**2 + (player_y - ball_y)**2)**0.5
                if distance < 60:
                    pressure_count += 1
        
        if pressure_count >= 3:
            return "High pressure"
        elif pressure_count >= 2:
            return "Medium pressure"
        else:
            return "Low pressure"
    
    def _detect_passing_lanes(self, players, ball_data):
        """Detect potential passing lanes"""
        if not players or not ball_data or 1 not in ball_data:
            return "No passing data"
        
        ball_pos = ball_data[1].get('position', None)
        if not ball_pos:
            return "No ball position"
        
        open_players = 0
        for player_id, player in players.items():
            if 'position' in player and 'team' in player:
                player_pos = player['position']
                if self._is_good_passing_option(ball_pos, player_pos):
                    open_players += 1
        
        return f"{open_players} open players"
    
    def _is_good_passing_option(self, ball_pos, player_pos):
        """Determine if a player is a good passing option"""
        ball_x, ball_y = ball_pos
        player_x, player_y = player_pos
        
        distance = ((ball_x - player_x)**2 + (ball_y - player_y)**2)**0.5
        return 20 < distance < 100
    
    def _detect_defensive_line(self, players):
        """Detect defensive line position"""
        if not players:
            return "No defensive data"
        
        defensive_positions = []
        for player_id, player in players.items():
            if 'position' in player and player.get('team') in [1, 2]:
                defensive_positions.append(player['position'][0])
        
        if defensive_positions:
            avg_line = np.mean(defensive_positions)
            if avg_line < 0.4:
                return "Deep defense"
            elif avg_line < 0.7:
                return "Medium defense"
            else:
                return "High defense"
        
        return "Defense unclear"