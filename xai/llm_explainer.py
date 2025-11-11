import openai
from openai import OpenAI
import groq
import json
import os

class LLMExplainer:
    def __init__(self, api_key=None, provider="groq"):
        self.provider = provider
        self.api_key = api_key
        
        if provider == "openai" and api_key:
            self.client = OpenAI(api_key=api_key)
        elif provider == "groq" and api_key:
            self.client = groq.Groq(api_key=api_key)
        else:
            self.client = None
    
    def generate_match_report(self, analysis_data):
        """
        Generate comprehensive match report using LLM
        """
        if not self.client:
            return self._generate_fallback_report(analysis_data)
        
        prompt = self._create_report_prompt(analysis_data)
        
        try:
            if self.provider == "groq":
                response = self.client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=1500
                )
                return response.choices[0].message.content
            else:  # openai
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=1500
                )
                return response.choices[0].message.content
        except Exception as e:
            return f"LLM analysis failed: {str(e)}\n\n{self._generate_fallback_report(analysis_data)}"
    
    def _create_report_prompt(self, analysis_data):
        """Create prompt for match report"""
        return f"""
        You are a professional football analyst. Write a comprehensive match analysis report based on the following data:
        
        MATCH ANALYSIS DATA:
        {json.dumps(analysis_data, indent=2)}
        
        Please provide a detailed analysis including:
        1. Match overview and key statistics
        2. Team performance analysis based on possession and formations
        3. Key moments and turning points in the match
        4. Tactical analysis of both teams
        5. Standout performers based on speed and distance covered
        6. Overall assessment and recommendations
        
        Write in a professional, analytical tone suitable for coaching staff.
        """
    
    def _generate_fallback_report(self, analysis_data):
        """Generate a basic report when LLM is not available"""
        stats = analysis_data.get('match_statistics', {})
        events = analysis_data.get('key_events', {})
        
        report = "MATCH ANALYSIS REPORT (Basic Analysis)\n"
        report += "=" * 50 + "\n\n"
        
        # Basic statistics
        possession = stats.get('ball_possession', {})
        report += f"Ball Possession: Team 1: {possession.get('team_1', 'N/A')}, Team 2: {possession.get('team_2', 'N/A')}\n"
        report += f"Total Frames Analyzed: {stats.get('total_frames_analyzed', 0)}\n"
        report += f"Unique Players Detected: {stats.get('unique_players_detected', 0)}\n\n"
        
        # Top performers
        performers = stats.get('top_performers', {})
        if 'fastest_players' in performers:
            report += "Fastest Players:\n"
            for player in performers['fastest_players']:
                report += f"  - {player}\n"
        
        # Key events
        report += f"\nKey Events: {events.get('total_possession_changes', 0)} possession changes detected\n"
        
        return report