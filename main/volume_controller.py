

import numpy as np
import ctypes
from typing import Optional
from utils import Logger, clamp_value, calculatenormalized_distance
from hand_tracker import KEY_LANDMARKS


class VolumeController:
    
    
    def __init__(
        self, 
        min_distance: float = 0.02,
        max_distance: float = 0.15
    ):
       
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.logger = Logger(name="VolumeController")
        
        # Current volume level
        self.current_volume = 50  # Start at 50%
        
        # Windows volume API
        self.shell = ctypes.windll.shell32
        
        self.logger.info("VolumeController initialized")
        self.logger.info(f"Distance range: {min_distance} - {max_distance}")
    
    def set_volume_by_gesture(self, hand: dict) -> int:
       
        if not hand or 'landmarks' not in hand:
            return int(self.current_volume)
        
        landmarks = hand['landmarks']
        thumb_tip = landmarks[KEY_LANDMARKS['thumb_tip']]
        index_tip = landmarks[KEY_LANDMARKS['index_tip']]
        
        # Calculate distance
        distance = calculatenormalized_distance(thumb_tip, index_tip)
        
        # Map distance to volume percentage
        volume_percentage = self._map_distance_to_volume(distance)
        
        # Update volume if changed significantly (avoid rapid changes)
        if abs(volume_percentage - self.current_volume) > 2:
            self.current_volume = volume_percentage
            self._set_system_volume(volume_percentage)
            self.logger.info(f"Volume set to {volume_percentage}%")
        
        return int(self.current_volume)
    
    def _map_distance_to_volume(self, distance: float) -> int:
        
        # Clamp distance to valid range
        clamped_distance = clamp_value(distance, self.min_distance, self.max_distance)
        
        # Linear interpolation
        volume = (
            (clamped_distance - self.min_distance) / 
            (self.max_distance - self.min_distance) * 
            100
        )
        
        # Round and clamp to 0-100
        return int(clamp_value(volume, 0, 100))
    
    def _set_system_volume(self, volume_percentage: int):
        
        try:
            # Windows volume control using pycaw or direct API
            # For simplicity, use pycaw if available, otherwise fallback
            try:
                from pycaw import CTx
                from ctypes import cast, POINTER
                from ctypes import wintypes
                
                # Initialize audio endpoint
                # This is a simplified version - full implementation requires pycaw
                self.logger.debug(f"Setting volume to , {volume_percentage}%")
            except ImportError:
                # Fallback: Use pyautogui volume controls
                import pyautogui
                
                # Calculate difference from current
                diff = int (volume_percentage) - int(self.current_volume)
                
                if diff > 0:
                    # Increase volume
                    for _ in range(diff):
                        pyautogui.press('volumeup')
                elif diff < 0:
                    # Decrease volume
                    for _ in range(abs(diff)):
                        pyautogui.press('volumedown')
                
                self.logger.info("Volume adjusted using pyautogui")
        except Exception as e:
            self.logger.error(f"Failed to set volume: {str(e)}")
    
    def get_current_volume(self) -> int:
        
        return int(self.current_volume)
    
    def increase_volume(self, step: int = 5):
        
        new_volume = int(clamp_value(self.current_volume + step, 0, 100))
        self.current_volume = new_volume
        self._set_system_volume(new_volume)
        self.logger.info(f"Volume increased to {new_volume}%")
    
    def decrease_volume(self, step: int = 5):
        
        new_volume = int(clamp_value(self.current_volume - step, 0, 100))
        self.current_volume = new_volume
        self._set_system_volume(int(new_volume))
        self.logger.info(f"Volume decreased to {new_volume}%")
    
    def set_volume(self, volume_percentage: int):
        
        self.current_volume = int(clamp_value(volume_percentage, 0, 100))
        self._set_system_volume(self.current_volume)
        self.logger.info(f"Volume set to  {self.current_volume}%")


