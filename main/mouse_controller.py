
import numpy as np
import pyautogui
from typing import Optional, Tuple
from utils import Logger, smooth_coordinates, normalize_to_screen, get_screen_resolution


class MouseController:
    
    
    def __init__(self, smoothing_factor: float = 0.15):
        
        self.smoothing_factor = smoothing_factor
        self.logger = Logger(name="MouseController")
        
        # Get screen resolution
        self.screen_width, self.screen_height = get_screen_resolution()
        
        # Previous cursor position for smoothing
        self.previous_position = None
        
        # Mouse movement state
        self.is_moving = True
        
        # Configure pyautogui
        pyautogui.FAILSAFE = True  # Move mouse to corner to stop
        
        self.logger.info("MouseController initialized")
        self.logger.info(f"Screen resolution: {self.screen_width} x {self.screen_height}")
    
    def move_cursor(
        self, 
        index_tip_position: np.ndarray, 
        image_shape: Tuple[int, int]
    ) -> Tuple[int, int]:
       
        if index_tip_position is None:
            return None, None
        
        # Normalize hand coordinates to screen coordinates
        screen_coords = normalize_to_screen(
            index_tip_position, 
            image_shape,
            (self.screen_width, self.screen_height)
        )
        
        # Apply exponential smoothing
        if self.previous_position is not None:
            smoothed_coords = smooth_coordinates(
                screen_coords,
                self.previous_position,
                self.smoothing_factor
            )
        else:
            smoothed_coords = screen_coords
        
        # Update previous position
        self.previous_position = smoothed_coords
        
        # Convert to integer screen coordinates
        screen_x = int(smoothed_coords[0])
        screen_y = int(smoothed_coords[1])
        
        # Move mouse cursor
        if self.is_moving:
            pyautogui.moveTo(screen_x, screen_y, duration=0.01)
        
        self.logger.debug(f"Cursor moved to ({screen_x}, {screen_y})")
        
        return screen_x, screen_y
    
    def perform_left_click(self):
        
        pyautogui.click(button='left')
        self.logger.info("Left click performed")
    
    def perform_right_click(self):
        
        pyautogui.click(button='right')
        self.logger.info("Right click performed")
    
    def perform_double_click(self):
        
        pyautogui.click(button='left', clicks=2, interval=0.1)
        self.logger.info("Double click performed")
    
    def drag_mouse(
        self, 
        start_position: Tuple[int, int],
        end_position: Tuple[int, int],
        duration: float = 0.5
    ):
        
        pyautogui.drag(
            start_position[0], start_position[1],
            end_position[0], end_position[1],
            duration=duration
        )
        self.logger.info(f"Mouse drag from {start_position} to {end_position}")
    
    def scroll_mouse(self, clicks: int):
        
        pyautogui.scroll(clicks)
        self.logger.info(f"Mouse scrolled {clicks} clicks")
    
    def hover_over(self, x: int, y: int):
        
        pyautogui.moveTo(x, y, duration=0.01)
        self.logger.debug(f"Cursor hovered at ({x}, {y})")
    
    def stop_moving(self):
        
        self.is_moving = False
        self.logger.info("Cursor movement stopped")
    
    def start_moving(self):
       
        self.is_moving = True
        self.logger.info("Cursor movement started")
    
    def reset_position(self):
        
        self.previous_position = None
        self.logger.debug("Position reset")


# End of mouse_controller.py        this code have 9 error and what is the error