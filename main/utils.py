

import os
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple, Optional


class Logger:

    def __init__(self, name: str = "GesturePCControl", level: int = logging.INFO):

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Prevent duplicate handlers if logger already exists
        if not self.logger.handlers:
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)

            # Create formatter with timestamp, level, and message
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)

            # Add handler to logger
            self.logger.addHandler(console_handler)

    def info(self, message: str):
        """Log INFO level message"""
        self.logger.info(message)

    def debug(self, message: str):
        """Log DEBUG level message"""
        self.logger.debug(message)

    def warning(self, message: str):
        """Log WARNING level message"""
        self.logger.warning(message)

    def error(self, message: str):
        """Log ERROR level message"""
        self.logger.error(message)

    def critical(self, message: str):
        """Log CRITICAL level message"""
        self.logger.critical(message)


def calculate_distance(landmark1: np.ndarray, landmark2: np.ndarray) -> float:

    return np.sqrt(
        np.sum((landmark2 - landmark1) ** 2)
    )


def calculatenormalized_distance(landmark1: np.ndarray, landmark2: np.ndarray) -> float:

    return np.sqrt(
        np.sum((landmark2[:2] - landmark1[:2]) ** 2)
    )


def get_screen_resolution() -> Tuple[int, int]:

    import ctypes

    # Get Windows desktop width and height
    width = ctypes.windll.user32.GetSystemMetrics(0)
    height = ctypes.windll.user32.GetSystemMetrics(1)

    return width, height


def load_config(config_path: str = "config/gestures.json") -> Dict[Any, Any]:

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        print("Using default configuration...")
        return get_default_config()
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in configuration file: {e}")
        return get_default_config()


def get_default_config() -> Dict[Any, Any]:

    return {
        "mouse": {
            "left_click": {"threshold": 0.03},
            "right_click": {"fingers": [0, 1, 2]}
        },
        "volume": {
            "min_distance": 0.02,
            "max_distance": 0.15
        },
        "general": {
            "smoothing_factor": 0.15,
            "fps_update_interval": 0.5
        }
    }


def create_screenshots_folder(folder_path: str = "screenshots") -> str:

    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return folder_path
    except OSError as e:
        print(f"Error creating screenshots folder: {e}")
        return "screenshots"


def save_screenshot(image: np.ndarray, folder_path: str = "screenshots") -> str:

    try:
        folder = create_screenshots_folder(folder_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        filepath = os.path.join(folder, filename)

        import cv2
        cv2.imwrite(filepath, image)
        return filepath
    except Exception as e:
        print(f"Error saving screenshot: {e}")
        return ""


def smooth_coordinates(
        current_coords: np.ndarray,
        previous_coords: np.ndarray,
        smoothing_factor: float = 0.15
) -> np.ndarray:


    if previous_coords is None:
        return current_coords

    smoothed = (
            smoothing_factor * previous_coords +
            (1 - smoothing_factor) * current_coords
    )
    return smoothed


def normalize_to_screen(
        hand_coords: np.ndarray,
        image_shape: Tuple[int, int],
        screen_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Normalize hand coordinates from image space to screen space.

    Converts hand position detected in webcam image to corresponding
    screen position for mouse control. Handles aspect ratio differences.

    Args:
        hand_coords: Hand coordinates in image space (x, y)
        image_shape: Image dimensions (height, width)
        screen_shape: Screen dimensions (width, height)

    Returns:
        np.ndarray: Normalized coordinates in screen space

    Note:
        Image coordinates: x (0 to width), y (0 to height)
        MediaPipe returns normalized coordinates (0 to 1)
    """
    img_height, img_width = image_shape
    screen_width, screen_height = screen_shape

    # Convert normalized MediaPipe coords to screen coords
    screen_x = hand_coords[0] * screen_width
    screen_y = hand_coords[1] * screen_height

    return np.array([screen_x, screen_y])


def clamp_value(
        value: float,
        min_value: float,
        max_value: float
) -> float:

    return max(min_value, min(value, max_value))


def format_fps(fps: float) -> str:

    return f"{fps:.2f} FPS"


def get_timestamp() -> str:

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# End of utils.py