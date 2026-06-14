"""
gesture_recognizer.py - Gesture detection and recognition logic

This module implements gesture recognition algorithms using hand landmark
positions from MediaPipe. It detects various gestures including pinch,
open hand, pointing, and custom gestures for PC control.

Key Features:
- Pinch gesture detection (thumb + index)
- Open hand detection
- Pointing gesture recognition
- Finger state detection (raised/folded)
- Distance-based gesture recognition
- Configurable gesture thresholds
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from utils import Logger, calculatenormalized_distance, clamp_value
from hand_tracker import HAND_LANDMARKS, KEY_LANDMARKS


class GestureRecognizer:
    """
    Recognizes hand gestures from landmark coordinates.
    
    This class implements gesture recognition algorithms using geometric
    analysis of hand landmarks. It supports multiple gesture types with
    configurable thresholds for accurate detection.
    
    Attributes:
        pinch_threshold: Maximum distance for pinch gesture (default: 0.03)
        finger_threshold: Threshold for finger raised detection
    """
    
    def __init__(self, pinch_threshold: float = 0.03):
        """
        Initialize GestureRecognizer with configurable thresholds.
        
        Args:
            pinch_threshold: Maximum normalized distance between thumb and index
                           finger to detect pinch gesture (default: 0.03)
        
        Example:
            >>> recognizer = GestureRecognizer(pinch_threshold=0.03)
            >>> gesture = recognizer.recognize_gesture(hand)
        """
        self.pinch_threshold = pinch_threshold
        self.logger = Logger(name="GestureRecognizer")
        
        self.logger.info(f"GestureRecognizer initialized with pinch_threshold={pinch_threshold}")
    
    def recognize_gesture(self, hand: Dict) -> Dict:
        """
        Main gesture recognition method.
        
        Analyzes hand landmarks to determine the current gesture.
        Returns gesture name, description, and additional metadata.
        
        Args:
            hand: Detected hand dictionary from HandTracker
        
        Returns:
            Dict: Gesture information containing:
                - name: Gesture name (e.g., 'pinch', 'open_hand')
                - description: Human-readable description
                - metadata: Additional gesture data (distances, finger states)
        
        Example:
            >>> gesture = recognizer.recognize_gesture(hand)
            >>> print(f"Gesture: {gesture['name']}")
            >>> if gesture['name'] == 'pinch':
            >>>     print("Left click detected!")
        """
        if not hand or 'landmarks' not in hand:
            return {
                'name': 'unknown',
                'description': 'No hand detected',
                'metadata': {}
            }
        
        landmarks = hand['landmarks']
        
        # Get key landmark positions
        thumb_tip = landmarks[KEY_LANDMARKS['thumb_tip']]
        index_tip = landmarks[KEY_LANDMARKS['index_tip']]
        middle_tip = landmarks[KEY_LANDMARKS['middle_tip']]
        ring_tip = landmarks[KEY_LANDMARKS['ring_tip']]
        palm_base = landmarks[KEY_LANDMARKS['palm_base']]
        
        # Calculate distances
        thumb_index_distance = calculatenormalized_distance(thumb_tip, index_tip)
        
        # Get finger states (raised or folded)
        finger_states = self._get_finger_states(landmarks)
        
        # Detect gesture based on conditions
        gesture = self._detect_gesture(
            thumb_index_distance, 
            finger_states,
            landmarks,
            palm_base
        )
        
        # Add metadata
        gesture['metadata'] = {
            'thumb_index_distance': thumb_index_distance,
            'finger_states': finger_states,
            'all_landmarks': landmarks
        }
        
        return gesture
    
    def _detect_gesture(
        self, 
        thumb_index_distance: float,
        finger_states: Dict[str, bool],
        landmarks: List[np.ndarray],
        palm_base: np.ndarray
    ) -> Dict:
        """
        Detect specific gesture based on hand configuration.
        
        Implements gesture detection logic using:
        - Thumb-index distance for pinch
        - Finger states for open/closed hand
        - Geometric analysis for pointing
        
        Args:
            thumb_index_distance: Distance between thumb and index tips
            finger_states: Dictionary of finger raised states
            landmarks: All 21 hand landmarks
            palm_base: Palm base landmark
        
        Returns:
            Dict: Detected gesture information
        """
        
        # 1. PINCH GESTURE (Left Click)
        # Thumb and index finger close together
        if thumb_index_distance < self.pinch_threshold:
            return {
                'name': 'pinch',
                'description': 'Thumb and index finger pinched together',
                'action': 'left_click'
            }
        
        # 2. OPEN HAND (Neutral - No Action)
        # All five fingers raised
        if all(finger_states.values()):
            return {
                'name': 'open_hand',
                'description': 'All five fingers open',
                'action': 'neutral'
            }
        
        # 3. POINTING GESTURE (Brightness Control)
        # Only index finger raised
        if finger_states['index'] and not any([
            finger_states['thumb'],
            finger_states['middle'],
            finger_states['ring']
        ]):
            return {
                'name': 'pointing',
                'description': 'Index finger pointing',
                'action': 'brightness_control'
            }
        
        # 4. THREE FINGERS (Right Click)
        # Index, middle, ring fingers raised
        if finger_states['index'] and finger_states['middle'] and finger_states['ring']:
            if not finger_states['thumb'] and not finger_states['ring']:
                return {
                    'name': 'three_fingers',
                    'description': 'Three fingers raised (index, middle, ring)',
                    'action': 'right_click'
                }
        
        # 5. FIST (Closed Hand)
        # All fingers folded
        if not any(finger_states.values()):
            return {
                'name': 'fist',
                'description': 'All fingers closed (fist)',
                'action': 'none'
            }
        
        # 6. TWO FINGERS (V-sign / Victory)
        # Index and middle raised
        if finger_states['index'] and finger_states['middle']:
            if not finger_states['thumb'] and not finger_states['ring']:
                return {
                    'name': 'two_fingers',
                    'description': 'Index and middle fingers raised (V-sign)',
                    'action': 'none'
                }
        
        # 7. THUMBS UP
        if finger_states['thumb'] and not any([
            finger_states['index'],
            finger_states['middle'],
            finger_states['ring']
        ]):
            return {
                'name': 'thumbs_up',
                'description': 'Thumb raised up',
                'action': 'none'
            }
        
        # Default: Unknown gesture
        return {
            'name': 'unknown',
            'description': 'Unknown gesture configuration',
            'action': 'none'
        }
    
    def _get_finger_states(self, landmarks: List[np.ndarray]) -> Dict[str, bool]:
        """
        Determine if each finger is raised or folded.
        
        Uses geometric analysis: compares finger tip position to finger
        base position. If tip is "above" base (smaller y-coordinate),
        finger is considered raised.
        
        Args:
            landmarks: List of 21 hand landmark coordinates
        
        Returns:
            Dict[str, bool]: Dictionary with finger states
                - thumb: True if raised
                - index: True if raised
                - middle: True if raised
                - ring: True if raised
        """
        finger_states = {}
        
        # Check each finger
        for finger_name in ['thumb', 'index', 'middle', 'ring']:
            finger_indices = HAND_LANDMARKS[finger_name]
            
            # Get tip and base landmarks
            tip_idx = finger_indices[-1]  # Last index is tip
            base_idx = finger_indices[1]  # Second index is base
            
            tip = landmarks[tip_idx]
            base = landmarks[base_idx]
            
            # Finger is raised if tip y < base y (tip is above base)
            # Note: MediaPipe y-coordinates: 0 = top, 1 = bottom
            is_raised = tip[1] < base[1]
            
            finger_states[finger_name] = is_raised
        
        return finger_states
    
    def calculate_pinch_distance(self, hand: Dict) -> float:
        """
        Calculate normalized distance between thumb and index finger tips.
        
        Used for volume control - distance maps to volume percentage.
        
        Args:
            hand: Detected hand dictionary
        
        Returns:
            float: Normalized distance (0.0 to 1.0)
        
        Example:
            >>> distance = recognizer.calculate_pinch_distance(hand)
            >>> volume = int(distance * 100)
            >>> print(f"Volume: {volume}%")
        """
        if not hand or 'landmarks' not in hand:
            return 0.0
        
        landmarks = hand['landmarks']
        thumb_tip = landmarks[KEY_LANDMARKS['thumb_tip']]
        index_tip = landmarks[KEY_LANDMARKS['index_tip']]
        
        distance = calculatenormalized_distance(thumb_tip, index_tip)
        
        # Normalize to 0-1 range
        normalized_distance = clamp_value(distance, 0.0, 1.0)
        
        return normalized_distance
    
    def is_pinch_gesture(self, hand: Dict) -> bool:
        """
        Check if current gesture is a pinch (thumb + index closed).
        
        Simple boolean check for pinch gesture detection.
        
        Args:
            hand: Detected hand dictionary
        
        Returns:
            bool: True if pinch gesture detected, False otherwise
        
        Example:
            >>> if recognizer.is_pinch_gesture(hand):
            >>>     perform_left_click()
        """
        gesture = self.recognize_gesture(hand)
        return gesture['name'] == 'pinch'
    
    def is_open_hand(self, hand: Dict) -> bool:
        """
        Check if hand is fully open (all fingers raised).
        
        Args:
            hand: Detected hand dictionary
        
        Returns:
            bool: True if open hand, False otherwise
        """
        gesture = self.recognize_gesture(hand)
        return gesture['name'] == 'open_hand'
    
    def is_pointing_gesture(self, hand: Dict) -> bool:
        """
        Check if hand is in pointing gesture (index finger only).
        
        Args:
            hand: Detected hand dictionary
        
        Returns:
            bool: True if pointing, False otherwise
        """
        gesture = self.recognize_gesture(hand)
        return gesture['name'] == 'pointing'
    
    def get_finger_count(self, hand: Dict) -> int:
        """
        Count number of raised fingers.
        
        Args:
            hand: Detected hand dictionary
        
        Returns:
            int: Number of raised fingers (0-5)
        """
        if not hand or 'landmarks' not in hand:
            return 0
        
        landmarks = hand['landmarks']
        finger_states = self._get_finger_states(landmarks)
        
        return sum(finger_states.values())
    
    def calculate_hand_position(self, hand: Dict) -> Optional[np.ndarray]:
        """
        Calculate average hand position (palm center).
        
        Computes centroid of hand landmarks for cursor positioning.
        
        Args:
            hand: Detected hand dictionary
        
        Returns:
            Optional[np.ndarray]: Average (x, y) position or None
        """
        if not hand or 'landmarks' not in hand:
            return None
        
        landmarks = hand['landmarks']
        
        # Calculate average of all landmarks
        x_coords = [landmark[0] for landmark in landmarks]
        y_coords = [landmark[1] for landmark in landmarks]
        
        avg_x = np.mean(x_coords)
        avg_y = np.mean(y_coords)
        
        return np.array([avg_x, avg_y])


# End of gesture_recognizer.py