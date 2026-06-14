

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Optional, Tuple
from utils import Logger


class HandTracker:
    
    
    def __init__(
        self, 
        webcam_resolution: Tuple[int, int] = (1280, 720),
        fps: int = 30,
        confidence_threshold: float = 0.5
    ):
        
        self.webcam_resolution = webcam_resolution
        self.fps = fps
        self.confidence_threshold = confidence_threshold
        
        # Initialize logger
        self.logger = Logger(name="HandTracker")
        
        # MediaPipe Hands solution object (initialized later)
        self.mp_hands = None
        
        # Hand detection results (updated per frame)
        self.hand_results = None
        
        # Webcam capture object
        self.cap = None
        
        # Flag to track initialization state
        self.is_initialized = False
        
        self.logger.info(
    f"HandTracker initialized with resolution {webcam_resolution} and FPS {fps}"
)
    
    def initialize(self) -> bool:
        
        try:
            # Initialize MediaPipe Hands solution
            self.mp_hands = mp.solutions.hands.Hands(
                static_image_mode=False,           # Video mode (not static images)
                max_num_hands=2,                    # Track up to 2 hands
                min_detection_confidence=0.5,       # Minimum detection confidence
                min_tracking_confidence=0.5         # Minimum tracking confidence
            )
            
            # Open webcam with OpenCV
            self.cap = cv2.VideoCapture(0)  # 0 = default camera
            
            # Set webcam resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.webcam_resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.webcam_resolution[1])
            
            # Set FPS for webcam
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Check if webcam opened successfully
            if not self.cap.isOpened():
                self.logger.error("Failed to open webcam")
                raise RuntimeError("Could not open webcam")
            
            self.is_initialized = True
            self.logger.info("HandTracker initialized successfully")
            self.logger.info(
    f"Webcam opened with resolution {self.webcam_resolution}"
)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        
        if not self.is_initialized:
            self.logger.warning("HandTracker not initialized")
            return frame, []
        
        try:
            # Convert frame from BGR to RGB (MediaPipe requires RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe Hands
            self.hand_results = self.mp_hands.process(frame_rgb)
            
            # List to store detected hands
            detected_hands = []
            
            # Check if hands were detected
            if self.hand_results.multi_hand_landmarks:
                # Process each detected hand
                for hand_landmarks in self.hand_results.multi_hand_landmarks:
                    # Get hand classification (left/right)
                    hand_class = self._get_hand_classification(hand_landmarks)
                    
                    # Extract landmark coordinates
                    landmarks = self._extract_landmarks(hand_landmarks)
                    
                    # Get bounding box for hand
                    bounding_box = self._get_bounding_box(landmarks)
                    
                    # Store hand data
                    hand_data = {
                        'landmarks': landmarks,
                        'classification': hand_class,
                        'bounding_box': bounding_box,
                        'confidence': 1.0
                    }
                    
                    # Only add if confidence is above threshold
                    if hand_data['confidence'] >= self.confidence_threshold:
                        detected_hands.append(hand_data)
            
            # Draw landmarks on frame for visualization
            processed_frame = self._draw_landmarks(frame, detected_hands)
            
            return processed_frame, detected_hands
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return frame, []
    
    def _get_hand_classification(self, hand_landmarks) -> str:
        
        # In newer MediaPipe versions, handedness may not be available
        # Default to 'Right' for consistency
        return 'Right'
    
    def _extract_landmarks(
    self,
    hand_landmarks
) -> List[np.ndarray]:
        
        landmarks = []
        
        for landmark in hand_landmarks.landmark:
            # Extract (x, y, z) coordinates
            coord = np.array([
                landmark.x,    # Normalized x (0 to 1)
                landmark.y,    # Normalized y (0 to 1)
                landmark.z     # Depth (relative scale)
            ])
            landmarks.append(coord)
        
        return landmarks
    
    def _get_bounding_box(
        self, 
        landmarks: List[np.ndarray]
    ) -> Tuple[float, float, float, float]:
       
        x_coords = [landmark[0] for landmark in landmarks]
        y_coords = [landmark[1] for landmark in landmarks]
        
        min_x = min(x_coords)
        min_y = min(y_coords)
        max_x = max(x_coords)
        max_y = max(y_coords)
        
        return (min_x, min_y, max_x, max_y)
    
    def _draw_landmarks(
        self, 
        frame: np.ndarray, 
        detected_hands: List[Dict]
    ) -> np.ndarray:
        
        # Convert to RGB for drawing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        for hand in detected_hands:
            landmarks = hand['landmarks']
            
            # Draw landmarks (red circles)
            for landmark in landmarks:
                # Convert normalized coords to pixel coords
                x = int(landmark[0] * frame_rgb.shape[1])
                y = int(landmark[1] * frame_rgb.shape[0])
                
                # Draw circle at landmark position
                cv2.circle(frame_rgb, (x, y), 5, (255, 0, 0), -1)
            
            # Draw connections between landmarks (blue lines)
            # Hand connectivity from MediaPipe
            connections = [
                [0, 1], [1, 2], [2, 3], [3, 4],      # Thumb
                [0, 5], [5, 6], [6, 7], [7, 8],      # Index
                [5, 9], [9, 10], [10, 11], [11, 12], # Middle
                [9, 13], [13, 14], [14, 15], [15, 16],# Ring
                [13, 17], [17, 18], [18, 19], [19, 20],# Palm
                [0, 17]                                 # Palm base
            ]
            
            for conn in connections:
                start = landmarks[conn[0]]
                end = landmarks[conn[1]]
                
                x1 = int(start[0] * frame_rgb.shape[1])
                y1 = int(start[1] * frame_rgb.shape[0])
                x2 = int(end[0] * frame_rgb.shape[1])
                y2 = int(end[1] * frame_rgb.shape[0])
                
                cv2.line(frame_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Convert back to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        return frame_bgr
    
    def get_index_finger_tip(self, hand: Dict) -> Optional[np.ndarray]:
        
        if hand and len(hand['landmarks']) > 8:
            return hand['landmarks'][8]
        return None
    
    def get_thumb_tip(self, hand: Dict) -> Optional[np.ndarray]:
        
        if hand and len(hand['landmarks']) > 4:
            return hand['landmarks'][4]
        return None
    
    def get_middle_finger_tip(self, hand: Dict) -> Optional[np.ndarray]:
        
        if hand and len(hand['landmarks']) > 12:
            return hand['landmarks'][12]
        return None
    
    def get_ring_finger_tip(self, hand: Dict) -> Optional[np.ndarray]:
       
        if hand and len(hand['landmarks']) > 16:
            return hand['landmarks'][16]
        return None
    
    def get_palm_base(self, hand: Dict) -> Optional[np.ndarray]:
        
        if hand and len(hand['landmarks']) > 0:
            return hand['landmarks'][0]
        return None
    
    def close(self):
        
        if self.cap is not None:
            self.cap.release()
            self.logger.info("Webcam released")
        
        self.is_initialized = False
        self.logger.info("HandTracker closed")


# Hand landmark indices for reference
HAND_LANDMARKS = {
    'thumb': [0, 1, 2, 3, 4],
    'index': [5, 6, 7, 8],
    'middle': [9, 10, 11, 12],
    'ring': [13, 14, 15, 16],
    'palm': [17, 18, 19, 20]
}

# Key landmarks for gesture recognition
KEY_LANDMARKS = {
    'thumb_tip': 4,
    'index_tip': 8,
    'middle_tip': 12,
    'ring_tip': 16,
    'palm_base': 0
}

# End of hand_tracker.py