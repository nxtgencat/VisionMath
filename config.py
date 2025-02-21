# Window Configuration
WINDOW_CONFIG = {
    'WIDTH': 950,
    'HEIGHT': 550,
    'BRIGHTNESS': 130
}

# Drawing Configuration
DRAWING_CONFIG = {
    'DRAW_COLOR': (255, 0, 255),  # Pink
    'DRAW_THICKNESS': 5,
    'ERASE_COLOR': (0, 0, 0),    # Black
    'ERASE_THICKNESS': 15
}

# Gesture Configuration
# Format: [thumb, index, middle, ring, pinky]
GESTURE_CONFIG = {
    'DRAW': {
        'fingers': [1, 1, 0, 0, 0],    # Thumb + Index
        'name': 'Drawing Mode'
    },
    'ERASE': {
        'fingers': [1, 0, 1, 0, 0],    # Thumb + Middle
        'name': 'Eraser Mode'
    },
    'ANALYZE': {
        'fingers': [0, 1, 1, 0, 0],    # Index + Middle
        'name': 'Analysis Mode'
    },
    'RESET': {
        'fingers': [1, 0, 0, 0, 1],    # Thumb + Pinky
        'name': 'Reset Canvas'
    },
    'DISABLE': {
        'fingers': [1, 1, 1, 0, 0],    # Thumb + Index + Middle
        'name': 'Disable Drawing'
    }
}

# Analysis Configuration
ANALYSIS_PROMPT = """
Analyze the image and provide:
- Type of content (equation, text, shapes, artwork)
- For equations: Show equation and solution
- For text: Extract and explain content
- For shapes: Identify and describe relationships
- For artwork: Describe main elements
"""