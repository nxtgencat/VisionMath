import os
import cv2
import PIL
import numpy as np
import google.generativeai as genai
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from mediapipe.python.solutions import hands, drawing_utils
from dotenv import load_dotenv
from warnings import filterwarnings
import threading
from queue import Queue
from config import *

filterwarnings('ignore')


class VirtualCalculator:
    def __init__(self):
        load_dotenv()
        self._setup_components()

    def _setup_components(self):
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_CONFIG['WIDTH'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_CONFIG['HEIGHT'])

        # Initialize other components
        self.canvas = np.zeros((WINDOW_CONFIG['HEIGHT'], WINDOW_CONFIG['WIDTH'], 3), dtype=np.uint8)
        self.hand_detector = hands.Hands(max_num_hands=1, min_detection_confidence=0.75)
        self.landmark_list = []
        self.fingers = []
        self.prev_point = (0, 0)

        # Status tracking
        self.current_mode = "Ready"
        self.is_analyzing = False
        self.result_queue = Queue()
        self.status_queue = Queue()

    @staticmethod
    def setup_page():
        st.set_page_config(page_title='Calculator', layout="wide")
        st.markdown("""
            <style>
            [data-testid="stHeader"] { background: rgba(0,0,0,0); }
            .block-container { padding-top: 0rem; }
            </style>
            <h1 style="text-align: center;">Virtual Calculator</h1>
        """, unsafe_allow_html=True)
        add_vertical_space(1)

    def check_gesture(self, gesture_name):
        if not self.fingers:
            return False
        return self.fingers == GESTURE_CONFIG[gesture_name]['fingers']

    def process_hands(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hand_detector.process(frame_rgb)
        self.landmark_list = []
        self.fingers = []

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                drawing_utils.draw_landmarks(frame, hand, hands.HAND_CONNECTIONS)

                # Get landmarks
                for idx, lm in enumerate(hand.landmark):
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.landmark_list.append([idx, cx, cy])

            # Check finger positions
            if self.landmark_list:
                # Thumb
                self.fingers.append(int(self.landmark_list[4][1] < self.landmark_list[3][1]))
                # Other fingers
                for tip in [8, 12, 16, 20]:
                    self.fingers.append(int(self.landmark_list[tip][2] < self.landmark_list[tip - 2][2]))

    def handle_gestures(self):
        if not self.landmark_list:
            self.current_mode = "No hand detected"
            return

        # Drawing mode
        if self.check_gesture('DRAW'):
            self.current_mode = GESTURE_CONFIG['DRAW']['name']
            cx, cy = self.landmark_list[8][1:3]
            if self.prev_point == (0, 0):
                self.prev_point = (cx, cy)
            cv2.line(self.canvas, self.prev_point, (cx, cy),
                     DRAWING_CONFIG['DRAW_COLOR'],
                     DRAWING_CONFIG['DRAW_THICKNESS'])
            self.prev_point = (cx, cy)

        # Eraser mode
        elif self.check_gesture('ERASE'):
            self.current_mode = GESTURE_CONFIG['ERASE']['name']
            cx, cy = self.landmark_list[12][1:3]
            if self.prev_point == (0, 0):
                self.prev_point = (cx, cy)
            cv2.line(self.canvas, self.prev_point, (cx, cy),
                     DRAWING_CONFIG['ERASE_COLOR'],
                     DRAWING_CONFIG['ERASE_THICKNESS'])
            self.prev_point = (cx, cy)

        # Reset canvas
        elif self.check_gesture('RESET'):
            self.current_mode = GESTURE_CONFIG['RESET']['name']
            self.canvas = np.zeros_like(self.canvas)

        # Analyze drawing
        elif self.check_gesture('ANALYZE') and not self.is_analyzing:
            self.current_mode = GESTURE_CONFIG['ANALYZE']['name']
            self.is_analyzing = True
            self.status_queue.put("Starting analysis...")
            thread = threading.Thread(target=self._analyze_drawing)
            thread.daemon = True
            thread.start()

        # Disable drawing
        elif self.check_gesture('DISABLE'):
            self.current_mode = GESTURE_CONFIG['DISABLE']['name']
            self.prev_point = (0, 0)

        else:
            self.current_mode = "Ready"
            self.prev_point = (0, 0)

    def _analyze_drawing(self):
        try:
            self.status_queue.put("Converting image...")
            img_rgb = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB)
            img_pil = PIL.Image.fromarray(img_rgb)

            self.status_queue.put("Initializing AI model...")
            genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
            model = genai.GenerativeModel('gemini-1.5-flash')

            self.status_queue.put("Analyzing content...")
            response = model.generate_content([ANALYSIS_PROMPT, img_pil])
            self.result_queue.put(response.text)
            self.status_queue.put("Analysis complete!")

        except Exception as e:
            self.result_queue.put(f"Analysis error: {str(e)}")
            self.status_queue.put("Analysis failed!")
        finally:
            self.is_analyzing = False

    def blend_canvas(self, frame):
        canvas_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, canvas_mask = cv2.threshold(canvas_gray, 50, 255, cv2.THRESH_BINARY_INV)
        canvas_mask = cv2.cvtColor(canvas_mask, cv2.COLOR_GRAY2BGR)

        frame = cv2.addWeighted(frame, 0.7, self.canvas, 1, 0)
        frame = cv2.bitwise_and(frame, canvas_mask)
        return cv2.bitwise_or(frame, self.canvas)

    def run(self):
        # Setup Streamlit layout
        col1, _, col3 = st.columns([0.8, 0.02, 0.18])
        frame_placeholder = col1.empty()

        with col3:
            st.markdown('<h5 style="color:green;">STATUS</h5>', unsafe_allow_html=True)
            mode_indicator = st.empty()
            st.markdown('<h5 style="color:green;">OUTPUT</h5>', unsafe_allow_html=True)
            result_placeholder = st.empty()

        while True:
            success, frame = self.cap.read()
            if not success:
                st.error("Cannot access webcam")
                break

            # Process frame
            frame = cv2.resize(frame, (WINDOW_CONFIG['WIDTH'], WINDOW_CONFIG['HEIGHT']))
            frame = cv2.flip(frame, 1)

            # Process hands and gestures
            self.process_hands(frame)
            self.handle_gestures()

            # Update status
            status_text = f"Current Mode: {self.current_mode}"
            if self.is_analyzing:
                while not self.status_queue.empty():
                    status_text += f"\n{self.status_queue.get()}"
            mode_indicator.markdown(f"```{status_text}```")

            # Update result
            if not self.result_queue.empty():
                result = self.result_queue.get()
                result_placeholder.write(f"Result: {result}")

            # Display frame
            frame = self.blend_canvas(frame)
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    try:
        calc = VirtualCalculator()
        calc.setup_page()
        calc.run()
    except Exception as e:
        st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()