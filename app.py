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
import time  # For timing metrics
from config import *

filterwarnings('ignore')


class VirtualCalculator:
    def __init__(self):
        load_dotenv()
        self._setup_components()
        # For gesture edge detection to prevent continuous ANALYZE triggering
        self.analysis_gesture_prev = False

        # Metrics for performance and usage
        self.metrics = {
            'frame_time_ms': 0,
            'fps': 0,
            'analysis_time_ms': None,
        }
        self.total_frames = 0
        self.total_frame_time_ms = 0.0
        self.min_frame_time_ms = float('inf')
        self.max_frame_time_ms = 0.0
        self.total_hand_detection_time_ms = 0.0
        self.total_gesture_handling_time_ms = 0.0
        self.total_canvas_blending_time_ms = 0.0
        self.analyses_triggered = 0
        self.analysis_errors = 0
        self.draw_strokes_count = 0
        self.erase_strokes_count = 0

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

            # Determine finger states if landmarks exist
            if self.landmark_list:
                # Thumb: compare x coordinates
                self.fingers.append(int(self.landmark_list[4][1] < self.landmark_list[3][1]))
                # Other fingers: compare y coordinates for tip vs. pip joint
                for tip in [8, 12, 16, 20]:
                    self.fingers.append(int(self.landmark_list[tip][2] < self.landmark_list[tip - 2][2]))

    def handle_gestures(self):
        if not self.landmark_list:
            self.current_mode = "No hand detected"
            self.analysis_gesture_prev = False
            return

        # DRAW mode
        if self.check_gesture('DRAW'):
            self.current_mode = GESTURE_CONFIG['DRAW']['name']
            cx, cy = self.landmark_list[8][1:3]
            if self.prev_point == (0, 0):
                self.prev_point = (cx, cy)
            cv2.line(self.canvas, self.prev_point, (cx, cy),
                     DRAWING_CONFIG['DRAW_COLOR'],
                     DRAWING_CONFIG['DRAW_THICKNESS'])
            self.prev_point = (cx, cy)
            self.draw_strokes_count += 1
            self.analysis_gesture_prev = False

        # ERASE mode
        elif self.check_gesture('ERASE'):
            self.current_mode = GESTURE_CONFIG['ERASE']['name']
            cx, cy = self.landmark_list[12][1:3]
            if self.prev_point == (0, 0):
                self.prev_point = (cx, cy)
            cv2.line(self.canvas, self.prev_point, (cx, cy),
                     DRAWING_CONFIG['ERASE_COLOR'],
                     DRAWING_CONFIG['ERASE_THICKNESS'])
            self.prev_point = (cx, cy)
            self.erase_strokes_count += 1
            self.analysis_gesture_prev = False

        # RESET canvas
        elif self.check_gesture('RESET'):
            self.current_mode = GESTURE_CONFIG['RESET']['name']
            self.canvas = np.zeros_like(self.canvas)
            self.analysis_gesture_prev = False

        # ANALYZE drawing: trigger on rising edge only
        elif self.check_gesture('ANALYZE'):
            self.current_mode = GESTURE_CONFIG['ANALYZE']['name']
            if not self.analysis_gesture_prev and not self.is_analyzing:
                self.analyses_triggered += 1
                self.is_analyzing = True
                self.status_queue.put("Starting analysis...")
                thread = threading.Thread(target=self._analyze_drawing)
                thread.daemon = True
                thread.start()
            self.analysis_gesture_prev = True

        # DISABLE drawing
        elif self.check_gesture('DISABLE'):
            self.current_mode = GESTURE_CONFIG['DISABLE']['name']
            self.prev_point = (0, 0)
            self.analysis_gesture_prev = False

        else:
            self.current_mode = "Ready"
            self.prev_point = (0, 0)
            self.analysis_gesture_prev = False

    def _analyze_drawing(self):
        try:
            self.status_queue.put("Converting image...")
            img_rgb = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB)
            img_pil = PIL.Image.fromarray(img_rgb)

            self.status_queue.put("Initializing AI model...")
            genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
            model = genai.GenerativeModel('gemini-2.0-flash')

            self.status_queue.put("Analyzing content...")
            analysis_start = time.time()
            response = model.generate_content([ANALYSIS_PROMPT, img_pil])
            analysis_end = time.time()
            self.metrics['analysis_time_ms'] = (analysis_end - analysis_start) * 1000

            self.result_queue.put(response.text)
            self.status_queue.put("Analysis complete!")

        except Exception as e:
            self.analysis_errors += 1
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
            st.markdown('<h5 style="color:green;">METRICS</h5>', unsafe_allow_html=True)
            metrics_placeholder = st.empty()

        while True:
            frame_start_time = time.time()
            self.total_frames += 1

            success, frame = self.cap.read()
            if not success:
                st.error("Cannot access webcam")
                break

            # Resize and flip frame for mirror effect
            frame = cv2.resize(frame, (WINDOW_CONFIG['WIDTH'], WINDOW_CONFIG['HEIGHT']))
            frame = cv2.flip(frame, 1)

            # Measure hand detection time
            start_hand = time.time()
            self.process_hands(frame)
            end_hand = time.time()
            hand_detection_time_ms = (end_hand - start_hand) * 1000
            self.total_hand_detection_time_ms += hand_detection_time_ms

            # Measure gesture handling time
            start_gesture = time.time()
            self.handle_gestures()
            end_gesture = time.time()
            gesture_handling_time_ms = (end_gesture - start_gesture) * 1000
            self.total_gesture_handling_time_ms += gesture_handling_time_ms

            # Blend canvas and measure blending time
            start_blend = time.time()
            frame = self.blend_canvas(frame)
            end_blend = time.time()
            canvas_blending_time_ms = (end_blend - start_blend) * 1000
            self.total_canvas_blending_time_ms += canvas_blending_time_ms

            # Calculate total frame time
            frame_end_time = time.time()
            frame_time_ms = (frame_end_time - frame_start_time) * 1000
            self.metrics['frame_time_ms'] = frame_time_ms
            self.total_frame_time_ms += frame_time_ms

            # Update min and max frame time
            if frame_time_ms < self.min_frame_time_ms:
                self.min_frame_time_ms = frame_time_ms
            if frame_time_ms > self.max_frame_time_ms:
                self.max_frame_time_ms = frame_time_ms

            fps = 1000.0 / frame_time_ms if frame_time_ms > 0 else 0
            self.metrics['fps'] = fps

            # Compute average times
            avg_frame_time = self.total_frame_time_ms / self.total_frames
            avg_hand_detection = self.total_hand_detection_time_ms / self.total_frames
            avg_gesture_handling = self.total_gesture_handling_time_ms / self.total_frames
            avg_canvas_blending = self.total_canvas_blending_time_ms / self.total_frames

            # Update status with current mode and queued messages
            status_text = f"Current Mode: {self.current_mode}"
            while not self.status_queue.empty():
                status_text += f"\n{self.status_queue.get()}"
            mode_indicator.markdown(f"```{status_text}```")

            # Update result if available
            if not self.result_queue.empty():
                result = self.result_queue.get()
                result_placeholder.write(f"Result: {result}")

            # Prepare detailed metrics text
            metrics_text = (
                f"Frame Time: {frame_time_ms:.2f} ms  \n"
                f"FPS: {fps:.2f}  \n"
                f"Total Frames: {self.total_frames}  \n"
                f"Avg Frame Time: {avg_frame_time:.2f} ms  \n"
                f"Min Frame Time: {self.min_frame_time_ms:.2f} ms  \n"
                f"Max Frame Time: {self.max_frame_time_ms:.2f} ms  \n"
                f"Hand Detection Time: {hand_detection_time_ms:.2f} ms (Avg: {avg_hand_detection:.2f} ms)  \n"
                f"Gesture Handling Time: {gesture_handling_time_ms:.2f} ms (Avg: {avg_gesture_handling:.2f} ms)  \n"
                f"Canvas Blending Time: {canvas_blending_time_ms:.2f} ms (Avg: {avg_canvas_blending:.2f} ms)  \n"
                f"Analyses Triggered: {self.analyses_triggered}  \n"
                f"Analysis Errors: {self.analysis_errors}  \n"
                f"Draw Strokes: {self.draw_strokes_count}  \n"
                f"Eraser Strokes: {self.erase_strokes_count}  \n"
            )
            if self.metrics['analysis_time_ms'] is not None:
                metrics_text += f"Latest Analysis Time: {self.metrics['analysis_time_ms']:.2f} ms"

            metrics_placeholder.markdown(f"```{metrics_text}```")

            # Display the updated frame
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
