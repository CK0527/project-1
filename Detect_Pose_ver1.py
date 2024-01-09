from kivy.app import App
from kivy.core.image import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.clock import Clock
import cv2
import mediapipe as mp
import numpy as np


class PoseDetectorApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')

        self.rep_count_label = Label(text="Rep Count: 0")
        self.stage_label = Label(text="Stage: ")
        self.web_cam = Image(size_hint=(1, 15))

        layout.add_widget(self.rep_count_label)
        layout.add_widget(self.stage_label)
        layout.add_widget(self.web_cam)

        self.capture = cv2.VideoCapture(0)

        # Clock.schedule_once(self.start_pose_detection)
        Clock.schedule_interval(self.start_pose_detection, 1.0 / 30.0)

        return layout

    def start_pose_detection(self, dt=None):
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        cap = self.capture

        counter = 0
        stage = None

        def calculate_angle(a, b, c):
            a = np.array(a)  # First
            b = np.array(b)  # Mid
            c = np.array(c)  # End

            radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            angle = np.abs(radians * 180.0 / np.pi)

            if angle > 180.0:
                angle = 360 - angle

            return angle


        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            def process_frame(dt):
                nonlocal counter, stage

                ret, frame = cap.read()



                if ret:
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False

                    try:
                        results = pose.process(image)

                        if results is not None and results.pose_landmarks:  # Check if pose landmarks are detected
                            landmarks = results.pose_landmarks.landmark

                            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

                            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

                            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

                            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]


                            left_angle1 = calculate_angle(left_shoulder, left_elbow, left_wrist)
                            left_angle2 = calculate_angle(left_hip, left_shoulder, left_elbow)
                            right_angle1 = calculate_angle(right_shoulder, right_elbow, right_wrist)
                            right_angle2 = calculate_angle(right_hip, right_shoulder, right_elbow)

                            # Rest of your code for pose estimation, angle calculation, and rep counting
                            # Visualize angle
                            cv2.putText(image, str(left_angle1),
                                        tuple(np.multiply(left_elbow, [640, 480]).astype(int)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )

                            cv2.putText(image, str(left_angle2),
                                        tuple(np.multiply(left_shoulder, [640, 480]).astype(int)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )

                            cv2.putText(image, str(right_angle1),
                                        tuple(np.multiply(right_elbow, [640, 480]).astype(int)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )

                            cv2.putText(image, str(right_angle2),
                                        tuple(np.multiply(right_shoulder, [640, 480]).astype(int)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )


                            if left_angle1 > 150 and left_angle2 > 8:
                                stage = "left down"
                            if left_angle1 < 100 and left_angle2 < 8 and stage == 'left down':
                                stage = 'left up'
                                counter += 1

                            first_landmark = landmarks[0]

                            # Update the UI labels with rep count and stage information
                            self.rep_count_label.text = f"Rep Count: {counter}"
                            self.stage_label.text = f"Stage: {stage}"

                            buf = cv2.flip(frame, 0).tostring()
                            img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]))
                            img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                            self.web_cam.texture = img_texture

                        else:
                            print("No pose landmarks detected.")

                    except Exception as e:
                        print(f"Error processing pose: {e}")
                else:
                    print("No frame captured from the camera.")

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Display the image with landmarks
                cv2.imshow('AI Martial Arts', image)

            Clock.schedule_interval(process_frame, 1.0 / 30.0)  # Adjust frame processing rate here

        def close_cv2_and_cleanup(dt):
            cap.release()
            cv2.destroyAllWindows()

        # Schedule the cleanup function when the app is closed
        self.bind(on_stop=close_cv2_and_cleanup)

if __name__ == '__main__':
    PoseDetectorApp().run()
