import cv2
import mediapipe as mp
import socket
import json
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# set server Ip address and port number
# Ip = "127.0.0.1", Prot = 5051
serverIpPort = ("127.0.0.1", 5051)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # add hand_landmarks--------------------------------
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS)
    # --------------------------------------------------
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,      #FACEMESH_CONTOURS / FACEMESH_TESSELATION
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())


    # collect hand_landmarks
    leftHandLandmarks = []
    if results.left_hand_landmarks:
        for idx, lm in enumerate(results.left_hand_landmarks.landmark):
            leftHandLandmarks.append({
                'index': idx,
                'x': lm.x,
                'y': lm.y,
                'z': lm.z
            })

    rightHandLandmarks = []
    if results.right_hand_landmarks:
        for idx, lm in enumerate(results.right_hand_landmarks.landmark):
            rightHandLandmarks.append({
                'index': idx,
                'x': lm.x,
                'y': lm.y,
                'z': lm.z
            })

    # json file
    data = json.dumps({
        "leftHandLandmarks": leftHandLandmarks,
        "rightHandLandmarks": rightHandLandmarks
    })

    # send data
    sock.sendto(data.encode('utf-8'), serverIpPort)


    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('Mediapipe Holistic', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()