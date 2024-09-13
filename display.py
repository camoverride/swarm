from collections import deque
import os

import cv2
import face_recognition
import mediapipe as mp
import numpy as np
from picamera2 import Picamera2
import yaml

from image_processing_utils import crop_face_with_margin, get_triangulation_indexes_for_basis_image, get_face_landmarks, get_additional_landmarks, morph




def processing_pipeline(image,
                        margin : float,
                        total_face_width : int,
                        total_face_height : int,
                        target_triangulation_indexes,
                        target_all_landmarks) -> np.ndarray:
    """
    Morph-aligns the input `image` to match the target, with the desired margin and dimensions.
    """
    # Convert to RGB.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Check if the face mesh can be applied/
    results_mesh = face_mesh.process(image_rgb)

    # If there are landmarks, continue.
    if results_mesh.multi_face_landmarks:

        # Detect the location of the face.
        results = mp_face_detection.process(image_rgb)

        # Use only the first detection.
        detection = results.detections[0]

        # Crop the face to the bounding box.
        bounding_box = detection.location_data.relative_bounding_box
        cropped_face = crop_face_with_margin(image, bounding_box, margin)

        # Resize the face to the desired dimensions.
        resized_face = cv2.resize(cropped_face, (total_face_height, total_face_width), interpolation=cv2.INTER_AREA)

        # Get the face landmarks and additional landmarks
        source_face_landmarks = get_face_landmarks(resized_face)

        # Only append the face if it exists!
        if source_face_landmarks:

            # Get all the landmarks.
            source_additional_landmarks = get_additional_landmarks(resized_face)
            source_all_landmarks = source_face_landmarks + source_additional_landmarks

            # Morph the source face onto the target face.
            morphed_face = morph(target_face, resized_face, target_all_landmarks, source_all_landmarks, target_triangulation_indexes)

            return morphed_face
    
    else:
        return None
            

def is_face_looking_forward(face_image):
    """
    Returns true if the face is forward.
    """
    # Process the image.
    results = face_mesh.process(face_image)

    # Convert back to the right colors.
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

    # Get image size information.
    img_h , img_w, img_c = image.shape

    # Collect the 2D and 3D landmarks.
    face_2d = []
    face_3d = []

    # If there are landmarks (face detected), continue
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx ==1 or idx == 61 or idx == 291 or idx==199:
                    if idx ==1:
                        nose_2d = (lm.x * img_w,lm.y * img_h)
                        nose_3d = (lm.x * img_w,lm.y * img_h,lm.z * 3000)
                    x,y = int(lm.x * img_w),int(lm.y * img_h)

                    face_2d.append([x,y])
                    face_3d.append(([x,y,lm.z]))


            # Get 2D coordinates
            face_2d = np.array(face_2d, dtype=np.float64)

            # Get 3D coordinates
            face_3d = np.array(face_3d,dtype=np.float64)

            # Calculate the orientation of the face.
            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length,0,img_h/2],
                                [0,focal_length,img_w/2],
                                [0,0,1]])
            distortion_matrix = np.zeros((4,1),dtype=np.float64)

            success, rotation_vec, translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)

            # Get the rotational vector of the face.
            rmat, jac = cv2.Rodrigues(rotation_vec)

            angles, mtxR, mtxQ ,Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # Check which way the face is oriented.
            # Previously -3 3 -3 7
            if y < -5: # Looking Left
                return False
            elif y > 5: # Looking Right
                return False
            elif x < -5: # Looking Down
                return False
            elif x > 7: # Looking Up
                return False
            else: # Looking Forward
                return True
    
    # No face was detected!
    else:
        return None



if __name__ == "__main__":
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
    face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5)

    # Set the display
    os.environ["DISPLAY"] = ':0'

    # Rotate the screen
    os.system("WAYLAND_DISPLAY=wayland-1 wlr-randr --output HDMI-A-1 --transform 270")

    # Hide the cursor
    os.system("unclutter -idle 0 &")

    # Load the config.yaml file
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Hold the most recent N faces, so they can't be repeated too often.
    known_face_encodings = deque(maxlen=config["face_memory"])

    # Get triangulation indexes for the target image
    target_triangulation_indexes, target_all_landmarks, target_face \
                                    = get_triangulation_indexes_for_basis_image(
                                            basis_image_path=config["basis_image"],
                                            total_face_width=config["display_width"],
                                            total_face_height=config["display_height"],
                                            margin=config["margin"],
                                            debug=False)

    # The current average face, which will be displayed on the screen.
    CURRENT_AVERAGE = []

    # Start video capture.
    picam2 = Picamera2()

    # Set the capture size to twice the width/height of the image.
    picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888",
                                                               "size": (config["display_width"] * 2, 
                                                                        config["display_height"] * 2)}))
    picam2.start()

    # Make the display fullscreen
    cv2.namedWindow("Running Average", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Running Average", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


    # Main event loop
    while True:

        # Get a picture from the webcam.
        frame = picam2.capture_array()

        # Reduce the frame size to speed up face recognition.
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Convert the image from BGR (OpenCV format) to RGB (face_recognition format)
        rgb_small_frame = np.ascontiguousarray(frame[:, :, ::-1])

        # Detect faces and get encodings for the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # If there is a face encoding, continue.
        if len(face_encodings) > 0:

            # Iterate over the encodings.
            for i, face_encoding in enumerate(face_encodings):

                print(f"{len(face_encodings)} faces detected!")

                # Check if this face has already been seen
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=config["tolerance"])

                # If no match, it means this is a new face
                if not any(matches):
                    print(f"  Analyzing Face {i}")

                    # Flip the frame for selfie view.
                    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

                    # Check if the face is forward
                    looking_forward = is_face_looking_forward(face_image=image)

                    # Convert back to the right colors.
                    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

                    # If the face is looking forward (passport style), continue.
                    if looking_forward:

                        print("    Face is looking forward!")

                        # Morph-align the face to match the target
                        new_morph = processing_pipeline(image=image,
                                                        total_face_width=config["display_width"],
                                                        total_face_height=config["display_height"],
                                                        margin=config["margin"],
                                                        target_triangulation_indexes=target_triangulation_indexes,
                                                        target_all_landmarks=target_all_landmarks)

                        # Sometimes morphing is unsuccessful and returns None
                        if new_morph is not None:

                            print("      Morphed the face")

                            # Alpha blend the image with the previous image.
                            if len(known_face_encodings) == 0:
                                alpha = 0.95
                            elif len(known_face_encodings) < 2:
                                alpha = 0.85
                            elif len(known_face_encodings) < 3:
                                alpha = 0.5
                            elif len(known_face_encodings) < 5:
                                alpha = 0.3
                            else:
                                alpha = config["alpha"]

                            # This should only trigger on the first loop.
                            if CURRENT_AVERAGE == []:
                                CURRENT_AVERAGE = new_morph

                            # Beta is the weight for image 2
                            beta = 1.0 - config["alpha"]

                            # Perform alpha blending
                            blended_image = cv2.addWeighted(new_morph, config["alpha"], CURRENT_AVERAGE, beta, 0)

                            # Display the blended image!
                            cv2.imshow("Running Average", blended_image)

                            # Set the current face to the blended face.
                            CURRENT_AVERAGE = blended_image

                            # Add the new face encoding to the list of known faces
                            known_face_encodings.append(face_encoding)

        print("---------------")

        # Exit if 'q' is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
