from collections import deque
import os
import random
from typing import List

import cv2
import face_recognition
import mediapipe as mp
import numpy as np
import yaml

from image_processing_utils import crop_face_with_margin, get_triangulation_indexes_for_basis_image, get_face_landmarks, get_additional_landmarks, morph



# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5)



def create_composite_image(image_list : List[np.ndarray], # TODO: this should be a path to a memmap
                           num_squares_height : int,
                           image_height : int,
                           image_width : int) -> np.ndarray :
    """
    Accepts a list of images and desired number of squares (along the vertical margin)
    and creates a composite image from them.

    Parameters
    ----------
    image_list : List[np.ndarray]
        A list of images encoded as numpy arrays.
    num_squares_height : int,
        The number of squares to tile the vertical of the image.
    image_height : int
        The height of the image, in pixels.
    image_width : int
        The width of the image, in pixels.
    
    Returns
    -------
    np.ndarray
        A composite image encoded as a numpy array.
    """
    # Check that the images all have the same shape
    if len(set((img.shape for img in image_list))) != 1:
        raise ValueError("All images must have the same dimensions.")

    # Set the image dimension info.
    crop_width = image_width - (image_width % num_squares_height)
    crop_height = image_height - (image_height % num_squares_height)
    image_list = [img[:crop_height, :crop_width] for img in image_list]

    square_size = crop_height // num_squares_height
    num_squares_width = crop_width // square_size

    # Generate the individual squares.
    # TODO: this should also be a memmap
    squares = [[[] for _ in range(num_squares_width)] for _ in range(num_squares_height)]
    for img in image_list:
        for i in range(num_squares_height):
            for j in range(num_squares_width):
                top = i * square_size
                left = j * square_size
                square = img[top:top + square_size, left:left + square_size]
                squares[i][j].append(square)

    # Combine the squares into an image.
    composite_image = np.zeros_like(image_list[0][:crop_height, :crop_width])
    for i in range(num_squares_height):
        for j in range(num_squares_width):
            selected_square = random.choice(squares[i][j])
            top = i * square_size
            left = j * square_size
            composite_image[top:top + square_size, left:left + square_size] = selected_square

    return composite_image


def processing_pipeline(image,
                        margin : float,
                        total_face_width : int,
                        total_face_height : int,
                        target_triangulation_indexes,
                        target_all_landmarks
                                                ) -> List[np.ndarray]:
    """
  
    """

    # First check is the face mesh can be applied, as the mesh (coordinate points) will be essential in later steps.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)

    if results_mesh.multi_face_landmarks:
        print("----results_mesh.multi_face_landmarks")
        results = mp_face_detection.process(image)

        # Second, check if there is actually a face (which there should be, give that there is a mesh). However, we also need the bounding box for cropping.
        if 1==1:#results.detections:
            print("----results.detection")
            detection = results.detections[0]
            bounding_box = detection.location_data.relative_bounding_box

            # Crop the face with the desired margin
            cropped_face = crop_face_with_margin(image, bounding_box, margin)



            resized_face = cv2.resize(cropped_face, (total_face_height, total_face_width), interpolation=cv2.INTER_AREA)
    




            # Get the face landmarks and additional landmarks
            # Problem! Getting landmarks on cropped image will often fail!
            source_face_landmarks = get_face_landmarks(resized_face)

            # Only append the face if it exists!
            if source_face_landmarks:
                print("----landmarks")
                source_additional_landmarks = get_additional_landmarks(resized_face)
                source_all_landmarks = source_face_landmarks + source_additional_landmarks

                morphed_face = morph(target_face, resized_face, target_all_landmarks, source_all_landmarks, target_triangulation_indexes)

                return morphed_face



if __name__ == "__main__":
    # Set the display
    os.environ["DISPLAY"] = ':0'

    # Rotate the screen
    os.system("WAYLAND_DISPLAY=wayland-1 wlr-randr --output HDMI-A-1 --transform 270")

    # Hide the cursor
    os.system("unclutter -idle 0 &")

    # Load the config.yaml file
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    total_face_width = config["display_width"]
    total_face_height = config["display_height"]

    # Hold the most recent N faces, so they can't be repeated too often.
    # TODO: check if this makes sense, given processing time on the Pi
    known_face_encodings = deque(maxlen=config["face_memory"])

    # Get triangulation indexes for the target image
    target_triangulation_indexes, target_all_landmarks, target_face \
                                    = get_triangulation_indexes_for_basis_image(
                                            basis_image_path="cam.jpg",
                                            total_face_width=total_face_width,
                                            total_face_height=total_face_height,
                                            margin=2.5,
                                            debug=False)

    # Morph a single face for the starting image.
    morphed_face = processing_pipeline(image=cv2.imread("cam.jpg"),
                                       total_face_width=total_face_width,
                                       total_face_height=total_face_height,
                                       margin=2.5,
                                       target_triangulation_indexes=target_triangulation_indexes,
                                       target_all_landmarks=target_all_landmarks)

    # The current average face, which will be displayed on the screen.
    CURRENT_AVERAGE = morphed_face

    # Track the number of faces previously

    # Start video capture.
    from picamera2 import Picamera2
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
    # picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888",
    #                                                            "size": (config["display_width"] * 2, 
    #                                                                     config["display_height"] * 2)}))
    picam2.start()

    # Make the display fullscreen
    cv2.namedWindow("Running Average", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Running Average", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


    while True:
        # Set this to False. If a face is looking forward, it will be set to True later.
        looking_forward = False

        # Get a picture from the webcam.
        frame = picam2.capture_array()

        # Face recognition:
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Convert the image from BGR (OpenCV format) to RGB (face_recognition format)
        rgb_small_frame = np.ascontiguousarray(frame[:, :, ::-1])

        # Detect faces and get encodings for the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        if len(face_encodings) > 0:
            # Iterate over the encodings.
            for i, face_encoding in enumerate(face_encodings):
                print(f"{len(face_encodings)} faces detected!")
                # Check if this face has already been seen
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=config["tolerance"])

                # If no match, it means this is a new face
                if any(matches):
                    print("  Face seen before!")
                
                else:
                    print(f"  Analyzing Face {i}")
                    # This image will be used for averaging
                    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

                    # Process the image.
                    image.flags.writeable = False
                    results = face_mesh.process(image)
                    image.flags.writeable = True

                    # TODO: check if this is redundant
                    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

                    # Get image size information.
                    img_h , img_w, img_c = image.shape

                    # Collect the 2D and 3D landmarks.
                    face_2d = []
                    face_3d = []

                    # If there are landmarks (face detected), continue
                    if results.multi_face_landmarks:
                        print("    Landmarks detected")
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

                            success,rotation_vec,translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)

                            # Get the rotational vector of the face.
                            rmat, jac = cv2.Rodrigues(rotation_vec)

                            angles, mtxR, mtxQ ,Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                            x = angles[0] * 360
                            y = angles[1] * 360
                            z = angles[2] * 360

                            # Check which way the face is oriented.
                            # Previously -3 3 -3 7
                            if y < -5: # Looking Left
                                looking_forward=False
                            elif y > 5: # Looking Right
                                looking_forward=False
                            elif x < -5: # Looking Down
                                looking_forward=False
                            elif x > 7: # Looking Up
                                looking_forward=False
                            else: # Looking Forward
                                looking_forward=True

                    else:
                        print("    No landmarks detected")
                
                    # If the face is looking forward (passport style), continue.
                    if looking_forward:
                        print("    Face is looking forward!")
                        try:
                            # Generate the morph.
                            new_morph = processing_pipeline(image=image,
                                                            total_face_width=total_face_width,
                                                            total_face_height=total_face_height,
                                                            margin=2.5,
                                                            target_triangulation_indexes=target_triangulation_indexes,
                                                            target_all_landmarks=target_all_landmarks)

                            if new_morph is not None:
                                print("      Morphed the face")
                                # Alpha blend the image with the previous image.
                                alpha = config["alpha"]

                                if len(known_face_encodings) == 0:
                                    alpha = 0.999
                                elif len(known_face_encodings) < 3:
                                    alpha = 0.5
                                elif len(known_face_encodings) < 5:
                                    alpha = 0.3
                                else:
                                    alpha = config["alpha"]

                                beta = 1.0 - alpha  # Weight for image2

                                # Perform alpha blending
                                blended_image = cv2.addWeighted(new_morph, config["alpha"], CURRENT_AVERAGE, beta, 0)

                                # Set the current face to the blended face.
                                CURRENT_AVERAGE = blended_image

                                # Add the new face encoding to the list of known faces
                                known_face_encodings.append(face_encoding)
                            else:
                                print("      Did not morph/blend the face")
                        except:
                            print("ERROR morphing face")
                    
                    else:
                        print("    Face NOT looking forward")

        else:
            print("No faces detected")

        print("---------------")

        cv2.imshow("Running Average", CURRENT_AVERAGE)
        # Exit if 'q' is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
