import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMAGE_HEIGHT, IMAGE_WIDTH = 320, 420
CLASSES_LIST = ['object_falling', 'normal']
model = load_model(
    "/home/inqedge/Desktop/satyam/Projects/Object_Falling_Recognition/convlstm_model___Date_Time_2023_07_19__15_40_17___Loss_0.349590003490448___Accuracy_0.9166666865348816.h5")
fw = open("Result_Obj_fall.txt", "w")

"""Create a Function To Perform Action Recognition on RTSP Stream"""


def predict_single_action_rtsp(rtsp_url, SEQUENCE_LENGTH, SKIP_FRAMES):
    '''
    This function will perform single action recognition prediction on an RTSP stream using the LRCN model.
    Args:
    rtsp_url:       The RTSP stream URL.
    SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
    SKIP_FRAMES:     The number of frames to skip before processing the next sequence.
    '''

    # Initialize the VideoCapture object to read from the RTSP stream.
    video_reader = cv2.VideoCapture(rtsp_url)

    # Declare a list to store video frames we will extract.
    frames_list = []

    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    frame_counter = 0

    while True:
        # Read a frame from the RTSP stream.
        success, frame = video_reader.read()

        # Check if frame is not read properly then break the loop.
        if not success:
            break

        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
        normalized_frame = resized_frame / 255

        # Appending the pre-processed frame into the frames list
        frames_list.append(normalized_frame)

        frame_counter += 1

        # Checking if we have enough frames for prediction.
        if len(frames_list) == SEQUENCE_LENGTH:
            # Passing the pre-processed frames to the model and get the predicted probabilities.
            predicted_labels_probabilities = model.predict(np.expand_dims(frames_list, axis=0))[0]

            # Get the index of class with highest probability.
            predicted_label = np.argmax(predicted_labels_probabilities)

            # Get the class name using the retrieved index.
            predicted_class_name = CLASSES_LIST[predicted_label]

            # Display the predicted action along with the prediction confidence.
            print(
                f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}')

            # Clear the frames_list for the next sequence of frames.
            frames_list.clear()

    # Release the VideoCapture object.
    video_reader.release()


# Call the function with the RTSP stream URL, SEQUENCE_LENGTH, and SKIP_FRAMES
predict_single_action_rtsp("rtsp://your_rtsp_stream_url", 20, 3)
