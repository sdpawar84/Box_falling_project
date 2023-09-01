import os

import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMAGE_HEIGHT, IMAGE_WIDTH = 320, 420
CLASSES_LIST = ['object_falling', 'normal']
model = load_model(
    "/home/inqedge/Desktop/satyam/Projects/Object_Falling_Recognition/convlstm_model___Date_Time_2023_07_19__15_40_17___Loss_0.349590003490448___Accuracy_0.9166666865348816.h5")
fw = open("Result_Obj_fall.txt", "w")

"""Create a Function To Perform Action Recognition on Videos"""


def predict_single_action(video_file_path, SEQUENCE_LENGTH):
    '''
    This function will perform single action recognition prediction on a video using the LRCN model.
    Args:
    video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
    SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
    '''

    # Initialize the VideoCapture object to read from the video file.
    video_reader = cv2.VideoCapture(video_file_path)

    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Declare a list to store video frames we will extract.
    frames_list = []

    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    # Get the number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)

    # Iterating the number of times equal to the fixed length of sequence.
    for frame_counter in range(SEQUENCE_LENGTH):

        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        # Read a frame.
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

    # Passing the  pre-processed frames to the model and get the predicted probabilities.
    predicted_labels_probabilities = model.predict(np.expand_dims(frames_list, axis=0))[0]

    # Get the index of class with highest probability.
    predicted_label = np.argmax(predicted_labels_probabilities)

    # Get the class name using the retrieved index.
    predicted_class_name = CLASSES_LIST[predicted_label]

    # Display the predicted action along with the prediction confidence.
    print(
        f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]} File name{os.path.basename(video_file_path)}',
        file=fw)

    # Release the VideoCapture object.
    video_reader.release()


if __name__ == "__main__":
    """Perform Single Prediction on a Test Video"""
    fol_path = "/home/inqedge/Desktop/satyam/Projects/Object_Falling_Recognition/dataset/media/object_falling"
    fr = open("/home/inqedge/Desktop/satyam/Projects/Object_Falling_Recognition/object_falling.txt", "r")
    for file in fr:
        video_file_path = os.path.join(fol_path, file.strip())
        SEQUENCE_LENGTH = 20
        predict_single_action(video_file_path, SEQUENCE_LENGTH)

# # Download the youtube video.
# video_title = download_youtube_videos('https://youtu.be/fc3w827kwyA', test_videos_directory)
#
# # Construct tihe nput youtube video path
# input_video_file_path = f'{test_videos_directory}/{video_title}.mp4'
#
# # Perform Single Prediction on the Test Video.
# predict_single_action(input_video_file_path, SEQUENCE_LENGTH)
#
# # Display the input video.
# VideoFileClip(input_video_file_path, audio=False, target_resolution=(300,None)).ipython_display()
