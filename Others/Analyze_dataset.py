import os
import cv2

FRAME_WIDTH_LIST = []
FRAME_HEIGHT_LIST = []
FRAMES_PER_SECOND_LIST = []
total_frame_list = []
folder_path = "/home/inqedge/Desktop/satyam/Projects/AutoVideo/media-20230614T032819Z-001/media/"
# folder_path = "/home/inqedge/Downloads/UCF11_updated_mpg/UCF11_updated_mpg/basketball/v_shooting_01/"
def Average(lst):
    return sum(lst) / len(lst)

file_list = [i for i in os.listdir(folder_path) if "Object" in i]
# file_list = [i for i in os.listdir(folder_path)]
print("No of files", len(file_list))

for f in file_list:
    src = os.path.join(folder_path, f)
    cap = cv2.VideoCapture(src)
    ret, frame = cap.read()
    ### To read properties of frame
    """
    Check this link for more properties
    https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
    """
    FRAME_WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    FRAME_HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    FRAME_WIDTH_LIST.append(FRAME_WIDTH)
    FRAME_HEIGHT_LIST.append(FRAME_HEIGHT)

    ## WHEN VIDEO FILE PATH PROVIDED
    FPS = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    FRAMES_PER_SECOND_LIST.append(FPS)
    total_frame_list.append(total_frames)

    if FRAME_WIDTH < 420: print(f)

    cap.release()

print(total_frame_list)
print("Average Frame Width", Average(FRAME_WIDTH_LIST))
print("Average Frame Height", Average(FRAME_HEIGHT_LIST))
print("Average FPS", Average(FRAMES_PER_SECOND_LIST))
print("Average Total Frames", Average(total_frame_list))

print(min(FRAME_WIDTH_LIST))
print(min(FRAME_HEIGHT_LIST))
