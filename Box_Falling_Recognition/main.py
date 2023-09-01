import logging
import sys
from config import *
from Camera import Camera

# log_filename = '/app/images/Box_falling.log'
# log_filename = 'Box_falling.log'
# sys.stdout = open(log_filename, 'a')
# sys.stderr = sys.stdout
#
# FORMAT = "\n\n %(asctime)s -- %(name)s -- %(funcName)s --  %(message)s"
# logging.basicConfig(stream=sys.stderr, format=FORMAT, datefmt="%d-%b-%y %H:%M:%S", level=eval(logging_level))


def main():
    cam_url = "/home/inqedge/Desktop/satyam/Projects/Object_Falling_Recognition/dataset/media/normal/Normal_045_43.avi"
    cam_obj = Camera(cam_url=cam_url)
    cam_obj.monitor_stream()

if __name__ == "__main__":
    main()