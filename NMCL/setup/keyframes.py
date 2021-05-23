import cv2
import numpy as np
import sys

global_frame_count = 1

numberOfImages = int(sys.argv[1])

for i in range(1, numberOfImages + 1):
    video_path = f"../data/video/video{str(i).zfill(4)}.mp4"
    p_frame_thresh = 50000  # You may need to adjust this threshold

    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened() == False):
        print("Error opening video file")
    print("zone start")

    # Read the first frame.
    ret, prev_frame = cap.read()

    count_P_frames = 0
    count_frames = 0

    while ret:
        ret, curr_frame = cap.read()
        count_frames += 1
        if ret:
            gray_curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            gray_prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray_curr_frame, gray_prev_frame)
            non_zero_count = np.count_nonzero(diff)

            if non_zero_count > p_frame_thresh:
                count_P_frames += 1
                # print("Got P-Frame")
            else:
                imageName = "../data/images/image{}.jpg".format(
                    str(global_frame_count).zfill(6))
                global_frame_count += 1
                # cv2.imshow(imageName, curr_frame)
                cv2.imwrite(imageName, curr_frame)
            prev_frame = curr_frame

    print("Number of PFrames: {} out of {} frames".format(
        count_P_frames, count_frames))

    print("zone over")
