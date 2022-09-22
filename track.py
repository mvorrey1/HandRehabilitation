#!python
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from hand_tracker import HandTracker
from scipy.signal import savgol_filter
import cv2
import numpy as np
import pickle
import sys
import glob
import os


palm_model_path = "./models/palm_detection.tflite"
landmark_model_path = "./models/hand_landmark.tflite"
anchors_path = "./data/anchors.csv"

detector = HandTracker(palm_model_path, landmark_model_path, anchors_path)


def write_kps(infile, outfile):
    cap = cv2.VideoCapture(infile)
    data = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_counter = 0
    while True:
        res, frame = cap.read()

        if not res:
            if frame_counter == total_frames:
                print('finished, writing output to', outfile)
                pickle.dump(data, open(outfile, 'wb'))
            else:
                print('failed to read all frames')
                cap.release()
                return

            break

        printProgressBar(frame_counter, total_frames)
        print('processed', frame_counter, '/', total_frames, 'frames')

        frame_counter += 1
        try: 
            kp, box = detector(frame)
            data.append(kp)
        except ValueError as e: 
            print('failed to detect hand in frame')
            # skip value


def draw_pts(frame, pts, color=(0, 255, 0), weight=2):
    for p in pts:
        cv2.circle(frame, (int(p[0]), int(p[1])), 3, color, thickness=weight)


def visRT(vidfile, posefile):
    print(vidfile)
    cap = cv2.VideoCapture(vidfile)
    data = np.array(pickle.load(open(posefile, 'rb')))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for i in range(total_frames):
        res, frame = cap.read()

        try: 
            kp,box = detector(frame)
            draw_pts(frame, kp, color=(0,255,0), weight=5)
        except ValueError as ve: 
            print(f'frame {i}: failed to get hand pose')
        
        draw_pts(frame, data[i], color=(0,0,255), weight=2)

        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        cv2.imshow('real time', frame)
        k = cv2.waitKey(1)
        if k == 27:
            break
        elif k == ord(' '):
            cv2.waitKey(0)    

def visualize(vidfile, posefile):
    print(vidfile)
    cap = cv2.VideoCapture(vidfile)
    data = np.array(pickle.load(open(posefile, 'rb')))
    smoothed = smooth(data)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    canvas = None
    for i in range(total_frames):
        res, frame = cap.read()
        draw_pts(frame, data[i], color=(0,0,255), weight=5)
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)


        cv2.imshow('viz', frame)
        k = cv2.waitKey(1)
        if k == 27:
            break
        elif k == ord(' '):
            cv2.waitKey(0)


"""Apply Savgol Filter"""


def smooth(data, m=9, n=3):
    filtered = data.copy()
    # print(filtered.shape)
    for i in range(21):
        filtered[:, i, 0] = savgol_filter(data[:, i, 0], m, n)
        filtered[:, i, 1] = savgol_filter(data[:, i, 1], m, n)
    return filtered


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


v = None

def onScrubChange(new_value):
    v.set(cv2.CAP_PROP_POS_FRAMES,new_value)
    err,img = v.read()
    
    try: 
        kp, box = detector(img)
        draw_pts(img, kp)
    except ValueError as ve: 
        print(f'frame {new_value}: failed to get hand pose')

    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("display", img)
    cv2.waitKey()

def test(vidfile):
    global v
    v = cv2.VideoCapture(vidfile)
    frame_count = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
    cv2.namedWindow('display')
    cv2.createTrackbar('frame', 'display', 0, frame_count, onScrubChange)
    onScrubChange(0)

def main():

    ex = ['cap']
    paths = ['worst']

    for e in ex:
        for p in paths:
            folder = f'./videos/{e}/{p}/'
            
            for i,v in enumerate(glob.glob(folder + '*')): 
                visualize(vidfile=v,posefile=f'./poses/{e}/{p}/{p}{i}.pose')
    

if __name__ == '__main__':
    main()
