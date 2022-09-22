import cv2
from cv2 import aruco
import numpy as np 
import os,sys
import pickle

CAP = 0#sys.argv[1]
cap = cv2.VideoCapture(CAP)
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()
board = aruco.CharucoBoard_create(7, 5, 1, .8, aruco_dict)
markerLength = 0.023


def load_calib(): 
    with open('calib.pickle','rb') as f: 
        data = pickle.load(f)

    ret,camera_matrix,dist,rvecs,tvecs=data
    
    return  ret,camera_matrix,dist,rvecs,tvecs

def process(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray,
        aruco_dict,
        parameters=parameters
    )

    output = aruco.drawDetectedMarkers(img, corners, ids)
    
    R, T, _ = aruco.estimatePoseSingleMarkers(corners, markerLength, camera_matrix, dist)

    if R is not None:
        for i in range(len(R)):
            aruco.drawAxis(output,camera_matrix,dist,R[i],T[i], markerLength)
    
    return output

def extract(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray,
        aruco_dict,
        parameters=parameters
    )

    rotations, translations, _= aruco.estimatePoseSingleMarkers(corners, markerLength, camera_matrix, dist)
    
    if translations is None: 
        print('T_NONE')
        return False
    
    return (corners,ids,rotations,translations)




def visualize(vidfile,posefile): 
    with open(posefile,'rb') as f: 
        data = pickle.load(f)

    print(f'Data missing for {data.count(False)}/{len(data)} frames')
    vid = cv2.VideoCapture(vidfile)
    i = 0
    while True: 
        res, frame = vid.read()
        out = frame.copy()
        if data[i]:
            corners,ids,rotations,translations = data[i]
            print('C')
            c = corners[0][0]
            cv2.circle(out,(c[:,0].mean(),c[:,1].mean()),10,(0,0,255),6)
        if not res: 
            print('Finished reading VideoCapture')
            break
        out = aruco.drawDetectedMarkers(out, corners, ids)
        i += 1
        cv2.imshow('output',out)
        
        
        k = cv2.waitKey(1)
        if k == 27: 
            break

    
    cv2.destroyAllWindows()
    vid.release()

def extract_poses(vidfile,posefile,show=True):
    data = []
    cap = cv2.VideoCapture(vidfile)
    while True: 
        res, frame = cap.read()
        if not res: 
            print('Error reading VideoCapture')
            break
        data.append(extract(frame))
        out = process(frame)

        if show: 
            out = cv2.resize(out, (1280,720))
            cv2.imshow('output', out)

            k = cv2.waitKey(1)
            if k==27:
                break

    cv2.destroyAllWindows()
    cap.release()

    with open(posefile,'wb') as f: 
        pickle.dump(data,f)

ret,camera_matrix,dist,rvecs,tvecs = load_calib()
extract_poses(sys.argv[1],sys.argv[2])