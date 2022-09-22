#!/home/ausaf/Documents/aruco_hand/venv/bin/python
import numpy as np
import cv2
import PIL
from cv2 import aruco
import matplotlib.pyplot as plt
import argparse
import pickle
import sys,os

CAP = "./aruco_vid.webm"
OUT = None
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()
board = aruco.CharucoBoard_create(7, 5, 1, .8, aruco_dict)
# dir(cv2.aruco)

datadir = "/home/ausaf/Documents/aruco_hand/calib2/"
images = np.array(
    [datadir + f for f in os.listdir(datadir) if f.endswith(".png")])
# order = np.argsort([int(p.split(".")[-2].split("_")[-1]) for p in images])
# images = images[order]

retval, camera_matrix, dist, rvecs, tvecs = None,None,None,None,None

def read_chessboards(images):
    """
    Charuco base pose estimation.
    """
    print("POSE ESTIMATION STARTS:")
    allCorners = []
    allIds = []
    decimator = 0
    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    for im in images:
        print("=> Processing image {0}".format(im))
        frame = cv2.imread(im)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
            gray, aruco_dict)

        if len(corners) > 0:
            # SUB PIXEL DETECTION
            for corner in corners:
                cv2.cornerSubPix(gray, corner,
                                 winSize=(3, 3),
                                 zeroZone=(-1, -1),
                                 criteria=criteria)
            res2 = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board)
            if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3 and decimator % 1 == 0:
                allCorners.append(res2[1])
                allIds.append(res2[2])

        decimator += 1

    imsize = gray.shape
    return allCorners, allIds, imsize


def calibrate_camera():
    allCorners, allIds, imsize = read_chessboards(images)
    """
    Calibrates the camera using the detected corners.
    """
    print("CAMERA CALIBRATION")

    cameraMatrixInit = np.array([[1000.,    0., imsize[0]/2.],
                                 [0., 1000., imsize[1]/2.],
                                 [0.,    0.,           1.]])

    distCoeffsInit = np.zeros((5, 1))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS +
             cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    #flags = (cv2.CALIB_RATIONAL_MODEL)
    (ret, camera_matrix, distortion_coefficients,
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
        charucoCorners=allCorners,
        charucoIds=allIds,
        board=board,
        imageSize=imsize,
        cameraMatrix=cameraMatrixInit,
        distCoeffs=distCoeffsInit,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

    with open('calib.pickle', 'wb') as f: 
        data = [ret, camera_matrix, distortion_coefficients,
     rotation_vectors, translation_vectors]
        pickle.dump(data, f)
        print('[*] Wrote calibration info to calib.pickle')

    return ret, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors

log = []
def process(img):
    img = CLAHE(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray,
        aruco_dict,
        parameters=parameters
    )
    
    if isinstance(OUT, str):
        log.append((corners, ids))
    elif OUT is not None: 
        OUT.write(img)
    
    # output = aruco.drawDetectedMarkers(img, corners, ids)
    output = aruco.drawAxis(img, camera_matrix, dist, rvecs[0], tvecs[0], 11)
    output = draw_axis(img, rvecs, tvecs, )
    return output

gridsize=5
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(gridsize,gridsize))

def CLAHE(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr


def main():
    capture = cv2.VideoCapture(CAP)
    # p = plt.imshow(None)
    while capture.isOpened():
        success, frame = capture.read()

        if not success:
            print('Finished.')
            break

        frame = process(frame)
        # small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5) 
        cv2.imshow(f'Camera {CAP}', frame)

        keyPressed = cv2.waitKey(1)
        if keyPressed == ord('q'):
            break
    
    if isinstance(OUT, str):
        with open(OUT, 'wb') as f: 
            pickle.dump(log, f)
    elif OUT is not None: 
        OUT.release()
    
    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print(sys.argv)
    if sys.argv[1]:
        c = sys.argv[1]
        if c == '-p':
            CAP = sys.argv[2]
            OUT = sys.argv[3]

            with open('calib.pickle', 'rb') as f: 
                data = pickle.load(f)
            retval, camera_matrix, dist, rvecs, tvecs = data 
            
        elif c == '-c':
            calibrate_camera()
            exit()
        elif c == '-a': 
            with open('calib.pickle', 'rb') as f: 
                data = pickle.load(f)
            retval, camera_matrix, dist, rvecs, tvecs = data 
            
            CAP = int(sys.argv[2])
            print('running visual')
            # fourcc = -1 
            # OUT = cv2.VideoWriter(sys.argv[3], fourcc, 20.0, (1920,1080) )

    main()
