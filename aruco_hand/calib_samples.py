"""Samples frames from the calibration video and saves them in the designated location"""
import cv2 

PATH = './vid/calib.mov'
OUTDIR = './calib2'
INT = 50 # sampling interval
cap = cv2.VideoCapture(PATH)

frames = []
count = 0
while True:
    res, frame = cap.read()
    if not res: 
        print('\nFinished')
        break
    
    if count % INT == 0:
        print('#'*int(count/INT), end='\r')
        frames.append(frame)
    count += 1
print(len(frames), count)\

for i in range(len(frames)):
    print(f'Writing image {i}/{len(frames)}', end='\r')
    path = f'{OUTDIR}/calib_{i}.png'
    cv2.imwrite(path, frames[i])
    