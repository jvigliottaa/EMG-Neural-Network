import sys
import keyboard
import csv
from time import time
from Myo.myo_raw import MyoRaw


FILE_NAME = "Data/db_3.csv"
current_pose = 0

time_start = int(time() * 1000)

csv_file = open(FILE_NAME, "w+", newline='')
write = csv.writer(csv_file)
"""
0: at rest
1: hand left
2: hand right
3: fist
4: fingers spread


DATASET:
5 Seconds On, 5 Seconds at rest
Poses (Repeat 30x):
Hand Left x 1
Hand Right x 1
Fist x 1
Fingers Spread x 1
"""
def proc_emg(emg, moving, times=[]):
    emg = list(emg)
    emg.append(current_pose)
    time_now = int(time()*1000) - time_start
    emg.insert(0, time_now)
    write.writerow(emg)
    print(emg)

m = MyoRaw(sys.argv[1] if len(sys.argv) >= 2 else None)

m.add_emg_handler(proc_emg)
m.connect()

m.add_arm_handler(lambda arm, xdir: print('arm', arm, 'xdir', xdir))
m.add_pose_handler(lambda p: print('pose', p))

while True:
    m.run(1)
    if keyboard.is_pressed("1"):
        current_pose = 1
    elif keyboard.is_pressed("2"):
        current_pose = 2
    elif keyboard.is_pressed("3"):
        current_pose = 3
    elif keyboard.is_pressed("4"):
        current_pose = 4
    else:
        current_pose = 0