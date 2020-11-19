import gpio
gpio.red_turn_on()


import sys
import real_time
import time
import argparse



model_file_name = "/home/pi/Desktop/REMAKE/Myo-EMG-NN-Raspi/linear.tflite"


def main():
    #wait for system bootup
    try:
        real_time.start_real_time(model_file_name)
    except Exception as e:
        print("ERROR RUNNING SCRIPT")
        print(e)
    finally:
        gpio.turn_off_all()

if __name__ == "__main__":
    main()
