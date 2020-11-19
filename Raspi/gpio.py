import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

#LEDs
GPIO.setup(21, GPIO.OUT)
GPIO.setup(16, GPIO.OUT)
GPIO.setup(20, GPIO.OUT)

#ARDUINOS
GPIO.setup(2, GPIO.OUT)
GPIO.setup(3, GPIO.OUT)
GPIO.setup(4, GPIO.OUT)



def turn_off_all():
    GPIO.output(21, GPIO.LOW)
    GPIO.output(20, GPIO.LOW)
    GPIO.output(16, GPIO.LOW)
    
    GPIO.output(2, GPIO.LOW)
    GPIO.output(3, GPIO.LOW)
    GPIO.output(4, GPIO.LOW)


turn_off_all()


def wave_left():
    GPIO.output(2, GPIO.LOW)
    GPIO.output(3, GPIO.LOW)
    GPIO.output(4, GPIO.HIGH)

def wave_right():
    GPIO.output(2, GPIO.LOW)
    GPIO.output(3, GPIO.HIGH)
    GPIO.output(4, GPIO.LOW)

def spread():
    GPIO.output(2, GPIO.LOW)
    GPIO.output(3, GPIO.HIGH)
    GPIO.output(4, GPIO.HIGH)

def fist():
    GPIO.output(2, GPIO.HIGH)
    GPIO.output(3, GPIO.LOW)
    GPIO.output(4, GPIO.LOW)

def rest():
    GPIO.output(2, GPIO.LOW)
    GPIO.output(3, GPIO.LOW)
    GPIO.output(4, GPIO.LOW)


def green_turn_on():
    GPIO.output(21, GPIO.HIGH)

def green_turn_off():
    GPIO.output(21, GPIO.LOW)

def yellow_turn_on():
    GPIO.output(20, GPIO.HIGH)

def yellow_turn_off():
    GPIO.output(20, GPIO.LOW)

def red_turn_on():
    GPIO.output(16, GPIO.HIGH)

def red_turn_off():
    GPIO.output(16, GPIO.LOW)



