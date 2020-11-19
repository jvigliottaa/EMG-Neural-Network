from Myo.myo_raw import MyoRaw
from common import *
from scipy.signal import butter, filtfilt
from tensorflow import lite
import gpio
import serial

def preprocessing_and_feature_extraction(emg_stack):
    data_formatted = np.array(emg_stack).T.tolist()

    sensor_group = []
    for sensor in data_formatted:
        # Change all to floats
        sensor = np.array([float(i) for i in sensor])

        # Normalized [-1,1]
        normalized = 2 * ((sensor - np.min(sensor)) / (np.max(sensor) - np.min(sensor))) - 1
        abs_norm = np.abs(normalized)

        # Butterworth filter
        normal_cutoff = CUTOFF / NYQ
        b, a = butter(BUTTER_WORTH_ORDER, normal_cutoff, btype='low', analog=False)
        process_sig = filtfilt(b, a, abs_norm)

        sensor_group.append(mav(process_sig))
        sensor_group.append(rms(process_sig))
        sensor_group.append(ssc(process_sig, 0.15))
        sensor_group.append(wav(process_sig))
        sensor_group.append(ahp(process_sig))
        sensor_group.append(mhp(process_sig))
        sensor_group.append(chp(process_sig))
        sensor_group.append(mav(sensor))
        sensor_group.append(rms(sensor))
        sensor_group.append(ssc(sensor, 0.15))
        sensor_group.append(wav(sensor))
        sensor_group.append(ahp(sensor))
        sensor_group.append(mhp(sensor))
        sensor_group.append(chp(sensor))


    return sensor_group


"""
0: WAVE LEFT
1: WAVE RIGHT
2: FIST
4: REST
"""

gesture_dict = {0: "Wave Left", 1: "Wave Right", 2: "Fist", 3:"Spread", 4: "Rest"}
previous_gesture = ""
gesture_stack = []
emg_stack = []


def handle_emg_stack(emg_stack):
    feature_extraction = preprocessing_and_feature_extraction(emg_stack)
    feature_extraction = np.array([feature_extraction], dtype=np.float32)

    interpreter.set_tensor(input_details[0]['index'], feature_extraction)
    interpreter.invoke()
    prediction_matrix = interpreter.get_tensor(output_details[0]["index"]).tolist()[0]


    predic = prediction_matrix.index(max(prediction_matrix))
    #print(gesture_dict.get(predic))
    show_gesture(predic)



def show_gesture(predic):
    global previous_gesture
    gesture_stack.append(predic)

    # If three are the same
    if all(ele == predic for ele in gesture_stack[-4:]):
        gesture = gesture_dict.get(predic)
        if gesture != previous_gesture:
            send_to_arduino(gesture)
            print(gesture)
            previous_gesture = gesture

def send_to_arduino(gesture):
    if gesture == "Wave Left":
        gpio.wave_left()
    elif gesture == "Wave Right":
        gpio.wave_right()
    elif gesture == "Spread":
        gpio.spread()
    elif gesture == "Fist":
        gpio.fist()
    elif gesture == "Rest":
        gpio.rest()


def proc_emg(emg, moving, times=[]):
    if len(emg_stack) < WINDOW_SAMPLE:
        emg_stack.append(emg)
    else:
        handle_emg_stack(emg_stack)
        del emg_stack[:STEP_SAMPLE]

def start_module():
    gpio.green_turn_on()
    m = MyoRaw(tty="/dev/ttyDONG")

    m.add_emg_handler(proc_emg)
    m.connect()
    gpio.green_turn_off()
    gpio.yellow_turn_on()
    return m


def start_real_time(model_path):
    global interpreter, input_details, output_details, ser

    #Get model
    interpreter = lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

       
    try:
        m = start_module()
    except Exception as e:
        gpio.yellow_turn_off()
        print("ERROR CONNECTION TO BLE")
        print(e)
        
    empty_list = []
    while True:
        packet = m.run(3)
        if not packet:
            empty_list.append("EMPTY")
            if len(empty_list) > 2:
                empty_list.clear()
                gpio.yellow_turn_off()
                print("CONNECTION LOST... RESTARTING.....")
                try:
                    m = start_module()
                except Exception as e:
                    gpio.yellow_turn_off()
                    print("ERROR CONNECTION TO BLE LOOP")
                    print(e)
        else:
            empty_list.clear()
