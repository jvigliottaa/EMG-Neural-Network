from Myo.myo_raw import MyoRaw
from common import *
from scipy.signal import butter, filtfilt
from keras.models import load_model, save_model
from tensorflow import lite

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
    feature_extraction = np.expand_dims(preprocessing_and_feature_extraction(emg_stack), axis=0)

    prediction_matrix = model.predict(feature_extraction)[0].tolist()

    predic = prediction_matrix.index(max(prediction_matrix))
    # #print("{}: {}".format(predic, prediction_matrix))
    show_gesture(predic)


def show_gesture(predic):
    global previous_gesture
    gesture_stack.append(predic)

    # If three are the same
    if all(ele == predic for ele in gesture_stack[-3:]):
        gesture = gesture_dict.get(predic)
        if gesture != previous_gesture and predic != 4:
            print(gesture)
            previous_gesture = gesture


def proc_emg(emg, moving, times=[]):
    if len(emg_stack) < WINDOW_SAMPLE:
        emg_stack.append(emg)
    else:
        handle_emg_stack(emg_stack)
        del emg_stack[:STEP_SAMPLE]

def save_tflite_model():
    converter = lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open("linear.tflite", "wb").write(tflite_model)


def start_real_time(model_path):
    global model
    model = load_model(model_path)

    m = MyoRaw()

    m.add_emg_handler(proc_emg)
    m.connect()

    #Convert a model to tfLite Model for raspi
    #save_tflite_model()

    while True:
        m.run(1)





