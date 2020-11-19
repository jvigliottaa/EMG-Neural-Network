from keras import Model, optimizers, losses
from keras.layers import Dense, Input, Dropout
from keras.models import save_model, load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import numpy as np


params = {}
f = open("Models/nn_hyper_params.txt", "r")
for param in f.readlines():
    parameter = eval(param.strip("\n"))
    params[parameter[0]] = parameter[1]

LEARNING_RATE = float(params['learning_rate'])
DROPOUT_RATE = float(params['dropout_rate'])
NUM_HIDDEN_LAYERS = int(params['num_hidden_layers'])
NEURONS_PER_LAYER = int(params['neurons_per_layer'])
BATCH_SIZE = int(params['batch_size'])
ACTIVATION = params['activation']

if params['optimizer'] == 'adam':
    OPTIMIZER = optimizers.Adam(learning_rate=LEARNING_RATE)
elif params['optimizer'] == 'rms':
    OPTIMIZER = optimizers.RMSprop(learning_rate=LEARNING_RATE)
else:
    OPTIMIZER = optimizers.SGD(learning_rate=LEARNING_RATE)


def create_and_train_network(input_training_data, output_training_data, model_file_path):

    visible = Input(shape=(112,))
    x = Dropout(DROPOUT_RATE)(visible)  # dropout on the weights.

    # Add the hidden layers.
    for i in range(NUM_HIDDEN_LAYERS):
        x = Dense(NEURONS_PER_LAYER,
                         activation=ACTIVATION)(x)
        x = Dropout(DROPOUT_RATE)(x)

    output = Dense(5, activation='sigmoid')(x)
    model = Model(inputs=visible, outputs=output)

    print(model.summary())

    model.compile(loss=losses.MeanSquaredError(), optimizer=OPTIMIZER, metrics=['accuracy'])
    model.fit(input_training_data, output_training_data, epochs=5, batch_size=BATCH_SIZE,
              validation_split=0.2)

    _, accuracy = model.evaluate(input_training_data, output_training_data, verbose=0)
    print('Accuracy: %.2f' % (accuracy*100))

    print("Saving File To: " + model_file_path)

    print(input_training_data)
    print(output_training_data)
    save_model(model, model_file_path)

def get_confusion_matrix(model_path, input_training_data, output_training_data):
    model = load_model(model_path)
    output_guesses = []
    input_training_data = input_training_data[:10000]
    output_training_data = output_training_data[:10000]
    for x in input_training_data:
        output_guesses.append(model.predict(np.expand_dims(x, axis=0))[0].tolist())
    output_guesses = np.array(np.argmax(np.array(output_guesses), axis=-1))
    output_answers = np.argmax(output_training_data, axis=-1)
    print(len(output_guesses))
    print(len(output_answers))
    cm = confusion_matrix(y_true=output_answers, y_pred=output_guesses)
    cm_plot_labels = ['Wave Left', 'Wave Right', "Fist", "Spread", "Rest"]
    plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title="Confusion Matrix")


def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.rcParams.update({'font.size': 28})

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

