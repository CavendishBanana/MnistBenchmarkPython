# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import constants
import tensorflow as tf
from keras.layers import Dense
from keras.activations import relu, softmax
from timeit import default_timer as timer


def get_numpy_arr_from_list_of_ints(list_of_ints, rows_count, cols_count):
    for i in range(len(list_of_ints)):
        list_of_ints[i] = np.ubyte(list_of_ints[i])
    bytes_np_arr = np.array(list_of_ints)
    bytesss = bytes_np_arr.tobytes()
    floats_array = np.frombuffer(bytesss, np.single)
    if cols_count > 1:
        return floats_array.reshape((rows_count, cols_count))
    return floats_array


def load_n_data_lines(path_to_file, n):
    lines = []
    file1 = open(path_to_file, "r")
    count = 0

    for i in range(n):
        line = file1.readline()
        lines.append(line)
    file1.close()
    return lines


def build_numpy_arr_from_file_line(fileline, ret_arr_rows_count, ret_arr_cols_count):
    string_numbers = fileline.split(",")
    label = int(string_numbers[0])
    floats_arr = np.zeros((ret_arr_rows_count * ret_arr_cols_count,), np.single)
    float_one = np.single(1.0)
    one_by_255 = 1.0 / 255.0
    j = 0
    for i in range(len(string_numbers)):
        if i == 0:
            continue
        floats_arr[j] = min(np.single(np.uint(string_numbers[i]) * one_by_255), float_one)
        j += 1
    return floats_arr, label


def build_dataset(data_string_lines, data_frame_rows_count, data_frame_cols_count):
    dataset = np.zeros((len(data_string_lines), 1, data_frame_rows_count, data_frame_cols_count,))
    labels = np.zeros((len(data_string_lines),), np.uint)
    i = 0
    for dataline in data_string_lines:
        dataframe, label = build_numpy_arr_from_file_line(dataline, data_frame_rows_count, data_frame_cols_count)
        dataframe = dataframe.reshape(data_frame_rows_count, data_frame_cols_count)
        dataset[i, 0] = dataframe
        labels[i] = label
        i += 1
    return [dataset, labels]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    flat_input_pixels_count = 784
    input_rows = 28
    input_cols = 28

    n_samples = 10000
    # load benchmarking data
    first_n_lines = load_n_data_lines("data\\mnist_test.csv", n_samples)
    dataset_and_labels = build_dataset(first_n_lines, 28, 28)

    model_layer_1 = get_numpy_arr_from_list_of_ints(constants.layer_1_weights, constants.l_1_weights_rows_count,
                                                    constants.l_1_weights_cols_count)
    model_layer_2 = get_numpy_arr_from_list_of_ints(constants.layer_2_weights, constants.l_2_weights_rows_count,
                                                    constants.l_2_weights_cols_count)

    model_layer_1_biases = get_numpy_arr_from_list_of_ints(constants.layer_1_biases, constants.l_1_biases_cols_count,
                                                           constants.l_1_biases_rows_count)
    model_layer_2_biases = get_numpy_arr_from_list_of_ints(constants.layer_2_biases, constants.l_2_biases_cols_count,
                                                           constants.l_2_biases_rows_count)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Softmax()
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()], )

    list_layer_1 = [model_layer_1, model_layer_1_biases]
    list_layer_2 = [model_layer_2, model_layer_2_biases]
    model.layers[1].set_weights(list_layer_1)
    model.layers[2].set_weights(list_layer_2)

    results_report = np.zeros((2,), np.uint)

    start = timer()

    for frame_idx in range(dataset_and_labels[0].shape[0]):
        model_prediction = model.predict(dataset_and_labels[0][frame_idx], verbose=0)
        predicted_number = 0
        biggest = 0.0
        for i in range(model_prediction.shape[1]):
            if model_prediction[0, i] > biggest:
                biggest = model_prediction[0, i]
                predicted_number = i
        results_report[int(predicted_number == dataset_and_labels[1][frame_idx])] += 1
        # print("predicted number: ", predicted_number, ", label: ", dataset_and_labels[1][frame_idx])
    end = timer()
    print("Samples count: ", n_samples)
    print("Execution time: ", str(end - start), " seconds")
    print("Correct guesses: ", results_report[1], ", incorrect guesses: ", results_report[0])
