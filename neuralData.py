import math


class Data:
    def __init__(self):
        self.l1 = 1
        self.l2 = 0
        self.w1 = 0.45
        self.w2 = 0.78
        self.w3 = -0.12
        self.w4 = 0.13
        self.w5 = 1.5
        self.w6 = -2.3
        self.e = math.e
        self.Epsilon = 0.7  # скорость обучения
        self.Alpha = 0.3  # момент
        self.h1_input = 0
        self.h1_output = 0
        self.h2_input = 0
        self.h2_output = 0
        self.o1_input = 0
        self.o1_output = 0
        self.o1_ideal = 0
        self.error = None
        self.oldW1 = None
        self.oldW2 = None
        self.oldW3 = None
        self.oldW4 = None
        self.oldW5 = None
        self.oldW6 = None

    def save_synapse_data(self, values):
        self.h1_input = values['h1_input']
        self.h1_output = values['h1_output']
        self.h2_input = values['h2_input']
        self.h2_output = values['h2_output']
        self.o1_input = values['o1_input']
        self.o1_output = values['o1_output']
        self.o1_ideal = values['o1_ideal']
        self.error = values['error']

    def update_old_weight(self):
        self.oldW1 = self.w1
        self.oldW2 = self.w2
        self.oldW3 = self.w3
        self.oldW4 = self.w4
        self.oldW5 = self.w5
        self.oldW6 = self.w6


def sigmoid(x):
    return 1 / (1 + math.e ** -x)


def calculate_output(data):
    print('bla ---->', data.w5, data.w6)
    h1_input = round(data.l1 * data.w1 + data.l2 * data.w3, 3)
    h1_output = round(sigmoid(h1_input), 3)
    h2_input = round(data.l1 * data.w2 + data.l2 * data.w4, 3)
    h2_output = round(sigmoid(h2_input), 3)
    o1_input = round(h1_output * data.w5 + h2_output * data.w6, 3)
    o1_output = round(sigmoid(o1_input), 3)
    o1_ideal = data.l1 ^ data.l2
    error = round(((o1_ideal - o1_output) ** 2) / 1, 3)
    output_values = {
        'h1_input': h1_input,
        'h1_output': h1_output,
        'h2_input': h2_input,
        'h2_output': h2_output,
        'o1_input': o1_input,
        'o1_output': o1_output,
        'o1_ideal': o1_ideal,
        'error': error,
    }
    data.save_synapse_data(output_values)
    return data


def update_weight(data):
    delta_o1 = round((data.o1_ideal - data.o1_output) * ((1 - data.o1_output) * data.o1_output), 3)

    delta_h1 = round(((1 - data.h1_output) * data.h1_output) * (data.w5 * delta_o1), 3)
    #
    delta_h2 = round(((1 - data.h2_output) * data.h2_output) * (data.w6 * delta_o1), 3)
    #
    grad_w1 = round(data.l1 * delta_h1, 3)
    grad_w2 = round(data.l2 * delta_h2, 3)
    grad_w3 = round(data.l2 * delta_h1, 3)
    grad_w4 = round(data.l1 * delta_h2, 3)
    grad_w5 = round(data.h1_output * delta_o1, 3)
    grad_w6 = round(data.h2_output * delta_o1, 3)
    #
    delta_w1 = round(data.Epsilon * grad_w1 +
                     check_previous_weight(data.oldW1, data.w1) * data.Alpha, 3)
    delta_w2 = round(data.Epsilon * grad_w2 +
                     check_previous_weight(data.oldW2, data.w2) * data.Alpha, 3)
    delta_w3 = round(data.Epsilon * grad_w3 +
                     check_previous_weight(data.oldW3, data.w3) * data.Alpha, 3)
    delta_w4 = round(data.Epsilon * grad_w4 +
                     check_previous_weight(data.oldW4, data.w4) * data.Alpha, 3)
    delta_w5 = round(data.Epsilon * grad_w5 +
                     check_previous_weight(data.oldW5, data.w5) * data.Alpha, 3)
    delta_w6 = round(data.Epsilon * grad_w6 +
                     check_previous_weight(data.oldW6, data.w6) * data.Alpha, 3)
    #
    data.update_old_weight()

    data.w1 = round(data.w1 + delta_w1, 3)
    data.w2 = round(data.w2 + delta_w2, 3)
    data.w3 = round(data.w3 + delta_w3, 3)
    data.w4 = round(data.w4 + delta_w4, 3)
    data.w5 = round(data.w5 + delta_w5, 3)
    data.w6 = round(data.w6 + delta_w6, 3)

    return data


def check_previous_weight(previous_weight, weight):
    return 0 if previous_weight is None else weight - previous_weight
