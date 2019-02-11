import neuralData
# from Crypto.Util import number


data = neuralData.Data()

data = neuralData.calculate_output(data)

# print('new w5 = ', d.w5, 'new w6 = ', d.w6)

for i in range(0, 40):
    data = neuralData.update_weight(data)
    data = neuralData.calculate_output(data)

print('\ndelta_o1')

data.l1 = 0
data.l2 = 1

data = neuralData.calculate_output(data)

for i in range(0, 40):
    data = neuralData.update_weight(data)
    data = neuralData.calculate_output(data)

print('\ndelta_o1')
