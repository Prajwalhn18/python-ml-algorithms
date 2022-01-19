import numpy as np

x = np.array(([2,9],[1,5],[3,6]),dtype = float)
y = np.array(([92],[86],[89]),dtype = float)
x = x/np.amax(x,axis=0)
y = y / 100

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivatives_sigmoid(x):
    return x * (1-x)

epoch = 7000

learningRate = 0.1
inputLayerNeurons = 2
hiddenLayerNeurons = 3
outputLayerNeurons = 1

wh = np.random.uniform(size=(inputLayerNeurons,hiddenLayerNeurons))
bh = np.random.uniform(size=(1,hiddenLayerNeurons))
wo = np.random.uniform(size=(hiddenLayerNeurons,outputLayerNeurons))
bo = np.random.uniform(size=(1,outputLayerNeurons))

for i in range(epoch):
    netH = np.dot(x,wh) + bh
    sigmoidH = sigmoid(netH)
    netO = np.dot(sigmoidH,wo) + bo
    output =  sigmoid(netO)

    deltaK = (y-output) * derivatives_sigmoid(output)
    deltaH = deltaK.dot(wo.T) * derivatives_sigmoid(sigmoidH)

    wo = sigmoidH.T.dot(deltaK) * learningRate
    wh = wh + x.T.dot(deltaH) * learningRate

    error = sum(deltaK)**2/len(deltaK)
    print('Epoch -> {0}, lrate -> {1}, error -> {2}'.format(i, learningRate, error))

print("Input: \n " + str(x))

print("Actual Output: \n" + str(y))

print("Predicted Output: \n", output)