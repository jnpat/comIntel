import numpy as np
import copy
np.random.seed(0)
def read_data(fname):
  f = open(fname,'r').read().splitlines()
  line = f[2:] #remove 2 rows up
  m = []
  for i in line:
    m.append(i.split('\t'))
  return m

def cross(m): #10 ford cross validation data
  rd = np.random.randint(1,11)
  test, train = [],[]
  l = round(len(m)/10)
  a = l * rd
  test = m[a - l:a]
  del m[a - l:a]
  train = m
  return test, train

def split(m):
  x = []
  d = []
  for i in m:
    x.append(i[:8])
    d.append(i[8])
  return np.array(x), np.array(d)  

def normalize(m, max, min):
  m = (m - min) / (max - min)
  return m.tolist()

def denormalize(m, max, min):
  m = m * (max - min) + min
  return m

class NN(object):
  def __init__(self):
    node = [8,2,2,1]
    inputs = 8
    output = 1
    self.weight = []
    self.deltaW = []
    self.deltaB = []
    # self.hiddenLayer = int(input("Hidden Layer = "))
    self.hiddenLayer = 2
    # node.append(inputs)
    # i = 0
    # while i < self.hiddenLayer:
    #   node.append(int(input("Hidden Node Layer (" + str(i+1) + ") = ")))
    #   i = i + 1
    # self.learningRate = float(input("Learning Rate = "))
    # self.momentumRate = float(input("Momentum Rate = "))
    self.learningRate = 0.1
    self.momentumRate = 0.1
    # node.append(output)
    node1, node2 = node.copy(), node.copy()
    node1.pop()
    node2.pop(0)

    for j in range(len(node1)):
      w = 2 * np.random.rand(node1[j], node2[j]) - 1
      self.weight.append(w)

    self.biass(node2) #call def bias

  def biass(self, n):
    self.weightBias = []

    for i in n: 
      self.weightBias.append(2 * np.random.rand(i) - 1)

  def sigmoid(self, s):    # activation function
    return 1/(1+np.exp(-s))

  def error(self, d, y):
    e = d - y[-1] #error
    return e

  def diffSigmoid(self, s): #derivative of sigmoid
    return (s) * (1 - (s))

  def forward(self, x, d):
    y = []
    # self.bias = []
    print('wb' + str(self.weightBias) +'\n')
    print('x' + str(x) +'\n')
    print('w' + str(self.weight) +'\n')
    # for j in range(len(self.biasIn)):
    #   self.bias.append(self.biasIn[j][0] * self.weightBias[j][0])
    for i in range(self.hiddenLayer + 1):
      v = np.dot(x,self.weight[i]) 
      print('v' + str(v) +'\n')
      x = v + self.weightBias[i]
      print('x' + str(x) +'\n')
      x = self.sigmoid(x) #sigmoid
      print('xd' + str(x) +'\n')
      y.append(np.reshape(x, (len(x), 1)))
      print('y' + str(y) +'\n')
    
    # print(self.weight)

    e = self.error(d, y) #calculate error
    return y, e

  def gradients(self, e):
    gd = []
    graOut = self.diffSigmoid(self.y[-1]) * e #calculate gradient output
    self.weight1 = np.flip(self.weight).copy() #flip weight for back loop
    # print(self.weight)
    # print(self.bias)
    self.y1 = np.flip(self.y).copy() #flip output for back loop
    # print('s' + str(self.y1))
    yHid = self.y1[1:] #output hidden
    wHid = self.weight1[:-1] #weight hidden
    print('wwwwww' +str(wHid))

    # print(yHid)
    # print(wHid)
    gd.append(graOut)
    print('gd' + str(gd) +'\n')
    for i in range(self.hiddenLayer):
      gd.append((np.dot(gd[i], np.transpose(wHid[i]))) * np.transpose(self.diffSigmoid(yHid[i])))
    print('gd' + str(gd) +'\n') 
    print('yyyyy' + str(yHid))
    return gd, yHid

  def delta(self, g, yHid, x):
    dw = []
    dbs = []
    gd = []
    gd = g[:-1]
    print('gd' + str(gd) +'\n') 
    print('g' + str(g) +'\n') 
    # print(self.weightBias)
    # print(g)
    self.weightBias1 = np.flip(self.weightBias).copy()
    # print(self.weightBias)
    # print('gd' + str(gd) +'\n')
    # print('yhid' + str(yHid) +'\n')
    for i in range(self.hiddenLayer):
      dw.append(-(self.learningRate) * gd[i] * yHid[i])
      # print(dw)
    dw.append(-(self.learningRate) * g[-1] * np.reshape(x, (len(x), 1)))
    print('dw' + str(dw) +'\n') 
    print('wb1' + str(self.weightBias1) +'\n')
    for j in range(self.hiddenLayer + 1):
      db = (-(self.learningRate) * g[j][0] * self.weightBias1[j])
      dbs.insert(0, db)

    # print('dw' + str(dw) +'\n')
    print('dbs' + str(dbs) +'\n')

  
    return np.flip(dw), dbs

  def backward(self, x, e):
    # self.deltaWP = 0
    # self.deltaBP = np.array(0)
    gradient, yHidden = self.gradients(e)
    self.deltaW, self.deltaB = self.delta(gradient, yHidden, x)
    print('dw' + str(self.deltaW) +'\n')
    print('db' + str(self.deltaB) +'\n')  
    # self.deltaW, self.weight = np.flip(self.deltaW), np.flip(self.weight)
    # self.weightBias = np.flip(self.weightBias)
    print('w' + str(self.weight) +'\n')
    #save weight
    # print(self.deltaW)
    # print(self.deltaB)
    # self.weight = self.weight + self.momentumRate * (self.deltaW - self.deltaWP) + self.deltaW
    # self.deltaWP = self.deltaW
    # print(self.weight)
    # print(self.weightBias)
    for i in range(self.hiddenLayer + 1):
      self.weight[i] = self.weight[i] + self.momentumRate * (self.deltaW[i] - self.deltaWP[i]) + self.deltaW[i]
      self.weightBias[i] = self.weightBias[i] + self.momentumRate * (self.deltaB[i] - self.deltaBP[i]) + self.deltaB[i]
    # self.deltaBP = self.deltaB

    print(self.weightBias)

  def train(self, x, d):
    e = 0
    for i in range(2):
      self.y = []
      self.y, e = self.forward(x, d)
      if i <= 0:
        gd, yh = self.gradients(e)
        self.deltaWP, self.deltaBP = self.delta(gd, yh, x)
        # print('' + str(self.weight) +'\n')
        self.backward(x, e)
      else:
        self.backward(x, e)
        gd, yh = self.gradients(e)
        self.deltaWP, self.deltaBP = self.delta(gd, yh, x)
      print('epoch = ' + str(i) + ' Error = ' + str(e))
      e = e + e
    print('Average Error = ' + str(e/2))

  

#main
m = read_data('Flood_dataset.txt')
m = np.array(list(map(lambda sl: list(map(float, sl)), m)))
max = np.max(m)
min = np.min(m)
# normalization input data to [0,1]
m = normalize(m, max, min)
#10 ford cross validation data
test, train = cross(m)
#split test/train input and desired output
xTest, dTest = split(test)
xTrain, dTrain = split(train)
NN = NN()

for i in range(len(xTrain)):
  NN.train(xTrain[i], dTrain[i])

# e = NN.train(xTrain[0], dTrain[0])
# print(e)

# print(xTest, dTest, xTrain, dTrain)
# print(type(xTest), len(dTest), type(xTrain), len(dTrain))
# # denormalization output data[0,1] to real number
# m = denormalize(m,max,min)




