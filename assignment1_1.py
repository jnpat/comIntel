import numpy as np
import matplotlib.pyplot as plt

def read_data(fname):
  f = open(fname,'r').read().splitlines()
  line = f[2:] #remove 2 rows up
  m = []

  for i in line:
    m.append(i.split('\t'))

  return m

def cross(m, random): #10 ford cross validation data
  m = m.copy()
  test, train = [],[]
  rd = np.random.randint(1,11)

  while rd in random:
    rd = np.random.randint(1,11)
  random.append(rd)
  l = round(len(m)/10)
  a = l * rd
  test = m[(a - l):a]
  del m[(a - l):a]
  train = m

  return test, train

def randTrain(x):
  y = x.copy()
  random = []
  train = []
  t = []
  l = int(len(y)/10)

  for i in range(10):
    rd = np.random.randint(0,10)
    while rd in random:
      rd = np.random.randint(0,10)
    random.append(rd)

  for i in random:
    t.append(y[rd*l:(rd*l)+l])

  for i in range(len(t)):
    for j in t[i]:
      train.append(j)

  return train
 

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
    node = []
    inputs = 8
    output = 1
    self.weight = []
    self.deltaW = []
    self.deltaB = []
    self.wP = 0
    self.wbP = 0
    self.hiddenLayer = int(input("Hidden Layer = "))
    node.append(inputs)
    i = 0

    while i < self.hiddenLayer:
      node.append(int(input("Hidden Node Layer (" + str(i+1) + ") = ")))
      i = i + 1
    self.learningRate = float(input("Learning Rate = "))
    self.momentumRate = float(input("Momentum Rate = "))
    self.learningRate = -0.2
    self.momentumRate = 0.1
    node.append(output)
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

  def forward(self, x):
    y = []

    for i in range(self.hiddenLayer + 1):
      v = np.dot(x,self.weight[i]) 
      x = v + self.weightBias[i]
      x = self.sigmoid(x) #sigmoid
      y.append(np.reshape(x, (len(x), 1)))

    return y

  def gradients(self, e):
    gd = []
    graOut = self.diffSigmoid(self.y[-1]) * e #calculate gradient output
    self.weight1 = np.flip(self.weight).copy() #flip weight for back loop
    self.y1 = np.flip(self.y).copy() #flip output for back loop
    yHid = self.y1[1:] #output hidden
    wHid = self.weight1[:-1] #weight hidden
    gd.append(graOut)
    
    for i in range(self.hiddenLayer):
      gd.append((np.dot(gd[i], np.transpose(wHid[i]))) * np.transpose(self.diffSigmoid(yHid[i])))

    return gd, yHid

  def delta(self, g, yHid, x):
    dws = []
    dbs = []
    gd = []
    gd = g[:-1]
    self.weightBias1 = np.flip(self.weightBias).copy()
    
    for i in range(self.hiddenLayer):
      dw = -(self.learningRate) * gd[i] * yHid[i]
      dws.insert(0, dw)
    dw = -(self.learningRate) * g[-1] * np.reshape(x, (len(x), 1))
    dws.insert(0,dw)

    for j in range(self.hiddenLayer + 1):
      db = (-(self.learningRate) * g[j][0] * self.weightBias1[j])
      dbs.insert(0, db)

    return dws, dbs

  def backward(self, x, e):
    gradient, yHidden = self.gradients(e)
    self.deltaW, self.deltaB = self.delta(gradient, yHidden, x)

    if self.wP == 0 and self.wbP == 0:
      self.wP = self.weight.copy()
      self.wbP = self.weightBias.copy()

    for i in range(self.hiddenLayer + 1):
      self.weight[i] = self.weight[i] + self.momentumRate * (self.weight[i] - self.wP[i]) + self.deltaW[i]
      self.weightBias[i] = self.weightBias[i] + self.momentumRate * (self.weightBias[i] - self.wbP[i]) + self.deltaB[i]
    self.wP = self.weight.copy()
    self.wbP = self.weightBias.copy()

  def train(self, x, d):
    self.y = []
    self.y = self.forward(x)
    e = self.error(d, self.y) #calculate error
    self.backward(x, e)

    return e

  def test(self, x, d):
    y = []
    y = self.forward(x)
    e = self.error(d, y)

    return e
  
def sumSquare(e):
  err = 0

  for i in e:
    err = err + pow(i,2)
  return err/2

#main
random = []
epochs = 1000
ep = 0
etn = []
ets = []
error = []
errors = []
m = read_data('Flood_dataset.txt')
m = np.array(list(map(lambda sl: list(map(float, sl)), m)))
max = np.max(m)
min = np.min(m)
# normalization input data to [0,1]
m = normalize(m, max, min)
#10 ford cross validation data
test, train = cross(m, random)
train = randTrain(train)
# print(len(train))
#split test/train input and desired output
xTest, dTest = split(test)
xTrain, dTrain = split(train)
NN = NN()

while ep < epochs:
  avtn = []
  for i in range(int(epochs/10)): #How many epoch for 1 cross validation
    err = []
    errt = []
    for j in range(len(xTrain)):
      e = NN.train(xTrain[j], dTrain[j])
      err.append(e)
    train = randTrain(train)
    xTrain, dTrain = split(train)
    ep = ep + 1
    print('epoch' + str(ep))
    if ep == epochs:
      break
    etn.append(sumSquare(err))
    avtn.append(etn)
    # print('Sum Sqaure Error Train = ',etn)
  # print(xTest)
  error.append(np.average(avtn))
  print('Sum Sqaure Error Train[Average] = ',np.average(avtn))
  for k in range(len(xTest)):
    et = NN.test(xTest[k], dTest[k])
    # print(et)
    errt.append(et)
  # print(errt)
  ets.append(sumSquare(errt[0][0]))
  errors.append(ets[0])
  print('Sum Sqaure Error Test = ',ets)
  etn = []
  ets = []
  if len(random) == 10:
    random = []
    break
  #10 ford cross validation data
  test, train = cross(m, random)
  train = randTrain(train)
  # print(len(train))
  #split test/train input and desired output
  xTest, dTest = split(test)
  xTrain, dTrain = split(train)
  
#plot graph
x = [1,2,3,4,5,6,7,8,9,10]
# plt.bar(x, errors, width = 0.8, color = ['lightblue']) 
plt.plot(x, error, color='lightblue', linewidth = 3, marker='o', markerfacecolor='dodgerblue', markersize=12, label = "Train")
# plt.plot(x, errors, color='plum', linewidth = 3, marker='o', markerfacecolor='m', markersize=12, label = "Test")
plt.legend()
plt.xlabel('Iteration')  
plt.ylabel('Error')  
plt.show()



