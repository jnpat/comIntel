import numpy as np
import matplotlib.pyplot as plt
# np.random.seed(0)

def read_data(fname):
    f = open(fname,'r').read().splitlines()
    data,x,d = [],[],[]
    for i in range(len(f)):
        if i%3 != 0:
            data.append(f[i])
    for j in range(len(data)):
        if j%2 == 0:
            x.append(data[j].split('  '))
        else:
            d.append(data[j].split(' '))

    return x,d

def normalize(d):
    x = d.copy()
    for i in x :
        if i[-1] == 0:
            i[-1] = 0.1
        else:
            i[-1] = 0.9
        
        if i[-2] == 0:
            i[-2] = 0.1
        else:
            i[-2] = 0.9
    return x

def cross(x,d,random):
    x = x.tolist().copy()
    d = d.tolist().copy()
    xTs,xTn,dTs,dTn = [],[],[],[]
    rd = np.random.randint(1,11)
    while rd in random:
        rd = np.random.randint(1,11)
    random.append(rd)

    l = int(len(x)/10)
    a = int(l * rd)
    xTs = x[(a - l):a]
    del x[(a - l):a]
    xTn = x
    dTs = d[(a - l):a]
    del d[(a-l):a]
    dTn = d
    return xTs,xTn,dTs,dTn

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

def predict(y):
  if y[0] > y[1]:
    o = [0.9,0.1]
  else:
    o = [0.1,0.9]
  return o


class NN(object):
  def __init__(self):
    node = []
    inputs = 2
    output = 2
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
    e = []
    #error
    e.append(d[0] - y[-1][0])
    e.append(d[1] - y[-1][1])
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
      gd.append((np.dot(wHid[i],gd[i])) * self.diffSigmoid(yHid[i]))

    return gd, yHid

  def delta(self, g, yHid, x):
    dws = []
    dbs = []
    gd = []
    gd = g[:-1]
    self.weightBias1 = np.flip(self.weightBias).copy()
    
    for i in range(self.hiddenLayer):
      dw = -(self.learningRate) * gd[i] * np.transpose(yHid[i])
      dws.insert(0, dw)
    dw = -(self.learningRate) * g[-1] * x
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
      self.weight[i] = self.weight[i] + self.momentumRate * (self.weight[i] - self.wP[i]) + np.transpose(self.deltaW[i])
      self.weightBias[i] = self.weightBias[i] + self.momentumRate * (self.weightBias[i] - self.wbP[i]) + self.deltaB[i]
    self.wP = self.weight.copy()
    self.wbP = self.weightBias.copy()

  def train(self, x, d):
    self.y = []
    self.y = self.forward(x)
    e = self.error(d, self.y) #calculate error
    self.backward(x, e)
    c = predict(self.y[-1])

    return c

  def test(self, x, d):
    y = self.forward(x)
    c = predict(self.y[-1])

    return c

def confusion(d,o):
  tp,fp,fn,tn = 0,0,0,0
  for i in range(len(d)):
    if d[i] == [0.9, 0.1]:
      if o[i] == [0.9, 0.1]:
        tp = tp + 1
      else:
        fp = fp + 1
    else:
      if o[i] == [0.1, 0.9]:
        tn = tn + 1
      else:
        fn = fn + 1
  return tp,fp,fn,tn
    

#main
random = []
ep = 0
epochs = int(input("Epoch = "))
acc = []
accs = []
cl = []
clts = []
x,d = read_data('cross.pat')
#convert string to float
x = np.array(list(map(lambda sl: list(map(float, sl)), x)))
d = np.array(list(map(lambda sl: list(map(float, sl)), d)))
#normalize desired output (0-->0.1),(1-->0.9)
d = normalize(d)
#10 ford cross validation data
xTest,xTrain,dTest,dTrain = cross(x,d,random)
xTrain = randTrain(xTrain)
dTrain = randTrain(dTrain)
NN = NN()

while ep < epochs:
  for i in range(int(epochs/10)): #How many epoch for 1 cross validation
    for j in range(len(xTrain)):
      cl.append(NN.train(xTrain[j], dTrain[j]))
    xTrain = randTrain(xTrain)
    dTrain = randTrain(dTrain)
    ep = ep + 1

    print('epoch', ep)
    if ep == epochs:
      break
  
  #check & print confusion matrix
  tp,fp,fn,tn = confusion(dTrain,cl)
  print("actual\predict(train) [1 0]  [0 1]")
  print('[1 0]                   ',tp,'   ',fp)
  print('[0 1]                   ',fn,'   ',tn)
  acc.append(((tp+tn)/len(dTrain)*100))
  print('Accurancy',((tp+tn)/len(dTrain)*100),'%')

  for k in range(len(xTest)):
    clts.append(NN.test(xTest[k], dTest[k]))

  #check & print confusion matrix 
  tp,fp,fn,tn = confusion(dTest,clts)
  print("actual\predict(test)  [1 0]  [0 1]")
  print('[1 0]                   ',tp,'   ',fp)
  print('[0 1]                   ',fn,'   ',tn)
  accs.append(((tp+tn)/len(dTrain)*100))
  print('Accurancy',((tp+tn)/len(dTrain)*100),'%')
  cl = []
  clts = []

  if len(random) == 10:
    random = []
    break
  #10 ford cross validation data
  xTest,xTrain,dTest,dTrain = cross(x,d,random)
  xTrain = randTrain(xTrain)
  dTrain = randTrain(dTrain)

print('acc',acc)
#plot graph
x = [1,2,3,4,5,6,7,8,9,10]
# plt.bar(x, errors, width = 0.8, color = ['lightblue']) 
plt.plot(x, acc, color='lightgreen', linewidth = 3, marker='o', markerfacecolor='lightseagreen', markersize=12, label = "Train")
# plt.plot(x, errors, color='plum', linewidth = 3, marker='o', markerfacecolor='m', markersize=12, label = "Test")
plt.legend()
plt.xlabel('Iteration')  
plt.ylabel('Accuracy')  
plt.show()
    


