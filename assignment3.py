import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_data(fname):
    line = open(fname,'r').read().splitlines()
    m = []
    index = []
    d = []
    x= []

    for i in line:
        m.append(i.split(','))
    for i in range(len(m)):
            index.append(m[i][0])
            d.append(m[i][1])
            x.append(m[i][2:])
    return index,d,x

def minmax(x):
    np.array(x)
    mn = np.min(x)
    mx = np.max(x)
    return mn,mx

def normal(x,mn,mx):
    np.array(x)
    x = (x - mn) / (mx - mn)
    return x

def normald(d):
    n = []
    for i in d:
        if i == 'M':
            n.append([0.1])
        else:
            n.append([0.9])
    return n

def cross(x,fold):
    size = int(len(x) * fold/100)
    floor = 0
    index = []
    for i in range(1,fold+1):
        if i < fold:
            index.append([floor,i*size])
        else:
            index.append([floor,len(x)])
        floor = i*size
    return index

class NN(object):
    def __init__(self):
        node = [30,8,4,2,1]
        inputs = 30
        output = 1
        self.inBest = float("-inf")
        self.inBestPo = float("-inf")
        self.weight = []
        self.hiddenLayer = 3
        node1, node2 = node.copy(), node.copy()
        node1.pop()
        node2.pop(0)

        for j in range(len(node1)):
            w = 2 * np.random.rand(node1[j], node2[j]) - 1
            self.weight.append(w)
    
    def sigmoid(self, s):    # activation function
        return 1/(1+np.exp(-s))

    def forward(self, x):
        for i in range(self.hiddenLayer + 1):
            v = np.dot(x,self.weight[i]) 
            x = self.sigmoid(v) #sigmoid
        return x    

    def error(self, d, y):
        e = (d - y)**2 #cal mse
        return e

    def train(self, x, d):
        y = self.forward(x)
        e = self.error(d, y) #calculate error
        return e
    
    def predict(self, x, d):
        y = self.forward(x)
        if y[0] < 0.5:
            o = [0.1]
        else:
            o = [0.9]
        return o

def crossOver(mp,inv):
    Xover = []
    x = np.random.randint(len(mp)/2,inv+1)
    for i in range(x):
        partner = np.random.choice(mp, 2)
        child = pop.copy()
        for j,c in enumerate(child):
            for k,w in enumerate(c.weight):
                for n in range(w.shape[1]): #loop column in w
                    y = np.random.randint(0,2) 
                    w[:,n] = partner[y].weight[k][:,n].copy() #Xover choose partner weight to child
        for i in child:
            Xover.append(i)

    a = inv - len(Xover)
    for i in range(a):
        child = np.random.choice(pop)
        Xover.append(child)
    
    return Xover

def mutation(xo):
    mutate = xo.copy()
    for i in mutate:
        for w in i.weight:
            for n in range(w.shape[1]): #loop column in w
                y = np.random.uniform(0,1)
                if y <= 0.25:
                    mt = np.random.uniform(0,1,(w.shape[0]))
                    w[:,n] += mt
    return mutate


index,d,x = read_data('wdbc.txt')
#convert string to float
x = np.array(list(map(lambda sl: list(map(float, sl)), x)))
# normalize input to [0,1] and output
x = pd.DataFrame(x)
mn, mx = minmax(x)
m = normal(x,mn,mx)
n = normald(d)
#change format to numpy array
m = m.to_numpy()
n = np.array(n)
pop = []
err = []
errs = []
inv = 15
gen = 0
etn = 0
fx = 0


for i in range(inv):
    pop.append(NN()) #generate 15 NN individual in population

for x, y in cross(m,10):
    ftn = []
    fts = []
    etn = 0
    ets = 0
    xTn = np.concatenate((m[:x],m[y+1:]))
    dTn = np.concatenate((n[:x],n[y+1:]))
    xTs = m[x:y]
    dTs = n[x:y]
    print('---------------------- Cross Validation ----------------------')
    for gen in range(10):
        fit = []
        seed = np.random.randint(1,10000)
        np.random.seed(seed)
        np.random.shuffle(xTn)
        np.random.shuffle(dTn)
        for i,p in enumerate(pop):
            for j in range(len(m)):
                etn += p.train(m[j], n[j]) #cal avg error (mae) train
                o = p.predict(m[j], n[j])
            etn = etn/len(m) #mse error
            fx = np.abs(1/np.log(1 - etn))
            ftn.append(fx[0])
            fit.append(fx[0])

            #update best individual with fitness and weight
            if fx[0] > p.inBest:
                p.inBest = fx[0]
                p.inBestPo = p.weight.copy()

        #random selection
        matingpool = np.random.choice(pop,int(inv/2))

        #crossover
        Xover = crossOver(matingpool,inv)

        #mutate
        mutate = mutation(Xover)

        #t = t + 1
        pop = mutate.copy()

        print('epoch:',gen+1,'fitness:',np.average(fit))

    for j in range(len(xTs)):
        ets += p.train(xTs[j], dTs[j]) #cal avg error (mae) test
    ets = ets/len(xTs)
    fts.append(ets)
    print('Trianing Error:',np.average(ftn))
    err.append(np.average(ftn))
    print('Testing Error:',np.average(fts))
    err.append(np.average(fts))

#plot graph
x = [1,2,3,4,5,6,7,8,9,10]
# plt.bar(x, errors, width = 0.8, color = ['lightblue']) 
plt.plot(x, err, color='lightblue', linewidth = 3, marker='o', markerfacecolor='dodgerblue', markersize=12, label = "Train")
plt.plot(x, errs, color='plum', linewidth = 3, marker='o', markerfacecolor='m', markersize=12, label = "Test")
plt.legend()
plt.xlabel('Iteration')  
plt.ylabel('Error')  
plt.show()

            
