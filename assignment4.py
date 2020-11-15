import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def minmax(x):
    np.array(x)
    mn = np.min(x)
    mx = np.max(x)
    return mn,mx

def normal(x,mn,mx):
    np.array(x)
    x = (x - mn) / (mx - mn)
    return x

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
        node = [8,4,2,1] 
        inputs = 8
        output = 1
        self.weight = []
        self.velocity = []
        self.pBest = float("inf")
        self.gBest = float("inf")
        self.pBestPo = float("inf")
        self.gBestPo = float("inf")
        self.hiddenLayer = 2
        node1, node2 = node.copy(), node.copy()
        node1.pop()
        node2.pop(0)

        for j in range(len(node1)):
            w = 2 * np.random.rand(node1[j], node2[j]) - 1
            v = 2 * np.random.rand(node1[j], node2[j]) - 1
            self.weight.append(w)
            self.velocity.append(v)
        self.poBest = self.weight.copy()
    
    def sigmoid(self, s):    # activation function
        return 1/(1+np.exp(-s))

    def forward(self, x):
        for i in range(self.hiddenLayer + 1):
            v = np.dot(x,self.weight[i]) 
            x = self.sigmoid(v) #sigmoid
        return x

    def error(self, d, y):
        e = np.abs(d - y) #error
        return e

    def train(self, x, d):
        y = self.forward(x)
        e = self.error(d, y) #calculate error
        return e

#reading csv file
read = pd.read_csv("AirQualityUCI.csv")
#get x (input from csv file)
x = read[['PT08.S1(CO)','PT08.S2(NMHC)','PT08.S3(NOx)','PT08.S4(NO2)','PT08.S5(O3)','T','RH','AH']]

#get y (output from s\csv file)
d = read[['C6H6(GT)']]
#normalize input & output to [0,1]
mn, mx = minmax(x)
m = normal(x,mn,mx)
mn1,mx1 = minmax(d)
n = normal(d,mn1,mx1)

#predicted 5 days after
# d1 = n.iloc[120:].reset_index(drop = True)  
# m.drop(m.tail(120).index,inplace = True)
# #predicted 10 days after
d1 = n.iloc[240:].reset_index(drop = True)  
m.drop(m.tail(240).index,inplace = True)
# print('d1',d1,'x',x)
#change format to numpy array
d1 = d1.to_numpy()
m = m.to_numpy()
swarm = []
par = 15
ep = 0
err = []
errs = []

for i in range(par):
    swarm.append(NN()) #generate 15 NN Particle in swarm

for x,y in cross(m,10):
    etnc = []
    etsc = []
    etn = 0
    ets = 0
    xTn = np.concatenate((m[:x],m[y+1:]))
    dTn = np.concatenate((d1[:x],d1[y+1:]))
    xTs = m[x:y]
    dTs = d1[x:y]
    print('---------------------- Cross Validation ----------------------')
    for ep in range(15):
        e = []
        seed = np.random.randint(1,10000)
        np.random.seed(seed)
        np.random.shuffle(xTn)
        np.random.shuffle(dTn)
        for i,p in enumerate(swarm):
            for j in range(len(xTn)):
                etn += p.train(xTn[j], dTn[j]) #cal avg error (mae) train
            etn = etn/len(xTn)
            etnc.append(etn[0])
            e.append(etn[0])
            

            #update pBest
            if etn[0] < p.pBest:
                p.pBest = etn[0]
                p.pBestPo = p.weight.copy()

            #update gBest
            if etn[0] < p.gBest:
                p.gBest = etn[0]
                p.gBestPo = p.weight.copy()

        print('epoch:',ep+1,'mae:',np.average(e))
        #cal lo1 and lo2 to update velocity with no fix upper
        for i,p in enumerate(swarm):
            r1,r2 = np.random.uniform(0.1,1),np.random.uniform(0.1,1)
            c1,c2 = np.random.uniform(0.1,1),np.random.uniform(0.1,1)
            while c1+c2 > 4:
                c1,c2 = np.random.uniform(0.1,1),np.random.uniform(0.1,1)
            lo1 = c1*r1
            lo2 = c2*r2
            for j in range(len(p.velocity)):
                p.velocity[j] = p.velocity[j] + lo1*(p.pBestPo[j] - p.weight[j]) + lo2*(p.gBestPo[j] - p.weight[j]) #update velocity
                p.weight[j] = p.weight[j] + p.velocity[j] #move to new position
        
        # #cal lo1 and lo2 to update velocity with fix upper 
        # for i,p in enumerate(swarm):
        #     lo1,lo2 = np.random.randint(1,5),np.random.randint(1,5)
        #     lo = lo1+lo2
        #     while lo < 4 :
        #         lo1,lo2 = np.random.randint(1,5),np.random.randint(1,5)
        #     k = 1 - (1/lo) + (np.sqrt(np.abs((lo**2) - (4*lo)))/2)
        #     for j in range(len(p.velocity)):
        #         # print(p.velocity[j])
        #         p.velocity[j] = k*(p.velocity[j] + lo1*(p.pBestPo[j] - p.weight[j]) + lo2*(p.gBestPo[j] - p.weight[j])) #update velocity
        #         p.weight[j] = p.weight[j] + p.velocity[j] #move to new position

    for j in range(len(xTs)):
        ets += p.train(xTs[j], dTs[j]) #cal avg error (mae) test
    ets = ets/len(xTs)
    etsc.append(ets)
    print('Trianing Error:',np.average(etnc))
    err.append(np.average(etnc))
    print('Testing Error:',np.average(etsc))
    errs.append(np.average(etsc))

#plot graph
x = [1,2,3,4,5,6,7,8,9,10]
# plt.bar(x, errors, width = 0.8, color = ['lightblue']) 
plt.plot(x, err, color='lightblue', linewidth = 3, marker='o', markerfacecolor='dodgerblue', markersize=12, label = "Train")
plt.plot(x, errs, color='plum', linewidth = 3, marker='o', markerfacecolor='m', markersize=12, label = "Test")
plt.legend()
plt.xlabel('Iteration')  
plt.ylabel('Error')  
plt.show()