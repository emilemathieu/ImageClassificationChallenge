#%% Load DATA
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

X = pd.read_csv('../data/Xtr.csv', header=None).as_matrix()[:, 0:-1].astype(np.float32)
y = pd.read_csv('../data/Ytr.csv').as_matrix()[:,1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

X = X.reshape(-1, 3, 32, 32)
X_train = X_train.reshape(-1, 3, 32, 32)
X_test = X_test.reshape(-1, 3, 32, 32)

#%% Score on training and test datasets

def score(algo):
    correct_train = 0
    total_train = 0
    N = X_train.shape[0]
    nb_batchs = int(N / batch_size)
    for i in range(nb_batchs):
        inputs = X_train[i*batch_size:(i+1)*batch_size,:]
        labels = y_train[i*batch_size:(i+1)*batch_size]
        outputs = algo(inputs)
        predicted = np.argmax(outputs.data, axis=1)
        total_train += len(labels)
        correct_train += (predicted == labels).sum()
    
    correct_test = 0
    total_test = 0
    N = X_test.shape[0]
    nb_batchs = int(N / batch_size)
    for i in range(nb_batchs):
        inputs = X_test[i*batch_size:(i+1)*batch_size,:]
        labels = y_test[i*batch_size:(i+1)*batch_size]
        outputs = algo(inputs)
        predicted = np.argmax(outputs.data, axis=1)
        total_test += len(labels)
        correct_test += (predicted == labels).sum()
    
    return round(100*correct_train/total_train,2), round(100*correct_test/total_test,2)

#%%
from mllib import nn

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6, 12, 5),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        
        #self.flatten = nn.Flatten()
        
        self.classifier = nn.Sequential(
                nn.Linear(12*5*5, 10)
        )
    def forward(self, x):
        x = self.features(x)
        #x = self.flatten(x)
        x = x.reshape(-1, 12*5*5)
        x = self.classifier(x)
        return x.reshape(x.shape[0],-1)

    def backward(self, output_grad):
        output_grad = self.classifier.backward(output_grad)
        #output_grad = self.flatten.backward(output_grad)
        output_grad = output_grad.reshape(-1, 12, 5, 5)
        return self.features.backward(output_grad)    

    def step(self, optimizer):
        self.classifier.step(optimizer)
        self.features.step(optimizer)

    def zero_grad(self):
        self.classifier.zero_grad()
        self.features.zero_grad()

    def parameters(self):
        return self.features.parameters() + self.classifier.parameters()

mynet = MyNet()

#%% Copy weights from torch CNN
#parameters = []
#for parameter in net.parameters():
#    print(type(parameter))
#    parameters.append(parameter.data.numpy())
#
#mynet.features._modules[0]._weight = parameters[0].copy()
#mynet.features._modules[0]._bias = parameters[1].copy()
#mynet.features._modules[3]._weight = parameters[2].copy()
#mynet.features._modules[3]._bias = parameters[3].copy()
#mynet.classifier._modules[0]._weight = parameters[4].copy()
#mynet.classifier._modules[0]._bias = parameters[5].copy()

#%% Train
from importlib import reload 
from mllib import optim, loss
import timeit, pickle, json, os

experience_name = '3__depth_12_deterministic_batch'
directory_path = 'parameters/{}/'.format(experience_name)
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

#optimizer = optim.SGD(lr=0.001, momentum=0.9)
optimizer = optim.Adam(lr=0.001)

criterion = loss.CrossEntropyLoss()

N = X_train.shape[0]
batch_size = 8
nb_batchs = int(N / batch_size)

scores = {}
start_global = timeit.default_timer()
optimizer.zero_grad()
for epoch in range(0, 40, 1): # loop over the dataset multiple times
    
    running_loss = 0.0
    start = timeit.default_timer()
    
    for i in range(nb_batchs):
    #for i in np.random.permutation(nb_batchs):
        # get the inputs
        inputs = X_train[i*batch_size:(i+1)*batch_size,:]
        labels = y_train[i*batch_size:(i+1)*batch_size]

        # zero the parameter gradients
        mynet.zero_grad()
        # forward + backward + optimize
        outputs = mynet(inputs)
        loss = criterion(outputs, labels)
        #print(round(loss,4))
        grad = criterion.grad(outputs, labels)
        mynet.backward(grad)  
        mynet.step(optimizer)
        # print statistics
        running_loss += loss
        if i % 100 == 99: # print every 2000 mini-batches
            print('[{}, {}] - loss: {} | time: '.format(
                    epoch+1, i+1, round(running_loss / 100, 3)),
                    round(timeit.default_timer() - start, 2))
            running_loss = 0.0
            start = timeit.default_timer()
    if (epoch + 1) % 5 == 0:
        score_train, score_test = score(mynet)
        scores[epoch + 1] = (score_train, score_test)
        print('Accuracy -- Train: {} | Test: {}'.format(score_train, score_test))
        pickle.dump(mynet.parameters(),
                    open("parameters/{}/mynet_parameters_{}.p".format(experience_name, epoch+1), "wb"))

with open('{}/scores.json'.format(directory_path), 'w') as outfile:
    json.dump(scores, outfile)
print('Finished Training | {} seconds'.format(round(timeit.default_timer() - start_global, 2)))

#%%
#for i in range(0,0,20):
#    print(i)
i=0
parameters = pickle.load(open( "parameters/mynet_parameters_{}.p".format(i), "rb" ))
mynet.features._modules[0]._weight = parameters[0].copy()
mynet.features._modules[0]._bias = parameters[1].copy()
mynet.features._modules[3]._weight = parameters[2].copy()
mynet.features._modules[3]._bias = parameters[3].copy()
mynet.classifier._modules[0]._weight = parameters[4].copy()
mynet.classifier._modules[0]._bias = parameters[5].copy()

score(mynet)
#%%
import pandas as pd

N = X.shape[0]
X_features = np.empty((N, 400))
batch_size = 8
nb_batchs = int(N / batch_size)
for i in range(nb_batchs):
    inputs = X[i*batch_size:(i+1)*batch_size,:]
    outputs = mynet.features(inputs)
    X_features[i*batch_size:(i+1)*batch_size, :] = outputs.reshape(-1,16*5*5)

pd.DataFrame(X_features).to_csv('../data/Xtr_features_mycnn.csv',header=False, index=False)