#!/home/rui/anaconda3/envs/tf-gpu/bin/python

import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

from func import Encurvadura


'''Encurvadura Parametros [m]'''
min_heigth = 0.1
max_heigth = 2
min_eccentricity = 0.1/1000
max_eccentricity = 0.5
N_discretization = 100

heigth = np.arange(min_heigth, max_heigth, 
	(max_heigth-min_heigth)/N_discretization)
eccentricity = np.arange(min_eccentricity, max_eccentricity, 
	(max_eccentricity-min_eccentricity)/N_discretization)

flexa, tension_max = np.asarray(Encurvadura(
	[heigth, eccentricity]))

## DATASET numpy to Torch
d_data = np.column_stack(
	[heigth, eccentricity, flexa, tension_max])
np.random.shuffle(d_data)

TRAIN_PERCENT = 0.8
TRAIN_BATCH_SIZE = 10
TEST_SIZE = int(np.round(d_data.shape[0]*(1-TRAIN_PERCENT)))
TRAIN_SIZE = int(np.round(d_data.shape[0]*TRAIN_PERCENT))

train_data = torch.Tensor(
	d_data[0:TRAIN_SIZE,0:2]
	)
train_data_label = torch.Tensor(
	d_data[0:TRAIN_SIZE,2:4]
	)
test_data = torch.Tensor(
	d_data[0:TEST_SIZE,0:2]
	)
test_data_label = torch.Tensor(
	d_data[0:TEST_SIZE,2:4]
	)

## DATASET Torch learning base
train_dataset = torch.utils.data.TensorDataset(
	train_data,train_data_label)
test_dataset = torch.utils.data.TensorDataset(
	test_data,test_data_label)
train_loader = torch.utils.data.DataLoader(
	train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(
	test_data, batch_size=1, shuffle=False)

## Learn Model

model = torch.nn.Sequential(
	torch.nn.Linear(2, 64),
	torch.nn.Linear(64, 64),
	torch.nn.Linear(64, 64),
	torch.nn.Linear(64, 2),
	)

loss_fn = torch.nn.MSELoss(size_average=False)
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), 
	lr=learning_rate)
pl =[[] for _ in range(3)]

for t in range(100):
	for i, (data, target) in enumerate(train_loader):
		x = Variable(data)
		y = Variable(target)
		y_pred = model(x)
		loss = loss_fn(y_pred, y) ##
		print(t)
		print(loss.data)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	pl[1].append(loss.data)
	pl[0].append(t)
accurrancy= 0

x = Variable(test_data)
y = Variable(test_data_label, requires_grad=False)

y_pred = model(x)

for i in range(TEST_SIZE):
	d = y_pred[i,:]
	valuesx, indicesx = torch.max(d, 0)
	indices2 = np.argmax(test_data_label[i, :])
	indices1 =  indicesx.data.numpy()[0]
	print("predicted %f label %f" % (indices1,indices2  ))
	if (indices1==indices2):
		accurrancy += 1

print("Correct Predictions: %d " % (accurrancy))
print("Incorrect Predictions:  %d " 
	% (TEST_SIZE))
print("Success Rate:    %f" 
	% (accurrancy/TEST_SIZE*100))
print("Error Rate   %f" 
	%  (100-(accurrancy/TEST_SIZE*100)))

fig0 = plt.figure()
ax0 = fig0.add_subplot(111)
ax0.grid(True)
gridlines = ax0.get_xgridlines() + ax0.get_ygridlines()
plt.yscale('linear')
plt.xscale('linear')

print(pl)
for line in gridlines:
    line.set_linestyle('-.')
plt.plot(pl[0], pl[1], 'bs', pl[0], pl[1],markersize=2)
plt.ylabel('Quadratic Error')
plt.xlabel('Iteration')

blue_patch = mpatches.Patch(color='blue', label='Error')
plt.legend(handles=[blue_patch])
plt.show()