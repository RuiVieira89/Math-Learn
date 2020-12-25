#!/home/rui/anaconda3/envs/tf-gpu/bin/python

import numpy as np
import torch

from func import Encurvadura

def main():

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

	data = np.column_stack(
		[heigth, eccentricity, flexa, tension_max])
	np.random.shuffle(data)

	TRAIN_PERCENT = 0.8
	train_data = torch.Tensor(
		data[0:int(data.shape[0]*TRAIN_PERCENT),0:2]
		)
	train_data_label = torch.Tensor(
		data[0:int(data.shape[0]*TRAIN_PERCENT),2:4]
		)
	test_data = torch.Tensor(
		data[0:int(data.shape[0]*(1-TRAIN_PERCENT)),0:2]
		)
	test_data_label = torch.Tensor(
		data[0:int(data.shape[0]*(1-TRAIN_PERCENT)),2:4]
		)




if __name__ == '__main__':

	main()