
import numpy as np


def Encurvadura(x):

	'''Encurvadura'''
	l = x[0] # 0.4 # heigth [m]
	e = x[1] # 0.001 # force eccentricity[m], 0 if none 
	P = 10 # force [N]
	E = 117e9 # elastcity module ·[Pa] (copper)
	r = 1/100 # [m]
	I = (np.pi/4)*r**4
	k = np.sqrt(np.abs(P)/(E*I))

	flexa = e*(1-np.cos(k*l))/np.cos(k*l) # flexion [m]
	Mf_max = P*(e+flexa) # maximum moment of flextion
	W = l*(np.pi*r**2) # volume [mn]
	tension_max = P/(np.pi*r**2) + Mf_max/W

	return flexa, tension_max

