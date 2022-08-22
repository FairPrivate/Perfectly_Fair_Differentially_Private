import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import random
from bisect import bisect_left
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import rv_discrete
from scipy import interpolate
import warnings



plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"

epsilons = [0.05*x for x in range(400)]
epsilons[0] = 0.0001
theta =pd.read_csv('adult_wb_same_equal_opportunity_1.csv')


epsilons = [0.015*x for x in range(1334)]
epsilons[0] = 0.0001
theta =pd.read_csv('adult_wb_15_1_equal_opportunity_1.csv')




epsilons = [0.02*x for x in range(1000)]
epsilons[0] = 0.0001
theta =pd.read_csv('adult_wb_2_1_equal_opportunity_1.csv')

epsilons = [0.03*x for x in range(1000)]
epsilons[0] = 0.0001
theta =pd.read_csv('adult_wb_3_1_equal_opportunity_1.csv')

epsilons = [0.04*x for x in range(1000)]
epsilons[0] = 0.0001
theta =pd.read_csv('adult_wb_4_1_equal_opportunity_1.csv')

epsilons = [0.05*x for x in range(1000)]
epsilons[0] = 0.0001
theta =pd.read_csv('adult_wb_5_1_equal_opportunity_1.csv')

epsilons = [0.035*x for x in range(2000)]
epsilons[0] = 0.0001
theta =pd.read_csv('adult_wb_35_1_equal_opportunity_1.csv')

#epsilons = [0.042*x for x in range(2000)]
#epsilons[0] = 0.0001
#theta =pd.read_csv('adult_wb_42_1_equal_opportunity_1.csv')

epsilons = [0.01*x for x in range(2000)]
epsilons[0] = 0.0001
theta =pd.read_csv('adult_wb_1_11_equal_opportunity_1.csv')

epsilons = [0.0105*x for x in range(2000)]
epsilons[0] = 0.0001
theta =pd.read_csv('adult_wb_105_1_equal_opportunity_1.csv')

epsilons = [0.01*x for x in range(2000)]
epsilons[0] = 0.0001
theta =pd.read_csv('adult_wb_1_105_equal_opportunity_1.csv')


epsilons = [0.01*x for x in range(2000)]
epsilons[0] = 0.0001
theta =pd.read_csv('adult_wb_11_1_equal_opportunity_1.csv')


theta = theta.drop(columns='Unnamed: 0')
theta1 = np.array(theta)
theta = []
for each in theta1:
    theta.append(each[0])
z = np.polyfit(epsilons, theta, 5)
p = np.poly1d(z)
with warnings.catch_warnings():
    warnings.simplefilter('ignore', np.RankWarning)
    p30 = np.poly1d(np.polyfit(epsilons, theta, 4))


#plt.title(r'Accuracy as a function of $\epsilon$')
plt.title(r'Fairness as a function of $\epsilon$')
#plt.ylabel('Accuracy $(\Theta)$')
plt.ylabel('Fairness $(\gamma)$')
plt.xlabel(r'Privacy loss $\epsilon_{1}$ ($\frac{\epsilon_{1}}{\epsilon_{2}}$=1)')
xp = np.linspace(0, 20, 2000)
plt.plot(epsilons, theta, color='lightgray')
_ = plt.plot( xp, p(xp), '-', color='darkred')
plt.grid()
plt.show()
#print ('hi')

