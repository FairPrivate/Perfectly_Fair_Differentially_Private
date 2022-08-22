import pandas as pd
import numpy as np
import random
from bisect import bisect_left


number_of_samples = 100000
alpha_a = 0.9634 # \Pr[A='White+Hispanic'] = 1 - \Pr[A='Asian']
Y0a = 0.3099 # \Pr[Y=0|A='White+Hispanic']
Y0b = 0.1932 # \Pr[Y=0|A='Asian']

CDF = pd.read_csv("CDF.csv")
rho = CDF['Score']/100

P00 = pd.read_csv('P00_wha.csv', header=None) # \Pr[R=\rho|A='White+Hispanic', Y=0]
P01 = pd.read_csv('P01_wha.csv', header=None) # \Pr[R=\rho|A='White+Hispanic', Y=1]
P10 = pd.read_csv('P10_wha.csv', header=None) # \Pr[R=\rho|A='Asian', Y=0]
P11 = pd.read_csv('P11_wha.csv', header=None) # \Pr[R=\rho|A='Asian', Y=1]

# Assume $A=0$ is equal to $A='White+Hispanic' and $A=1$ is equal to $A='Asian'$

P00 = P00.to_numpy()
P01 = P01.to_numpy()
P10 = P10.to_numpy()
P11 = P11.to_numpy()

print ('hi')


a_y1_times = []
b_y1_times = []
a_y0_times = []
b_y0_times = []

Q_a1 = []
Q_b1 = []
Q_a0 = []
Q_b0 = []
for i in range (number_of_samples):
    randa1 = random.random()
    posa1 = bisect_left(np.cumsum(P01), randa1)
    Q_a1.append(rho[posa1])  # Samples from Pr(R=\rho|A=0, Y=1)

    randb1 = random.random()
    posb1 = bisect_left(np.cumsum(P11), randb1)
    Q_b1.append(rho[posb1])  # Samples from Pr(R=\rho|A=1, Y=1)

    randa0 = random.random()
    posa0 = bisect_left(np.cumsum(P00), randa0)
    if posa0 == 198:
        posa0 = 197
    Q_a0.append(rho[posa0])  # Samples from Pr(R=\rho|A=0, Y=0)

    randb0 = random.random()
    posb0 = bisect_left(np.cumsum(P10), randb0)
    Q_b0.append(rho[posb0])  # Samples from Pr(R=\rho|A=1, Y=0)

rand_numa = [random.random() for o in range(number_of_samples)]
temp0a = []  # Index of A=0 and Y=1
for j in range(number_of_samples):
    if rand_numa[j] < Y0a:
        temp0a.append(0)
    else:
        temp0a.append(1)
Q_a_first = [x * y for (x, y) in zip(temp0a, Q_a1)]  # Pr(R=\rho|A=0, Y=1)
temp0a_reversed = []  # Index of A=0 and Y=0
for each in temp0a:
    if each == 0:
        temp0a_reversed.append(1)
    else:
        temp0a_reversed.append(0)
Q_a_second = [x * y for (x, y) in zip(temp0a_reversed, Q_a0)]  # Pr(R=\rho|A=0, Y=0)
Q_a = [x + y for (x, y) in zip(Q_a_first, Q_a_second)]

rand_numb = [random.random() for o in range(number_of_samples)]
temp0b = []  # Index of A=1 and Y=1
for k in range(number_of_samples):
    if rand_numb[k] < Y0b:
        temp0b.append(0)
    else:
        temp0b.append(1)
Q_b_first = [x * y for (x, y) in zip(temp0b, Q_b1)]  # Pr(R=r|A=\rho, Y=1)
temp0b_reversed = []
for each in temp0b:
    if each == 0:
        temp0b_reversed.append(1)
    else:
        temp0b_reversed.append(0)
Q_b_second = [x * y for (x, y) in zip(temp0b_reversed, Q_b0)]  # Pr(R=\rho|A=1, Y=0)
Q_b = [x + y for (x, y) in zip(Q_b_first, Q_b_second)]

temp = []  # Index of A=0
rand_num = [random.random() for l in range(number_of_samples)]
for t in range(number_of_samples):
    if rand_num[t] < alpha_a:
        temp.append(1)
    else:
        temp.append(0)

temp_reversed = []  # Index of A=1
for each in temp:
    if each == 0:
        temp_reversed.append(1)
    else:
        temp_reversed.append(0)

Qual_first = [x * y for (x, y) in zip(temp, Q_a)]
Qual_second = [x * y for (x, y) in zip(temp_reversed, Q_b)]
Qual = [x + y for (x, y) in zip(Qual_first, Qual_second)]

temp0a_np = np.array(temp0a)
temp0b_np = np.array(temp0b)
temp_np = np.array(temp)
temp_reversed_np = np.array(temp_reversed)

all_from_a = np.where(temp_np == 1)[0]
all_a_1s = np.where(temp0a_np == 1)[0]
all_a_0s = np.where(temp0a_np == 0)[0]
a_y1s = list(set(all_from_a) & set(all_a_1s))
a_y0s = list(set(all_from_a) & set(all_a_0s))

all_from_b = np.where(temp_np == 0)[0]
all_b_1s = np.where(temp0b_np == 1)[0]
all_b_0s = np.where(temp0b_np == 0)[0]
b_y1s = list(set(all_from_b) & set(all_b_1s))
b_y0s = list(set(all_from_b) & set(all_b_0s))

labels = []
for ii in range (number_of_samples):
    if ii in b_y1s:
        labels.append(1)
    elif ii in a_y1s:
        labels.append(1)
    else:
        labels.append(0)


db = pd.DataFrame()
db_minority = pd.DataFrame()
Qual = np.asarray(Qual)
labels = np.asarray(labels)
temp_reversed = np.asarray(temp_reversed)
db['score'] = Qual
db['protected_class'] = temp_reversed
db['label'] = labels

db.to_csv('wha_generated_samples_corrected.csv')
print('hi')
