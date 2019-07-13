import pickle
from heatutils import heatmap
import matplotlib.pyplot as plt

data = pickle.load(open('calculated_data/test2.pk','rb'))
data.keys()


X = data['X']
Y = data['Y']
T = data['data']
dx = data['dx']
dt = data['dt']

for i in range(T.shape[2]-1):
    fig,ax = plt.subplots()
    heatmap(X, Y, T, dx, dt, time_elapsed=i, ax=ax)
    fig.savefig(f'img_seq/img_{i:04d}')
    plt.close('all')

os.system('echo %USERNAME%')

a = 1
f'{a:04d}'

import os
for filename,seq in zip(os.listdir('img_seq'), range(10)):
    os.system(f'ren img_seq/{filename} {seq:04d}.png')

os.system('ren img_seq/img_0001.png 0001.png')
!ren img_seq/img_0001.png img_seq/0001.png
pwd
