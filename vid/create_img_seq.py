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
