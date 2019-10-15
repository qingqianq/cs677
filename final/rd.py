import numpy as np
m = input('please input the number of features: ')
n = input('input the number of samples: ')
a = np.random.random((int(n),int(m)))
np.savetxt('kmeans_data.txt', a, delimiter = ',', header = '@data',comments = '')
