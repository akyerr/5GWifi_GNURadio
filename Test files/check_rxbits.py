import pickle
import matplotlib.pyplot as plt
import numpy as np

case = 0

pickle_dir = '/home/utsa/python_projects/5GWifi_GNURadio/Data files'
with open('{}/in_bits_{}.pckl'.format(pickle_dir, case)) as txbits:
    tx_binary_info = pickle.load(txbits)

with open('{}/rx_bits_{}.pckl'.format(pickle_dir, case)) as rxbits:
    rx_binary_info = pickle.load(rxbits)


print(tx_binary_info.shape)
print(rx_binary_info.shape)

# plt.plot(tx_binary_info[0, :])
win_size = 500
num_windows = int(np.ceil(rx_binary_info.shape[0]/win_size))

print(num_windows)

for i in range(num_windows):
    bits = rx_binary_info[i*win_size: (i+1)*win_size]

    plt.plot(bits)
    plt.show()