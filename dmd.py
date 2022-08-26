import matplotlib.pyplot as plt
import numpy as np
import h5py
from pydmd import DMD

f = h5py.File('job/dyn-yaw-8wt-anime/output/dyn-yaw-8wt-anime_flowfield.h5','r')
u = f['u']
print(u.shape)

# dmd = DMD(exact=True,forward_backward=True)
# dmd.fit(u)
# reconstructed_data = dmd.reconstructed_data   
# eig = dmd.eigs
# mode_data = dmd.modes.T
# dyn = dmd.dynamics
# plt.figure()
# # plt.plot(np.real(dyn.T))
# plt.figure(figsize=(24,4))
# plt.imshow(np.real(np.reshape(-mode_data[0,:],[u.shape[1],u.shape[2]])).T)
# # # plt.colorbar()
# plt.savefig('test.png')