import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

u_mean = np.load('../result/warmup/u_mean.npy')
u2_mean = np.load('../result/warmup/u2_mean.npy')
u_profile = np.mean(np.mean(u_mean,axis=0),axis=0)
u2_profile = np.mean(np.mean(u2_mean,axis=0),axis=0)
# z = linspace()
# print(u_mean.shape)
figure(num=None, figsize=(12, 3), dpi=100, facecolor='w', edgecolor='k')
# log_profle = 1.0/0.4*np.log((z/config['zo']))
# log_profle = 0.25/0.4*np.log((z/(config['zo'])))
# plt.semilogx(u_profile,'o')
# plt.semilogx(z/0.3975,log_profle,'k--')
plt.subplot(121)
plt.plot(u_profile,label='sp')
plt.xlabel('$\overline{u}/u_*$')
plt.ylabel('$z/H$')

plt.subplot(122)
plt.plot(u2_profile,label='sp')
plt.xlabel('$\overline{u}/u_*$')
plt.ylabel('$z/H$')
# plt.legend()

plt.savefig('profile.png')