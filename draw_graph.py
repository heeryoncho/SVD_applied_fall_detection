#https://matplotlib.org/gallery/lines_bars_and_markers/barh.html

import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))

approaches = ("Raw", "SMV", "SVD", "KPCA", "SPCA",
              "Raw+SMV", "Raw+SVD", "Raw+KPCA", "Raw+SPCA",
              "SMV+SVD", "SMV+KPCA", "SMV+SPCA", "SVD+KPCA", "SVD+SPCA", "KPCA+SPCA")

y_pos = np.arange(len(approaches))

#------------------------------------------------------
# UniMiB SHAR Dataset

ax1 = axs[0]

x1 = np.array([0.7436671499, 0.724111759, 0.7395555037, 0.7356039029, 0.7317909379, 0.748059282, 0.740205883, 0.7508165133, 0.7465736601, 0.7565017314, 0.7490259557, 0.7188600793, 0.7462046385, 0.7428297124, 0.7338684535])
#sd1 = np.array([0.080657586, 0.0822238994, 0.0842176004, 0.0713773462, 0.0741925397, 0.0893718049, 0.0877541738, 0.077351907, 0.0856656619, 0.0802861061, 0.0813236219, 0.0888540084, 0.0774471949, 0.0805119923, 0.0714214762])

ax1.axvline(x=0.7565017314, linewidth=0.5, color='red')
ax1.barh(y_pos, x1, align='center', color='gray')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(approaches)
ax1.invert_yaxis()
ax1.set_title('UniMiB SHAR Dataset')

ax1.set_xlim([0.71, 0.76])


#------------------------------------------------------
# SisFall Dataset

ax2 = axs[1]

x2 = np.array([0.6136972318, 0.4689147419, 0.5023712984, 0.4831914868, 0.4620598583, 0.6050191468, 0.6307888246, 0.6251302272, 0.6153451674, 0.5551476197, 0.54394793, 0.5284114259, 0.4855208808, 0.4934170551, 0.4921940791])
#sd2 = np.array([0.0747104254, 0.0530682598, 0.0657771, 0.0435936791, 0.0567289541, 0.0667974181, 0.0687332269, 0.0593793767, 0.0539394972, 0.049810905, 0.0452937096, 0.0416791154, 0.0570768792, 0.0602386223, 0.0520705879])

ax2.axvline(x=0.6307888246, linewidth=0.5, color='red')
ax2.barh(y_pos, x2, align='center', color='gray')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(approaches)
ax2.invert_yaxis()
ax2.set_title('SisFall Dataset')

ax2.set_xlim([0.4, 0.65])


#------------------------------------------------------
# UMAFall Dataset

ax3 = axs[2]

x3 = np.array([0.576531764, 0.4404125892, 0.5407785396, 0.5156050704, 0.4574088949, 0.5811311598, 0.6468791109, 0.6405206039, 0.5893623604, 0.5544869222, 0.5200556206, 0.5192051708, 0.5179853979, 0.5381716054, 0.5223146814])
#sd3 = np.array([0.1310814025, 0.1147977376, 0.0696986542, 0.0716025756, 0.0869818494, 0.108312508, 0.0828349305, 0.0844756884, 0.1125472913, 0.0775702745, 0.049160488, 0.0837790097, 0.0892150317, 0.0851352497, 0.0749019969])

ax3.axvline(x=0.6468791109, linewidth=0.5, color='red')
ax3.barh(y_pos, x3, align='center', color='gray')
ax3.set_yticks(y_pos)
ax3.set_yticklabels(approaches)
ax3.invert_yaxis()
ax3.set_title('UMAFall Dataset')

ax3.set_xlim([0.4, 0.65])

plt.subplots_adjust(left=0.15, bottom=0.17, wspace = 0.5)

fig.text(0.5, 0.01, 'Leave-One-Subject-Out Cross Validation Accuracy', ha='center', fontsize=12)

#plt.show()

plt.savefig("fig/LOSO_Accuracy.pdf", bbox_inches="tight", pad_inches=0)
