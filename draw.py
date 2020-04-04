import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np
plt.style.use("ggplot")

beta = [0, 1, 2, 3, 4, 5]

ABG_uh_beta = [0.7278, 0.7444, 0.7267, 0.7788, 0.7833, 0.7750]
HABG_uh_beta = [0.7361, 0.7389, 0.7722, 0.7778, 0.8000, 0.7750]
ABG_hu_beta = [0.8441, 0.8476, 0.8476, 0.8511, 0.8441, 0.8371]
HABG_hu_beta = [0.8144, 0.8126, 0.8267, 0.8249, 0.8476, 0.8494]

gamma_uh = [0, 0.005, 0.01, 0.015, 0.03]
gamma_hu = [0, 0.05, 0.1, 0.15, 0.3]

ABG_uh_gamma = [0.7833, 0.7833, 0.7917, 0.7833, 0.7833]
HABG_uh_gamma = [0.7806, 0.7778, 0.800, 0.7861, 0.7944]
ABG_hu_gamma = [0.8126, 0.8424, 0.8511, 0.8441, 0.8284]
HABG_hu_gamma = [0.8074, 0.8284, 0.8494, 0.8004, 0.8021]

fig, axs = plt.subplots(ncols=2, nrows=2)
ax1, ax2, ax3, ax4 = axs.ravel()

ax1.plot(beta, ABG_uh_beta, label='ABG')
ax1.plot(beta, HABG_uh_beta, label='HABG')
ax1.set_ylabel("Accuracy")
ax1.set_xlabel(r"$\beta$")
ax1.set_ylim(0.6, 0.85)
plt.tight_layout()
ax1.legend()

ax2.plot(gamma_uh, ABG_uh_gamma, label='ABG')
ax2.plot(gamma_uh, HABG_uh_gamma, label='HABG')
ax2.set_ylabel("Accuracy")
ax2.set_xlabel(r"$\gamma$")
ax2.set_ylim(0.6, 0.85)
# ax2.margins(0)
ax2.legend()

ax3.plot(beta, ABG_hu_beta, label='ABG')
ax3.plot(beta, HABG_hu_beta, label='HABG')
ax3.set_ylabel("Accuracy")
ax3.set_xlabel(r"$\beta$")
ax3.set_ylim(0.65, 0.9)
ax3.legend()

ax4.plot(gamma_hu, ABG_hu_gamma, label='ABG')
ax4.plot(gamma_hu, HABG_hu_gamma, label='HABG')
ax4.set_ylabel("Accuracy")
ax4.set_xlabel(r"$\gamma$")
ax4.set_ylim(0.65, 0.9)
ax4.legend()
# ax4.margins(0)


plt.show()