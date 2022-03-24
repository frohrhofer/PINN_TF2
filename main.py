'''
This code snippet trains a single PINN instance
with settings specified in the config file.
After training, learning curves and final
prediction are plotted (saved to log folder).
'''
import numpy as np
import matplotlib.pyplot as plt

from configs.config_loader import load_config
from pinn.neural_net import PhysicsInformedNN


################################
# Initialization
################################

# Update and load config file
config_update = {'version': 'run_0',
                 'seed': 0,
                 'y0': 0.5,
                 'T': 10}
config = load_config('configs/default.yaml', config_update, verbose=True)

# Initialize PINN instance
pinn = PhysicsInformedNN(config, verbose=True)

################################
# Training
################################

pinn.train()

################################
# Plotting
################################

# get logs file
log = pinn.callback.log

fig, axes = plt.subplots(1, 2, figsize=(6, 2))

# learning curves
n_epochs, freq_log = log['n_epochs'], log['freq_log']
epochs = np.arange(0, n_epochs, freq_log)

axes[0].plot(epochs, log['loss_F'])
axes[0].set_yscale('log')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel(r'$L$')

# PINN prediction
t_line, y_pred = log['t_line'], log['y_pred']
axes[1].set_xlabel(r'$t$')
axes[1].set_ylabel(r'$y(t)$')

# plot stable and unstable fixed points
axes[1].plot(t_line, y_pred)
for y_star in [-1, 0, 1]:
    axes[1].axhline(y_star, ls='--', lw=1, c='black')

plt.tight_layout()
fig_file = pinn.callback.model_path.joinpath('out.png')
plt.savefig(fig_file)
plt.show()
