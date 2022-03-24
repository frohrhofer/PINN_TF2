import json
import numpy as np

from pathlib import Path


class CustomCallback():
    '''
    provides custom callback that stores, prints and saves
    training logs in json-format
    '''
    def __init__(self, config):

        # determines digits for 'fancy' log printing
        self.n_epochs = config['n_epochs']
        self.digits = int(np.log10(self.n_epochs)+1)

        # create log from config file
        self.log = config.copy()

        # log and print frequencies
        self.freq_log = config['freq_log']
        self.freq_print = config['freq_print']

        # keys to be printed
        self.keys_print = config['keys_print']

        # create model folder
        version = config['version']
        self.model_path = Path('logs', version)
        self.model_path.mkdir(parents=True, exist_ok=True)

        # create log file path
        seed = self.log['seed']
        self.log_file = self.model_path.joinpath(f'log_{seed}.json')

    def write_logs(self, logs, epoch):
        '''
        is called during network training
        stores/prints training logs
        '''
        # store training logs
        if (epoch % self.freq_log) == 0:
            # exception errors catch different data formats
            for key, item in logs.items():
                # append if list already exists
                try:
                    self.log[key].append(item.numpy().astype(np.float64))
                # create list otherwise
                except KeyError:
                    try:
                        self.log[key] = [item.numpy().astype(np.float64)]
                    # if list is given
                    except AttributeError:
                        self.log[key] = item

        # print training logs
        if (epoch % self.freq_print) == 0:

            s = f"{epoch:{self.digits}}/{self.n_epochs}"
            for key in self.keys_print:
                try:
                    s += f" | {key}: {logs[key]:2.2e}"
                except KeyError:
                    pass
            print(s)

    def save_logs(self):
        '''
        saves recorded training logs in json-format
        '''
        with open(self.log_file, "w") as f:
            json.dump(self.log, f, indent=2)

        print("*** logs saved ***")
