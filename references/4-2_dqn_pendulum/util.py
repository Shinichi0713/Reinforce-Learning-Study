import csv
from datetime import datetime

import numpy as np


def now_str(str_format='%Y%m%d%H%M'):
    return datetime.now().strftime(str_format)


def idx2mask(idx, max_size):
    mask = np.zeros(max_size)
    mask[idx] = 1.0
    return mask


class RecordHistory:
    def __init__(self, csv_path, header):
        self.csv_path = csv_path
        self.header = header

    def generate_csv(self):
        with open(self.csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.header)

    def add_histry(self, history):
        history_list = [history[key] for key in self.header]
        with open(self.csv_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(history_list)

    def add_list(self, array):
        with open(self.csv_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(array)
