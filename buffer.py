import os
import statistics
import json
import numpy as np
import copy
from LBO import single_optimize

class Buffer:
    # Class for a buffer of data frames

    def __init__(self, frame_list=None, msg=None):
        self.data_ptrs = []
        self.index = []
        self.value = []
        self.cost = []
        self.anomaly_score = []
        self.objects = []

    def extend(self, new_buffer, extend_from=None, extend_until=None):
        # Appends the frames of new_buffer
        if not new_buffer or not new_buffer.index[extend_from:extend_until]:
            return

        self.data_ptrs.extend(new_buffer.data_ptrs[extend_from:extend_until])
        self.index.extend(new_buffer.index[extend_from:extend_until])
        self.value.extend(new_buffer.value[extend_from:extend_until])
        self.cost.extend(new_buffer.cost[extend_from:extend_until])
        self.anomaly_score.extend(new_buffer.anomaly_score[extend_from:extend_until])
        self.objects.extend(new_buffer.objects[extend_from:extend_until])
        self.objects = list(set(self.objects))

    def copy(self, new_buffer, copy_from=None, copy_until=None):
        # Copies the frames of new_buffer
        if not new_buffer or not new_buffer.index[copy_from:copy_until]:
            self.data_ptrs = []
            self.index = []
            self.value = []
            self.cost = []
            self.anomaly_score = []
            self.objects = []
            return

        self.data_ptrs = new_buffer.data_ptrs[copy_from:copy_until]
        self.index = new_buffer.index[copy_from:copy_until]
        self.value = new_buffer.value[copy_from:copy_until]
        self.cost = new_buffer.cost[copy_from:copy_until]
        self.anomaly_score = new_buffer.anomaly_score[copy_from:copy_until]
        self.objects = new_buffer.objects[copy_from:copy_until]

    def split(self, split_from=None, split_until=None):
        # Splits the buffer
        if not self.index[split_from:split_until]:
            self.data_ptrs = []
            self.index = []
            self.value = []
            self.cost = []
            self.anomaly_score = []
            self.objects = []
            return

        self.data_ptrs = self.data_ptrs[split_from:split_until]
        self.index = self.index[split_from:split_until]
        self.value = self.value[split_from:split_until]
        self.cost = self.cost[split_from:split_until]
        self.anomaly_score = self.anomaly_score[split_from:split_until]
        self.objects = self.objects[split_from:split_until]
        self.objects = list(set(self.objects))

    def append(self, new_frame):
        # Appends new_frame
        self.data_ptrs.append(new_frame.data_ptr)
        self.index.append(new_frame.index)
        self.value.append(new_frame.value)
        self.cost.append(new_frame.cost)
        self.anomaly_score.append(new_frame.anomaly_score)
        self.objects += new_frame.objects
        self.objects = list(set(self.objects))

    def calcSimilarity(self, track_ids):
        if len(track_ids) == 0:
            return 0.0
        intersecting = len(list(set(self.objects) & set(track_ids)))
        return intersecting / len(track_ids)

    def filterValue(self, sigma):
        # Applies gaussian filters to values
        N = len(self.value)
        filtered_value = copy.deepcopy(self.value)
        for i, value in enumerate(self.value):
            if i > 0:
                if self.value[i] > self.value[i-1]:
                    mu = i
                    start_idx = max(mu - 3*sigma, 0)
                    a = self.value[i] - self.value[i-1]
                    for j in range(start_idx, mu):
                        filtered_value[j] = max(filtered_value[j], gaussian(j,a,mu,sigma))
                elif self.value[i] < self.value[i-1]:
                    mu = i
                    end_idx = min(mu + 3*sigma, N)
                    a = self.value[i-1] - self.value[i]
                    for j in range(mu, end_idx):
                        filtered_value[j] = max(filtered_value[j], gaussian(j,a,mu,sigma))

        self.value = filtered_value

    def generateDecision(self, eta, zeta):
        # Generates the decisions for the buffer using LBO
        self.decision = []
        for cost, value in zip(self.cost, self.value):
            self.decision.append(single_optimize(cost, value, eta, zeta))

    def fakeCompress(self):
        # Compress buffer data based on LBO decision
        a = 0.1083869
        b = 0.99837249
        c = 0.02535
        for i, img_ptr in enumerate(self.data_ptrs):
            phi = -a*np.log2(1 - b*self.decision[i]) + c
            self.cost[i] = os.path.getsize(img_ptr) * phi
            self.value[i] *= self.decision[i]

    def fakeDump(self, path, fifo):
        # Dumps log to json
        name = "fifo_" if fifo else "priority_"
        name += "buffer" + str(self.buffer_index)

        self.buffer_cost = self.totalCost()

        self.log_addr = os.path.join(path, name + "_log.json")
        log = {"value":self.value, "cost":self.cost, "frame":self.index, "decision":self.decision}
        with open(self.log_addr, 'w') as log_out:
            json.dump(log, log_out)

    def fakeDrop(self):
        # Drops the buffer
        os.remove(self.log_addr)

    def maxValue(self):
        return max(self.value)

    def totalCost(self):
        return sum(self.cost)

    def totalValue(self):
        return sum(self.value)

    def setBufferValue(self, inflation_factor):
        self.buffer_value = (inflation_factor ** self.buffer_index) * max(self.value)

    def setBufferIndex(self, index):
        self.buffer_index = index

    def size(self):
        return len(self.index)

    def __lt__(self, other):
        # Less than comparator for priorityq
        return self.buffer_value < other.buffer_value

def gaussian(x,a,mu,sigma):
    return a*np.exp(-((x-mu)**2)/(2*sigma**2))
