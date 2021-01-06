import os
from heapq import *

class PriorityQ:
    # Class for prioritized data recording

    def __init__(self, max_memory_mb, inflation_factor, fifo=False):
        self.data = []
        self.max_memory = max_memory_mb * 1024 * 1024
        self.inflation_factor = inflation_factor
        self.fifo = fifo
        self.cost = 0
        self.value = 0
    
    def fakePush(self, buffer, path):
        # Pushes a buffer into the heapq
        buffer.setBufferValue(self.inflation_factor)

        if self.fifo:
            buffer.fakeDump(path, self.fifo)
            self.data.append(buffer)
        else:
            # Discard if there's no space and value is less than min and buffer is not first
            if len(self.data) == 0 and buffer.totalCost() > self.max_memory:
                return
            elif len(self.data) != 0 and buffer.buffer_value < self.data[0].buffer_value and \
               self.cost + buffer.totalCost() > self.max_memory:
                return
            buffer.fakeDump(path, self.fifo)
            heappush(self.data, buffer)
        
        self.cost += buffer.buffer_cost
        self.value += buffer.totalValue()

        print("Buffer " + str(buffer.buffer_index) + " with total value " +
              str(buffer.totalValue()) + " and cost " + str(buffer.buffer_cost) +
              " pushed!")

        # Pop buffers if needed
        while self.cost >= self.max_memory:
            if self.fifo:
                removed_buffer = self.data.pop(0)
                removed_buffer.fakeDrop()
                self.cost -= removed_buffer.buffer_cost
                self.value -= removed_buffer.buffer_value
            else:
                removed_buffer = heappop(self.data)
                removed_buffer.fakeDrop()
                self.cost -= removed_buffer.buffer_cost
                self.value -= removed_buffer.buffer_value
            print("Buffer " + str(removed_buffer.buffer_index) + " with total value " +
              str(removed_buffer.totalValue()) + " and cost " + str(removed_buffer.buffer_cost) +
              " popped!")
