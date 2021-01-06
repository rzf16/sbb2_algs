import os
import sys
import json
import shutil
import statistics
import numpy as np
from DMM import DMM
from DataFrame import DataFrame
from IBCC import IBCC
from PriorityQueue import PriorityQ
from buffer import Buffer

RESULTS_PATH = "sbb_output"

class SingleSBB:

    def __init__(self, frame_path, vad_path, oad_path, tracking_path):
        self.frame_addr = frame_path
        self.frames = [os.path.join(frame_path, name) for name in sorted(os.listdir(frame_path))]
        self.params = json.load(open("params.json"))

        self.dmm = DMM(major_buffer_max=self.params["major_buffer_max"],
                       wait_buffer_max=self.params["wait_buffer_max"],
                       pre_buffer_min=self.params["pre_buffer_min"],
                       similarity_threshold=self.params["similarity_threshold"],
                       value_threshold=self.params["value_threshold"])
        self.precursor = Buffer()

        self.ibcc = IBCC(self.params["confusion_prior_init"],self.params["class_prob_prior_init"])

        self.vad_scores = np.load(vad_path)
        # Normalize VAD scores
        self.vad_scores -= np.min(self.vad_scores)
        self.vad_scores /= (np.max(self.vad_scores) - np.min(self.vad_scores))

        self.oad_scores = np.load(oad_path)
        self.tracking_output = np.load(tracking_path, allow_pickle=True)

        self.data_values = np.zeros(self.vad_scores.shape)
        for i in range(len(self.vad_scores)):
            vad_score = self.vad_scores[i]
            self.data_values[i] = self.calcValue(self.oad_scores[i], vad_score)

        self.buffer_index = 0

        self.priorityq = PriorityQ(self.params["max_memory_mb"], self.params["inflation_factor"],
                                   self.params["fifo"])

        try:
            os.makedirs(RESULTS_PATH)
        except:
            shutil.rmtree(RESULTS_PATH)
            os.makedirs(RESULTS_PATH)

    def calcValue(self, oad_scores, vad_score):
        # Calculates value from VAR and OAD
        if self.params["value_type"] == "ibcc":
            probs = {}
            probs["VAD"] = [1-vad_score, vad_score]
            probs["OAD"] = [oad_scores[0], 1-oad_scores[0]]
            value = self.ibcc.inferVB(probs)[1]
        elif self.params["value_type"] == "oad":
            value = np.sum(np.multiply(np.array(self.params["class_values"]), oad_scores[1:]))
        elif self.params["value_type"] == "hybrid":
            value = self.params["hybrid_value_alpha"] * np.sum(np.multiply(np.array(self.params["class_values"]), oad_scores[1:]))
            value += self.params["hybrid_value_beta"] * vad_score
        else:
            value = vad_score

        return value

    def run(self):
        for i, img_ptr in enumerate(self.frames):
            print("Frame {}".format(i))

            frame = DataFrame(img_ptr, i, self.data_values[i], self.vad_scores[i], self.tracking_output[i])
            # Fill initial precursor before starting DMM
            if i < self.dmm.PRE_BUFFER_MIN:
                self.precursor.append(frame)
                print("Precursor size: " + str(self.precursor.size()) + '\n')
                continue
            elif i == self.dmm.PRE_BUFFER_MIN:
                print("Starting DMM")
                self.dmm.start(self.precursor)

            # Loop DMM until the frame is resolved
            while True:
                if self.dmm.state == DMM.State.TERMINATE:
                    self.pushBuffer()
                    input = self.dmm.reset(frame)
                else:
                    sim = self.dmm.major_buffer.calcSimilarity(frame.objects)
                    input = self.dmm.compute_input(frame, sim)
                action = self.dmm.update_state(input)
                resolved = self.dmm.run_action(frame, action)
                if resolved:
                    break

        if self.dmm.started:
            self.dmm.major_buffer.extend(self.dmm.wait_buffer)
            if self.dmm.major_buffer.size() > 0:
                self.pushBuffer()

    def pushBuffer(self):
        buffer = self.dmm.major_buffer
        buffer.setBufferIndex(self.buffer_index)
        self.buffer_index += 1

        buffer.filterValue(self.params["filter_sigma"])
        buffer.generateDecision(self.params["eta"], self.params["zeta"])
        buffer.fakeCompress()

        self.priorityq.fakePush(buffer, RESULTS_PATH)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Error! Usage is: python3 sbb.py <frames_dir> <vad_scores> <oad_scores> <obj_tracking_output>")
        exit()
    sbb = SingleSBB(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    sbb.run()
