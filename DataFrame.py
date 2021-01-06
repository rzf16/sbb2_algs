import os

class DataFrame:

    def __init__(self, img_ptr, index, value, anomaly_score, track_ids):
        self.data_ptr = img_ptr
        self.index = index
        self.value = value
        self.anomaly_score = anomaly_score
        self.cost = os.path.getsize(img_ptr)
        self.objects = track_ids