# Smart Black Box
The Smart Black Box (SBB) is an intelligent data recorder for video which prioritizes the storage and quality of notable data and compresses or discards insignificant data. An offline SBB is implemented here.

## Overview
The diagram below displays the SBB pipeline.
![SBB Pipeline](https://github.com/rzf16/sbb2_algs/blob/main/Diagram.jpg?raw=true)
In this offline implementation, the outputs of the orange boxes have already been computed and can be provided directly to the SBB. Anomaly detection is performed using [TAD](https://arxiv.org/abs/1903.00618), and action detection is performed using a [TRN](https://arxiv.org/abs/1811.07391).

## Usage
```bash
python3 sbb.py <directory of images> <vad scores> <oad scores> <object tracking output>
```

The offline SBB requires a directory of N images to compress. We assume VAD, OAD, and object tracking are provided by upstream systems. VAD scores are provided as a NumPy array with shape (N,) in a .npy file. OAD scores are provided as a NumPy array with shape (N,17) in a .npy file. Object tracking output is provided as a list of lists in a .pkl file. The main list should be of length N, and each nested list should contain the object tracking IDs detected in the corresponding frame.

After execution, the offline SBB will output a JSON file for each buffer. Each JSON file has four fields: "value", a list of values for each frame in the buffer; "cost", a list of estimated storage costs for each frame in the buffer; "frame", a list of frame indices in the buffer; and "decision", a list of compression factor decisions for each frame. The "decision" field can then be used to compress the images and view SBB-compressed images.
