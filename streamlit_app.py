
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av


flip = st.checkbox("Flip")

import yaml
from ultralytics import YOLO

config = {
    "nc": 80,
    "endtoend":True,
    "reg_max": 16,
    "scales": {"l": [1.00, 1.00, 1024]},  # YOLO26l scales
    "backbone": [
        [-1, 1, "Conv", [64, 3, 2]],      # 0-P1/2
        [-1, 1, "Conv", [128, 3, 2]],     # 1-P2/4
        [-1, 3, "C3k2", [128, True]],     # 2
        [-1, 1, "Conv", [256, 3, 2]],     # 3-P3/8
        [-1, 6, "C3k2", [256, True]],     # 4
        [-1, 1, "Conv", [512, 3, 2]],     # 5-P4/16
        [-1, 6, "C3k2", [512, True]],     # 6
        [-1, 1, "Conv", [1024, 3, 2]],    # 7-P5/32
        [-1, 3, "C3k2", [1024, True]],    # 8
        [-1, 1, "Conv", [1024, 3, 2]],    # 9-P6/64
        [-1, 3, "C3k2", [1024, True]],    # 10
        [-1, 1, "SPPF", [1024, 5]],       # 11
        [-1, 1, "C2PSA", [1024]]          # 12
    ],
    "head": [
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]], # P5 up
        [[-1, 8], 1, "Concat", [1]],
        [-1, 3, "C3k2", [1024, True]],                # 15-P5

        [-1, 1, "nn.Upsample", [None, 2, "nearest"]], # P4 up
        [[-1, 6], 1, "Concat", [1]],
        [-1, 3, "C3k2", [512, True]],                 # 18-P4

        [-1, 1, "nn.Upsample", [None, 2, "nearest"]], # P3 up
        [[-1, 4], 1, "Concat", [1]],
        [-1, 3, "C3k2", [256, True]],                 # 21-P3

        [-1, 1, "nn.Upsample", [None, 2, "nearest"]], # P2 up
        [[-1, 2], 1, "Concat", [1]],
        [-1, 3, "C3k2", [128, True]],                 # 24-P2

        [-1, 1, "nn.Upsample", [None, 2, "nearest"]], # P1 up
        [[-1, 0], 1, "Concat", [1]],
        [-1, 3, "C3k2", [64, True]],                  # 27-P1

        [-1, 1, "Conv", [64, 3, 2]],                  # 28 (P1->P2)
        [[-1, 24], 1, "Concat", [1]],
        [-1, 3, "C3k2", [128, True]],                 # 30-P2

        [-1, 1, "Conv", [128, 3, 2]],                 # 31 (P2->P3)
        [[-1, 21], 1, "Concat", [1]],
        [-1, 3, "C3k2", [256, True]],                 # 33-P3

        [-1, 1, "Conv", [256, 3, 2]],                 # 34 (P3->P4)
        [[-1, 18], 1, "Concat", [1]],
        [-1, 3, "C3k2", [512, True]],                 # 36-P4

        [-1, 1, "Conv", [512, 3, 2]],                 # 37 (P4->P5)
        [[-1, 15], 1, "Concat", [1]],
        [-1, 3, "C3k2", [1024, True]],                # 39-P5

        [-1, 1, "Conv", [1024, 3, 2]],                # 40 (P5->P6)
        [[-1, 12], 1, "Concat", [1]],
        [-1, 3, "C3k2", [1024, True]],                # 42-P6

        [[27, 30, 33, 36, 39, 42], 1, "Detect", [80]] # 43-Head (P1-P6)
    ]
}

# Save and load as Detect task
with open("yolo26l-p6-detect.yaml", "w") as f:
    yaml.dump(config, f)

model = YOLO("yolo26l-p6-detect.yaml", task="detect")
model.load("yolo26l.pt") # Load large weights
model.save("yolo_custum_26l.pt")

#For more details on scaling and architectures, check the [YOLO26 Documentation](https://docs.ultralytics.com/models/yolo26). ✅
result_list=[]
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    flipped = img[::-1,:,:] if flip else img
    results=model.pridict(flipped)
    result_list.append(results)

    return av.VideoFrame.from_ndarray(flipped, format="bgr24")


webrtc_streamer(key="example", video_frame_callback=video_frame_callback)
