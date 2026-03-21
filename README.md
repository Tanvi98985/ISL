# ISL Sign Language Classification

Classifies Indian Sign Language (ISL) gestures: A–Z, 0–9 (35 classes).  
Three model approaches are provided, from basic to state-of-the-art.

## Models

| #     | Script                 | Approach                                | Backbone            |
| ----- | ---------------------- | --------------------------------------- | ------------------- |
| 1     | model1_ultralight.py   | Image CNN                               | MobileNetV2         |
| 2     | model2_highperf.py     | Image CNN                               | ResNet50            |
| 3     | model_best.py          | MediaPipe Landmarks + Bone Geometry     | Feed-forward NN     |

## Recommended Model

Best approach uses MediaPipe landmarks + bone geometry features.

## Quick Start

Step 1
python collect_data.py --data_dir ./Indian

Step 2
python model_best.py --mode train

Step 3
python model_best.py --mode ui

## Dataset format

Indian/
 A/
 B/
 C/
 ...
 Z/
 1/
 2/
 ...
 9/

## Project

ISL Recognition using MediaPipe + Neural Network