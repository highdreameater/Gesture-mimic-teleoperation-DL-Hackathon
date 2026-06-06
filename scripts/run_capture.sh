#!/bin/bash
LABEL="${1:-unlabelled}"
python src/capture.py live --out data/gesture.h5 --label "$LABEL"