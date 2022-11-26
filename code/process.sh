#!/bin/bash

imgs="$1"
cd backbone/detect
python3 detect.py $imgs

cd ../align
python3 align.py $imgs

cd ..
