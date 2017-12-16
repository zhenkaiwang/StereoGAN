#!/bin/bash
trap "eixt" INT
while true; do
sl -e
python stereoGAN.py --mode="train" --save_freq=500
done
