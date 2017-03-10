# -*- coding: utf-8 -*-
import argparse

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu-device", type=int, default=0)
parser.add_argument("-m", "--model-dir", type=str, default="model")
parser.add_argument("-l", "--num-labeled-data", type=int, default=0)

# seed
parser.add_argument("-s", "--seed", type=int, default=None)

args = parser.parse_args()