#!/bin/bash

cd $( dirname -- "$0"; )
python3 test.py --dir $1 --image $2 --model $3