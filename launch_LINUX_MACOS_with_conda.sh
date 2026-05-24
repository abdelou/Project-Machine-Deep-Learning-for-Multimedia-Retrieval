#!/bin/bash

read -p "Voulez-vous creer un nouvel environnement conda? (o/n) " create_env

if [[ "$create_env" == [oO] ]]; then
    conda env create -n MIR --file "./requirements/requirements.yml"
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate MIR
    pip install -r "./requirements/requirements.txt"
else
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate MIR
fi

cd src
python interface.py