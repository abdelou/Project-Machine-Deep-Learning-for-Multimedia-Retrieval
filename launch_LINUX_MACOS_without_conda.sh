#!/bin/bash

read -p "Voulez-vous telecharger les dependances? (o/n) " download_dependencies

if [[ "$download_dependencies" == [oO] ]]; then
    pip install -r "./requirements/requirements.txt"
    
fi

cd src
python interface.py