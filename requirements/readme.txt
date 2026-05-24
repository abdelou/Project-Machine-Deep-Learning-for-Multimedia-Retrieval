conda env create -n MIR  --file requirements.yml
conda activate MIR
pip-compile requirements.in
pip install -r requirements.txt 