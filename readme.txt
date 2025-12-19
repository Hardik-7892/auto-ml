
Steps to setup python environment

conda create -n automl python=3.11 cudatoolkit=11.3 -y
conda activate AutoML

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

pip install -U pip
pip install -U setuptools wheel
pip install autogluon

pip install -r requirements.txt


After env is set, 

streamlit run app.py
