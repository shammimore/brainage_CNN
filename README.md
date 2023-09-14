# Basic information


Code for the SFCN model is taken from : https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain

Paper: Accurate brain age prediction with lightweight deep neural networks

The code by default uses the age range [42, 82] and trains the outermost layer on the user provided data.

We currently use the whole T1 image.


# Installation
**Install following python packages:**
1. Pytorch/torch and torchvision (https://pytorch.org/get-started/locally/)
2. pandas (https://pypi.org/project/pandas/)
3. nilearn (https://nilearn.github.io/stable/index.html)
4. matplotlib (https://matplotlib.org/)

**Install using requirements.txt:**
1. Create a virtual enviornment and activate it\
python -m venv ~/.venvs/cnn_env\
source ~/.venvs/cnn_env/bin/activate 

2. Install from requirements.txt file\
pip install -r requirements.txt

# Example
Look at brainage.py to get started.

