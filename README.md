# TrAC-GCN Source Code

## Set up

Create a virtual environment

```bash
conda create -n trac
conda activate trac
```

Install required dependencies

```bash
conda install python==3.9
pip3 install torch torchvision torchaudio
# conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
# conda install pyg -c pyg
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu102.html
pip install biomart
pip install h2o
pip install matplotlib
pip install seaborn
pip install networkx
```
