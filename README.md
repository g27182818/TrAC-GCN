# TrAC-GCN Source Code

## Set up

Create a virtual environment

```bash
conda create -n trac
conda activate trac
```
Download [STRING](https://string-db.org/cgi/download?sessionId=b74QbpZboXzM&species_text=Homo+sapiens) data

```bash
mkdir data
cd data
mkdir string
cd string
wget https://stringdb-static.org/download/protein.links.detailed.v11.5/9606.protein.links.detailed.v11.5.txt.gz
gzip -d 9606.protein.links.detailed.v11.5.txt.gz
cd ..
cd ..
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
pip install umap-learn
```


To run the main results:

```bash
bash bash experiment_scripts/one_exp.sh
```

