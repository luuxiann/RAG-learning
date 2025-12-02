# MinerU
https://github.com/opendatalab/MinerU
```
conda deactivate
git clone https://github.com/opendatalab/MinerU.git
cd MinerU
conda create -n mineru python=3.12
conda activate mineru
pip install uv
uv pip install -U "mineru[core]" -i https://mirrors.aliyun.com/pypi/simple 
python analyze.py   # config 路径记得修改
```