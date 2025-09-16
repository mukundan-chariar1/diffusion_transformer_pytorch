# diffusion_transformer_pytorch

## DiT Task for Mukundan Chariar from IISc Bangalore

### Installation

```bash
pip install -r requirements.txt
```

### Download dataset

```bash
mkdir data
cd data
curl -L -o ./landscape-pictures.zip  https://www.kaggle.com/api/v1/datasets/download/arnaud58/landscape-pictures
unzip ./landscape-pictures.zip
rm ./landscape-pictures.zip
```