# TextDetectorEAST
Re-Implementation Text Detector with [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155) using Pytorch Framework. The dataset using [MSRA Text Detection (MSRA-TD500)](http://www.iapr-tc11.org/mediawiki/index.php/MSRA_Text_Detection_500_Database_(MSRA-TD500))

## Dependencies
- Python    (>= 3.6)
- Pytorch   (>= 1.10)

# How it Works

## Requirements
```bash
pip3 install -r requirements.txt
```

## Getting Dataset
First of all you must download and extract the [dataset](http://www.iapr-tc11.org/dataset/MSRA-TD500/MSRA-TD500.zip) in data folder.
```bash
wget http://www.iapr-tc11.org/dataset/MSRA-TD500/MSRA-TD500.zip
unzip MSRA-TD500.zip
```

## In Details
```
.
├── data
│   └── MSRA-TD500
│       ├── MSRA Text Detection 500 Database (MSRA-TD500) Readme.doc
│       ├── test
│       └── train
├── LICENSE
├── README.md
├── requirements.txt
├── scripts
│   ├── dataset.py
│   ├── engine.py
│   ├── lossFn.py
│   ├── model.py
│   └── utils.py
└── train.py
```

## Training
```python3
>>> python3 train.py --epochs ${Number of epochs} --name-model ${Name of Model} --checkpoint --early-stop
>>> python3 train.py --epochs 100 --name-model model.pth --checkpoint --early-stop # Enable Checkpoint and Early stopping function
```

