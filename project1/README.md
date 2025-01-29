# Programming project 1 — Prototype selection for nearest neighbors classification

This work implements 5 strategies for sampling $M$ examples from the MNIST dataset of handwritten images, and then runs a 1-Nearest Neighbour classifier trained on this reduced sample of the train set. Each strategy is run 30 times, and the average accuracy and 95% confidence interval is reported in the ```results``` directory, with the appropriate boxplots in the ```plots``` directory. 

The project's file structure is as below, with the dataset extracted in the below format

```
├── cse251a_proj1_wi25.pdf
├── dataset
│   ├── t10k-images-idx3-ubyte
│   │   └── t10k-images-idx3-ubyte
│   ├── train-labels.idx1-ubyte
│   ├── train-images.idx3-ubyte
│   ├── t10k-labels-idx1-ubyte
│   │   └── t10k-labels-idx1-ubyte
│   ├── train-images-idx3-ubyte
│   │   └── train-images-idx3-ubyte
│   ├── t10k-labels.idx1-ubyte
│   ├── t10k-images.idx3-ubyte
│   └── train-labels-idx1-ubyte
│       └── train-labels-idx1-ubyte
├── testbench.py
├── README.md
├── dataloader.py
├── datasampler.py
├── main.py
└── knn.py
```

## Execution

First ensure that the required Python packages are installed by running 

```bash
pip install -r requirements.txt
```

Execute the sampling and inference pipeline using 

```bash
python main.py
```

## Code Structure

```main.py``` is the entry point of the entire project. It runs the entire sampling and inference pipeline

```testbench.py``` contains code that executes the pipeline in sequence, given a KNea