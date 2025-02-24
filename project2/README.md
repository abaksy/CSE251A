# Programming project 2 — Co-ordinate descent for Logistic Regression

This work implements co-ordinate descent using a greedy co-ordinate selection mechanism, and two weight update strategies: one using first-order information (i.e. the gradient of the loss), and the other using second order information (the Hessian and the gradient of the loss, resembling Newton's method). 

The project follows the following file structure

```
├── README.md
├── baseline.py
├── constants.py
├── cse251a_proj2_wi25.pdf
├── dataloader.py
├── dataset
│   ├── Index
│   ├── wine.data
│   ├── wine.names
│   └── wine.zip
├── main.py
└── model.py
```

## Execution
Enter the ```project2``` directory and execute all the commands below

First ensure that the required Python packages are installed by running 

```bash
pip install -r requirements.txt
```

Execute the co-ordinate descent algorithm using 

```bash
python main.py
```

Inspect the results in the generated PNG files.

## Code Structure

```main.py``` is the entry point of the entire project. It loads the wine dataset and runs the entire co-ordinate descent algorithm.

```dataloader.py``` implements logic to load the wine dataset from the internet and then transforms the data by scaling each feature, adding a column of 1s to model the intercept term, and then shuffling the examples

```baseline,py``` runs a ```LogisticRegression``` classifier on the dataset. The final training set loss is our baseline value for the experiments conducted.

```model.py``` implements the co-ordinate selection algorithms, including weight update and co-ordinate selection methods
