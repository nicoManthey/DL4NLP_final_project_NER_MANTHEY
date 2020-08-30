# Comparison of a LSTM model and a LSTM-CRF model for Named Entity Recognition on the CoNLL-2003 dataset

This is my implementation of a NER model based on that in the paper
[Neural Architectures for Named Entity Recognition](https://www.aclweb.org/anthology/N16-1030/) for the seminar Deep Learning for NLP at the university of Potsdam.

To run the provided python script, first download the glove vectors 50d from  [this web page](https://www.kaggle.com/rtatman/glove-global-vectors-for-word-representation?select=glove.6B.50d.txt).
The script can now be executed with
```sh
python3 ner.py --glove_file PATH_TO_YOUR_GLOVE_FILE
```

### Requirements:
Best is to create a new python3 environment and install the required packages with:

```sh
pip3 install -r requirements.txt
```

The script was tested only with python 3.8.5 and the following package versions:
- numpy==1.17.3
- pandas==1.0.5
- scikit-learn==0.23.1
- scipy==1.5.1
- seaborn==0.10.1
- torch==1.5.1
- TorchCRF==1.0.8
- torchtext==0.6.0
- pytorch-crf==0.7.2
