# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This is a simple model to classify if salary of a person is greater than 50k dolars or not. It is based on a gradient boosting decision tree algorithm known as LightGBM. The implementation used was the one available at [LightGBM library](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html) and were used here with default parameters (not tunned). 

## Intended Use

Classify basead on census data if a person has a salary over 50k dolars per year. 

## Training Data

Training data is composed of 80% of the dataset and is available at `src/data/train.csv`. Data used in this set was selected randomly using sklearn `train_test_split` and can be generated again by running the script `train_model.py`

## Evaluation Data

Evaluation data is composed of 20% of the dataset and is available at `src/data/test.csv`. Similar to training data, data used in this set was selected randomly using sklearn `train_test_split` and can be generated again by running the script `train_model.py`

## Metrics

To evaluate model performance was used precision, recall and f-beta. The performance on training and evaluation data is shown below:

Train set Metrics
Precision: 0.8195
Recall:    0.6934
F-beta:    0.7512

Test set Metrics
Precision: 0.7757
Recall:    0.6710
F-beta:    0.7196

## Caveats and Recommendations

If you are planning to use this model in a real world scenario, you should consider re-training it to fine tune it's hyperparameters 