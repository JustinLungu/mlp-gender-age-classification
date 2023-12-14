Information relevant for assignment submission:
The script necessary for the assignment can be found in gender_age_classification/models/


Notes for the team:
- model - see assignment 1 for architecture (half tinyVGG) - train + val
    - for the baseline, input is smaller than final
    - hard parameter sharing
    - two convolutional layers with 64 filters each, followed by a max-pooling layer with a pool size of 2x2. and flatten and dense
    - after that split into two heads for the multi task output
    - output: one regression (RMSE) and one classification (binary cross entropy and accuracy optional)
    - somehow ensure the model only predicts integers (some sort of rounding)
    - what hyperparameters do we want to tune?
        - number of filters
        - number of layers
        - kernel size
        - number of max pooling
        - regularisation such as dropout
        - activation function
    - save final model as .pkl
