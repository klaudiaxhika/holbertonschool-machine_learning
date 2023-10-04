Gaussian Processes (GPs):
A Gaussian Process is a non-parametric probabilistic model used for regression and classification tasks. It defines a distribution over functions, allowing us to capture uncertainty in predictions. In Bayesian optimization, GPs are commonly used to model the objective function, allowing us to make informed decisions about where to sample next in order to find the optimal set of hyperparameters.

Bayesian Optimization:
Bayesian optimization is a global optimization technique that aims to find the maximum (or minimum) of an expensive, noisy, and non-convex objective function. It uses a probabilistic surrogate model (usually a GP) to model the objective function and an acquisition function to determine where to evaluate the objective next. This process iteratively refines the surrogate model, aiming to find the global optimum efficiently.

Model and Hyperparameters:
For this example, we'll optimize a simple feedforward neural network using GPyOpt. We'll focus on the following hyperparameters:

Learning Rate
Number of Units in a Layer
Dropout Rate
L2 Regularization Weight
Batch Size
I'll use the accuracy on a validation dataset as our satisficing metric, as I aim to maximize the model's performance.


# First, I selected Iris dataset for simplicity, dividing it into training and validation sets.

import GPyOpt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Loading and preprocessing our data
data = load_iris()
X, y = data.data, data.target
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# - Learning Rate
# - Number of Units in a Layer
# - Dropout Rate
# - L2 Regularization Weight
# - Batch Size

def create_model(learning_rate, num_units, dropout_rate, l2_weight, batch_size):
    # Building neural architecture
    model = Sequential()
    model.add(Dense(num_units, activation='relu', input_dim=4))
    model.add(Dropout(dropout_rate))
    model.add(Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    
    # optimizer and loss function
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def train_and_evaluate_model(learning_rate, num_units, dropout_rate, l2_weight, batch_size):
    model = create_model(learning_rate, num_units, dropout_rate, l2_weight, batch_size)
    
    # checkpoint and early stopping
    checkpoint_filepath = f"model_checkpoint_lr{learning_rate}_units{num_units}_dropout{dropout_rate}_l2{l2_weight}_batch{batch_size}.h5"
    model_checkpoint = ModelCheckpoint(checkpoint_filepath, save_best_only=True, save_weights_only=True, monitor='val_accuracy', mode='max', verbose=1)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max', verbose=1)
    
    # train
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=batch_size, callbacks=[model_checkpoint, early_stopping], verbose=0)
    
    # validation accuracy
    best_val_accuracy = max(history.history['val_accuracy'])
    
    return best_val_accuracy

param_space = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.001, 0.1)},
    {'name': 'num_units', 'type': 'discrete', 'domain': (16, 32, 64)},
    {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.0, 0.5)},
    {'name': 'l2_weight', 'type': 'continuous', 'domain': (0.0, 0.01)},
    {'name': 'batch_size', 'type': 'discrete', 'domain': (8, 16, 32)}
]

# Bayesian optimization
def bayesian_optimization_function(params):
    # These are the ingredients for our magic potion – hyperparameters
    learning_rate, num_units, dropout_rate, l2_weight, batch_size = params[0], int(params[1]), params[2], params[3], int(params[4])
    return -train_and_evaluate_model(learning_rate, num_units, dropout_rate, l2_weight, batch_size)

optimizer = GPyOpt.methods.BayesianOptimization(f=bayesian_optimization_function, domain=param_space, initial_design_numdata=5)

# maximum of 30 iterations
max_iterations = 30
optimizer.run_optimization(max_iter=max_iterations)

with open('bayes_opt.txt', 'w') as f:
    f.write(str(optimizer.X))
    f.write('\n')
    f.write(str(optimizer.Y))

# The plot
optimizer.plot_convergence()

best_params = dict(zip([param['name'] for param in param_space], optimizer.x_opt))
best_val_accuracy = -optimizer.fx_opt
print("Best Hyperparameters:", best_params)
print("Best Validation Accuracy:", best_val_accuracy)


This script uses GPyOpt to perform Bayesian optimization to find the best hyperparameters for a simple neural network on the Iris dataset. It saves checkpoints of the best model during training and plots the optimization convergence.

Conclusion:
In this example, I've implemented a Bayesian optimization approach to optimize hyperparameters for a neural network model. I've discussed Gaussian Processes, Bayesian Optimization, and my choice of hyperparameters and satisficing metric. By iteratively optimizing hyperparameters, I can find the best configuration for my model, ultimately leading to improved performance.
