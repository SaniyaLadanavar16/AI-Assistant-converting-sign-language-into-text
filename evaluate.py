import numpy as np
from model_training import model

# Load preprocessed data
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy:.2f}')