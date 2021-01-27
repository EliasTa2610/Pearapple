import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle


# Create log. reg. object, load data and learn it
logiReg = LogisticRegression(solver = 'lbfgs')
data = np.load('data.npy')
inputs = (data[:, 0]).reshape(-1, 1)
lbl = data[:, 1]
logiReg.fit(inputs, lbl)


# Save logistic regression model
pkl_filename = "train_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(logiReg, file)
