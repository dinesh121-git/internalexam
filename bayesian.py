

import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# file_path = '/content/dataset.csv'  # Adjust path if needed
data = pd.read_csv("Heart_disease_statlog.csv")


# Data preprocessing: Replace '?' with NaN, convert to numbers, drop missing values
data = data.replace('?', np.nan).apply(pd.to_numeric, errors='coerce').dropna().astype(int)

# Define Bayesian Network structure
model = DiscreteBayesianNetwork([
    ('age', 'chol'),
    ('age', 'trestbps'),
    ('chol', 'target'),
    ('trestbps', 'target'),
    ('thalach', 'target'),
    ('exang', 'target'),
    ('fbs', 'target')
])


model.fit(data, estimator=MaximumLikelihoodEstimator)


infer = VariableElimination(model)

# Example query: Probability of heart disease for a 37-year-old
result = infer.query(variables=['target'], evidence={'age': 37})

print(result)