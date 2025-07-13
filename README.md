# Kernel Density Estimation (KDE) Plots for Model Evaluation

## Overview

Kernel Density Estimation (KDE) plots are used to visualize the probability density function of a dataset, providing a smooth approximation of the data distribution. They are particularly helpful in regression analysis for comparing the distributions of actual and predicted values to assess model performance.

## Benefits of KDE Plots

- Smooth estimation of data distribution  
- Effective comparison of actual vs predicted data  
- Not sensitive to bin size like histograms  
- Highlights deviations between observed and predicted values  

## How to Use KDE Plots in Python

Use the `seaborn.kdeplot()` function to plot KDE curves for actual and predicted values.

### Example Code

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Generate sample data
np.random.seed(42)
x = np.random.rand(100) * 10
y = 3 * x + np.random.normal(0, 3, 100)  # Linear relation with noise
data = pd.DataFrame({'X': x, 'Y': y})

# Split data
X_train, X_test, y_train, y_test = train_test_split(data[['X']], data['Y'], test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Plot KDE of actual vs predicted values
plt.figure(figsize=(8, 5))
sns.kdeplot(y_test, label='Actual', fill=True, color='blue')
sns.kdeplot(y_pred, label='Predicted', fill=True, color='red')
plt.xlabel('Target Variable')
plt.ylabel('Density')
plt.title('KDE Plot of Actual vs. Predicted Values')
plt.legend()
plt.show()
