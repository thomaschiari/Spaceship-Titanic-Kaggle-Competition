import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier

# Set up file names and directories
train = pd.read_csv(os.path.join('data', 'train.csv'))
test = pd.read_csv(os.path.join('data', 'test.csv'))

# Drop null values
train = train.dropna(axis=0)

# Create target object and call it y
y = train.Transported
X = train.drop(['Transported'], axis=1)

# Select only numeric columns
X = X.select_dtypes(include='float64')

# Create the model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

# Fit the model
model.fit(X, y)

# Get test data
test_X = test.select_dtypes(include='float64')

# fill in missing values in test data
test_X = test_X.fillna(test_X.mean())

# Use the model to make predictions
predicted = model.predict(test_X)

# Create submission file
my_submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Transported': predicted})
my_submission.to_csv(os.path.join('submission', 'submission.csv'), index=False)