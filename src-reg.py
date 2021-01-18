# Scikit-learn regression #
# ----------------------- #

import neptune
from neptunecontrib.monitoring.sklearn import log_regressor_summary
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Create and fit random forest regressor
parameters = {'n_estimators': 70,
              'max_depth': 7,
              'min_samples_split': 3,
              'min_samples_leaf': 1}

rfr = RandomForestRegressor(**parameters)
X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28743)
rfr.fit(X_train, y_train)

# Initialize Neptune
neptune.init('neptune-ai/tour-with-scikit-learn')

# Create an experiment
neptune.create_experiment(params=parameters,
                          name='regression-example',
                          tags=['RandomForestRegressor', 'regression'])

# Log regressor summary
log_regressor_summary(rfr, X_train, X_test, y_train, y_test)
