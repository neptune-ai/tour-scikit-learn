# Scikit-learn classification #
# --------------------------- #

import neptune
from neptunecontrib.monitoring.sklearn import log_classifier_summary
from sklearn.datasets import load_digits
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# Create and fit gradient boosting classifier
parameters = {'n_estimators': 100,
              'learning_rate': 0.17,
              'min_samples_split': 3,
              'min_samples_leaf': 1}

gbc = GradientBoostingClassifier(**parameters)
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=28743)
gbc.fit(X_train, y_train)

# Initialize Neptune
neptune.init('neptune-ai/tour-with-scikit-learn')

# Create an experiment
neptune.create_experiment(params=parameters,
                          name='classification-example',
                          tags=['GradientBoostingClassifier', 'classification'])

# Log classifier summary
log_classifier_summary(gbc, X_train, X_test, y_train, y_test)
