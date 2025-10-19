Examples
========

This page provides practical examples of using FCVOpt for various machine learning tasks.

Scikit-learn Random Forest
------------------------------

Optimize Random Forest hyperparameters on a classification task:

.. code-block:: python

   from fcvopt.optimizers import FCVOpt
   from fcvopt.crossvalidation import SklearnCVObj
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.datasets import load_digits
   from sklearn.model_selection import train_test_split
   from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical

   # Load data
   X, y = load_digits(return_X_y=True)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # Create CV objective
   cv_obj = SklearnCVObj(
       estimator=RandomForestClassifier(random_state=42),
       X=X_train, y=y_train,
       cv=5,
       scoring='f1_macro'
   )

   # Define configuration space
   config_space = ConfigurationSpace()
   config_space.add_hyperparameter(Integer("n_estimators", 50, 500))
   config_space.add_hyperparameter(Float("max_features", 0.1, 1.0))
   config_space.add_hyperparameter(Integer("max_depth", 5, 30))
   config_space.add_hyperparameter(Integer("min_samples_split", 2, 20))
   config_space.add_hyperparameter(Categorical("criterion", ["gini", "entropy"]))

   # Optimize
   optimizer = FCVOpt(cv_obj, config_space, acq='kg', mlflow_tracking=True)
   best_config = optimizer.optimize(n_iter=100)

   # Train final model
   final_model = RandomForestClassifier(**best_config, random_state=42)
   final_model.fit(X_train, y_train)

   print(f"Test accuracy: {final_model.score(X_test, y_test):.4f}")

XGBoost with Early Stopping
-------------------------------

Optimize XGBoost with early stopping for better performance:

.. code-block:: python

   from fcvopt.optimizers import FCVOpt
   from fcvopt.crossvalidation import XGBoostCVObjEarlyStopping
   from sklearn.datasets import load_breast_cancer
   from ConfigSpace import ConfigurationSpace, Integer, Float
   import xgboost as xgb

   # Load data
   X, y = load_breast_cancer(return_X_y=True)

   # Create XGBoost CV objective with early stopping
   cv_obj = XGBoostCVObjEarlyStopping(
       estimator=xgb.XGBClassifier(random_state=42),
       X=X, y=y,
       cv=5,
       scoring='roc_auc',
       early_stopping_rounds=20,
       eval_metric='auc'
   )

   # Configuration space
   config_space = ConfigurationSpace()
   config_space.add_hyperparameter(Integer("n_estimators", 100, 2000))
   config_space.add_hyperparameter(Float("learning_rate", 0.01, 0.3, log=True))
   config_space.add_hyperparameter(Integer("max_depth", 3, 12))
   config_space.add_hyperparameter(Float("subsample", 0.6, 1.0))
   config_space.add_hyperparameter(Float("colsample_bytree", 0.6, 1.0))
   config_space.add_hyperparameter(Float("reg_alpha", 1e-5, 1.0, log=True))
   config_space.add_hyperparameter(Float("reg_lambda", 1e-5, 1.0, log=True))

   # Optimize
   optimizer = FCVOpt(cv_obj, config_space, acq='kg')
   best_config = optimizer.optimize(n_iter=150)

Neural Network Optimization
-------------------------------

Optimize a multi-layer perceptron:

.. code-block:: python

   from fcvopt.optimizers import FCVOpt
   from fcvopt.crossvalidation import MLPCVObj
   from sklearn.datasets import make_classification
   from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical

   # Generate data
   X, y = make_classification(n_samples=2000, n_features=50, n_classes=3,
                              n_informative=30, random_state=42)

   # Create MLP CV objective
   cv_obj = MLPCVObj(
       X=X, y=y,
       cv=5,
       max_epochs=200,
       early_stopping_patience=20,
       batch_size=64
   )

   # Configuration space for neural network
   config_space = ConfigurationSpace()
   config_space.add_hyperparameter(Integer("hidden_size", 32, 512))
   config_space.add_hyperparameter(Integer("num_layers", 1, 4))
   config_space.add_hyperparameter(Float("learning_rate", 1e-4, 1e-1, log=True))
   config_space.add_hyperparameter(Float("dropout", 0.0, 0.5))
   config_space.add_hyperparameter(Categorical("activation", ["relu", "tanh", "elu"]))
   config_space.add_hyperparameter(Float("weight_decay", 1e-6, 1e-2, log=True))

   # Optimize
   optimizer = FCVOpt(cv_obj, config_space, acq='kg')
   best_config = optimizer.optimize(n_iter=80)

Regression with Support Vector Regression
---------------------------------------------

Optimize SVR for a regression task:

.. code-block:: python

   from fcvopt.optimizers import FCVOpt
   from fcvopt.crossvalidation import SklearnCVObj
   from sklearn.svm import SVR
   from sklearn.datasets import load_diabetes
   from sklearn.preprocessing import StandardScaler
   from ConfigSpace import ConfigurationSpace, Float, Categorical

   # Load regression data
   X, y = load_diabetes(return_X_y=True)

   # Scale features (important for SVR)
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)

   # Create CV objective for regression
   cv_obj = SklearnCVObj(
       estimator=SVR(),
       X=X_scaled, y=y,
       cv=5,
       scoring='neg_mean_squared_error'  # Regression metric
   )

   # Configuration space for SVR
   config_space = ConfigurationSpace()
   config_space.add_hyperparameter(Float("C", 0.1, 1000, log=True))
   config_space.add_hyperparameter(Float("gamma", 1e-5, 1.0, log=True))
   config_space.add_hyperparameter(Float("epsilon", 0.01, 1.0, log=True))
   config_space.add_hyperparameter(Categorical("kernel", ["rbf", "poly", "sigmoid"]))

   # Optimize
   optimizer = FCVOpt(cv_obj, config_space, acq='kg')
   best_config = optimizer.optimize(n_iter=100)

Multi-class Classification with Custom Scoring
--------------------------------------------------

Optimize for a custom scoring function:

.. code-block:: python

   from fcvopt.optimizers import FCVOpt
   from fcvopt.crossvalidation import SklearnCVObj
   from sklearn.ensemble import GradientBoostingClassifier
   from sklearn.datasets import load_wine
   from sklearn.metrics import make_scorer, balanced_accuracy_score
   from ConfigSpace import ConfigurationSpace, Integer, Float

   # Load multi-class data
   X, y = load_wine(return_X_y=True)

   # Custom scorer
   balanced_accuracy_scorer = make_scorer(balanced_accuracy_score)

   # Create CV objective
   cv_obj = SklearnCVObj(
       estimator=GradientBoostingClassifier(random_state=42),
       X=X, y=y,
       cv=5,
       scoring=balanced_accuracy_scorer
   )

   # Configuration space
   config_space = ConfigurationSpace()
   config_space.add_hyperparameter(Integer("n_estimators", 50, 300))
   config_space.add_hyperparameter(Float("learning_rate", 0.01, 0.5, log=True))
   config_space.add_hyperparameter(Integer("max_depth", 3, 10))
   config_space.add_hyperparameter(Float("subsample", 0.5, 1.0))
   config_space.add_hyperparameter(Integer("max_features", 3, X.shape[1]))

   # Optimize
   optimizer = FCVOpt(cv_obj, config_space, acq='kg')
   best_config = optimizer.optimize(n_iter=80)

Comparing Different Acquisition Functions
---------------------------------------------

Compare performance of different acquisition functions:

.. code-block:: python

   from fcvopt.optimizers import FCVOpt
   from fcvopt.crossvalidation import SklearnCVObj
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.datasets import make_classification
   import mlflow

   # Generate data
   X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

   # Create CV objective
   cv_obj = SklearnCVObj(
       estimator=RandomForestClassifier(random_state=42),
       X=X, y=y,
       cv=5,
       scoring='accuracy'
   )

   # Simple configuration space
   from ConfigSpace import ConfigurationSpace, Integer, Float
   config_space = ConfigurationSpace()
   config_space.add_hyperparameter(Integer("n_estimators", 50, 500))
   config_space.add_hyperparameter(Float("max_features", 0.1, 1.0))

   # Compare acquisition functions
   acquisition_functions = ['kg', 'lcb']
   results = {}

   mlflow.set_experiment("acquisition_comparison")

   for acq in acquisition_functions:
       with mlflow.start_run(run_name=f"fcvopt_{acq}"):
           optimizer = FCVOpt(cv_obj, config_space, acq=acq, mlflow_tracking=True)
           best_config = optimizer.optimize(n_iter=50)
           results[acq] = {
               'best_config': best_config,
               'best_score': optimizer.best_observed_value
           }

   # Print comparison
   for acq, result in results.items():
       print(f"{acq}: {result['best_score']:.4f}")

Custom Cross-Validation Objective
-------------------------------------

Create a custom CV objective for specialized models:

.. code-block:: python

   from fcvopt.optimizers import FCVOpt
   from fcvopt.crossvalidation import CVObjective
   from sklearn.model_selection import cross_val_score
   from sklearn.linear_model import ElasticNet
   from sklearn.datasets import make_regression
   from ConfigSpace import ConfigurationSpace, Float

   class ElasticNetCVObj(CVObjective):
       def __init__(self, X, y, cv=5):
           super().__init__(X, y, cv)
           self.model = ElasticNet(random_state=42)

       def __call__(self, config):
           # Set hyperparameters
           self.model.set_params(**config)

           # Evaluate with cross-validation
           scores = cross_val_score(
               self.model, self.X, self.y,
               cv=self.cv, scoring='neg_mean_squared_error'
           )
           return scores.mean()

   # Generate regression data
   X, y = make_regression(n_samples=500, n_features=20, noise=0.1, random_state=42)

   # Create custom CV objective
   cv_obj = ElasticNetCVObj(X, y, cv=5)

   # Configuration space
   config_space = ConfigurationSpace()
   config_space.add_hyperparameter(Float("alpha", 1e-4, 10.0, log=True))
   config_space.add_hyperparameter(Float("l1_ratio", 0.0, 1.0))

   # Optimize
   optimizer = FCVOpt(cv_obj, config_space, acq='kg')
   best_config = optimizer.optimize(n_iter=60)

Parallel Processing
----------------------

Utilize parallel processing for faster CV evaluation:

.. code-block:: python

   from fcvopt.optimizers import FCVOpt
   from fcvopt.crossvalidation import SklearnCVObj
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.datasets import load_digits

   # Load data
   X, y = load_digits(return_X_y=True)

   # Create CV objective with parallel processing
   cv_obj = SklearnCVObj(
       estimator=RandomForestClassifier(random_state=42, n_jobs=-1),  # Parallel RF
       X=X, y=y,
       cv=5,
       scoring='accuracy',
       n_jobs=4  # Parallel CV evaluation
   )

   # Configuration space
   from ConfigSpace import ConfigurationSpace, Integer, Float
   config_space = ConfigurationSpace()
   config_space.add_hyperparameter(Integer("n_estimators", 50, 300))
   config_space.add_hyperparameter(Float("max_features", 0.1, 1.0))

   # Optimize with parallel evaluation
   optimizer = FCVOpt(cv_obj, config_space, acq='kg')
   best_config = optimizer.optimize(n_iter=50)

Tips for Better Results
---------------------------

1. Scale Your Features: Many algorithms benefit from feature scaling
2. Choose Appropriate CV: Use stratified CV for imbalanced datasets
3. Set Random Seeds: Ensure reproducible results
4. Monitor Progress: Use MLflow to track optimization progress
5. Start Small: Begin with fewer iterations to validate your setup
6. Configuration Space: Keep hyperparameter ranges reasonable