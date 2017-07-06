# MetaOpt

Simple and easy-to-use implementation of the most recent ideas in _Bayesian optimization_, targeted for use in hyperparameter tuning.


## Usage

```
# 1. create a MetaOptimizer object
# with the bounds for your parameters
mop = MetaOptimizer(feature_meta={
    'max_depth': (2, 25),
    'n_estimators': (5, 250),
    'eta': (.01, 2.)
})

# 2. generate next parameters to be explored
params = mop.suggest()
# {'max_depth': 4, 'n_estimators': 52, 'eta': 0.1574}

# 3. crossvalidate the model with the target parameters
# and obtain a score (target value) for these
# used to update the optimizer
result = crossvalidation(params)
# 0.87
mop.update(params, result)

# 4. Go back to 2.

# shorthand for update and suggest (2. and 3.)
params = mop.step(params, result)
```


## References

Jasper Snoek, Hugo Larochelle, and Ryan P. Adams, 2012, [Practical Bayesian Optimization of Machine Learning Algorithms](https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf)
