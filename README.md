# MetaOpt

Simple and easy-to-use implementation of the most recent ideas in Bayesian Optimization, targeted for use in hyperparameter tuning.


## Usage

```
# create a MetaOptimizer object
mop = MetaOptimizer(feature_meta={
    'max_depth': (2, 25),
    'n_estimators': (5, 250),
    'eta': (.01, 2.)
})

# generate next parameters to be explored
params = mop.suggest()
# {'max_depth': 4, 'n_estimators': 52, 'eta': 0.1574}

result = crossvalidation(params)
# 0.87
# refit the model with the target value for the suggested parameters
mop.update(params, result)

# shorthand for update and suggest
params = mop.step(params, result)
```


## References

Jasper Snoek, Hugo Larochelle, and Ryan P. Adams, 2012, [Practical Bayesian Optimization of Machine Learning Algorithms](https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf)
