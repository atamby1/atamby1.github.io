Title: Ensemble Classifiers
Date: 2017-01-27
Category: Data Science
Tags: Data Science
Author: Avinash TAMBY

Ensemble techniques are supervised learning methods to improve predictive accuracy by combining several base models in order to enlarge the space of possible hypotheses to represent our data. Ensembles are often much more accurate than the base classifiers that compose them.

There are two general types of ensemble methods: bagged models and boosted models. With bagged models (short for bootstrap aggregated models), we run a base classifier (maybe a decision tree or a kNN model) k times on bootstrapped samples of our data. We then average the results from each model for our final prediction. Specifically bagging with decision trees, we can add several layers of randomization to improve the model.

For example, suppose we have an n x p feature matrix X and we want to predict binary class labels. If we wanted to run a decision tree, we would try to find the feature that distinguishes between the two classes the best, and create a split, then for each split, find another feature that distinguishes between the two classes the best, and continue until we reach some pre-specified criteria. This method is prone to high variance (overfitting) because it is very, very dependent on the data we use to train the tree. A bagged tree can correct for some of this variance because since each tree is built with a different subset of the data, the features that best separate classes will be different for each tree.

But we can go one step further in randomization with random forests. With random forests, not only do we built trees with bootstrapped subsamples, but we also take a random subset of features for each tree that's built. That means that there's a set of features is not even being *considered* when building each tree. This corrects for decision trees' high variance even more than bagged DTs.

But we can even take another step with randomization with ExtRaTrees (Extremely Randomized Trees). While DTs, Bagged DTs and RFs try to find the optimal split, Extra Trees are random forests, but instead of selecting the optimal value when determining where to split the tree, a random value is selected.

More randomization leads to lower variance, which can be a good thing with decision trees, and generally speaking random forests tend to perform pretty well on many machine learning problems. Extra Trees are not really used as often.

The second general category of ensemble methods is the boosted model. Boosted models are also generally used with decision trees. While with bagged models, each base classifier in the "bag" can be trained simultaneously, boosted models are iterative. We run a model once, then adjust it to try to minimize the misclassifications from the first model and run it again, and repeat this process until some stopping criteria is met.

One example of a boosted model is AdaBoost (adaptive boosting). With this model, we start with a base classifier (usually a decision tree) and then find all of the missclassified labels and increase their weights. We fit a new decision tree on the weighted data so that the DT will put more effort into correctly classifying the more heavily weighted data. We continue this process until a stopping criteria is met.\

One of the most popular machine learning algorithms is the gradient boosting tree. This is a more generalized version of AdaBoost where we calculate "pseudo-residuals" or the quantified difference between the true classification and thep predicted classification and apply a gradient descent algorithm to minimize a loss function (which can be interpreted as trying to minimize the pseudo-residuals that we calculate for each model. Again, we continue until we meet some stopping criteria (usually once we've hit a specificed number of trees built).

So these are just a few different types of ensemble methods for machine learning classification. Random Forests and Gradient Boosted Trees are the most popular ones right now, so I thought it would be cool to give a high-level overview of what these methods actually entail without going too deep into the math behind them.
