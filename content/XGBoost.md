Title: XGBoost
Date: 2017-02-03
Category: Data Science
Tags: Data Science
Author: Avinash TAMBY

Everyone who uses Python for data science and machine learning probably uses scikit-learn for model building. Scikit-learn is an open-source Python library which features many of the common regression, classification and clustering machine learning techniques and it's an essential toolkit for anyone trying to build a machine learning model in Python.

In this post, however, I want to talk about a *different* machine learning library: XGBoost. XGBoost works with C++, Java, Python, R, and Julia. It can run on a single machine, but also supports distributed processing frameworks such as Apache Hadoop, Spark and Flink. It's gained popularity recently because it's used frequently as the algorithm of choice for many winning teams for machine learning competitions (e.g. Kaggle).

In my last post, I talked about ensemble classifiers, and one type of ensemble classifier is gradient boosting. XGBoost uses gradient boosting to build predictive models. Specifically, it boosts random forests.

Recall that with boosting (I'll talk specifically about boosted random forests) , we have our training data and we build a random forest. The model then reweights observations based on whether or not the observations were classified correctly or incorrectly (correctly classified observations are weighted less heavily and incorrectly classified observations are weighted more heavily). With this technique, the model can put more effort into trying to classify previously misclassified observations correctly.

What's cool about XGBoost is its flexibility and its scalability. While you can tune a lot of parameters with scikit-learn, you can tune even more with XGBoost. XGBoost has more loss functions to choose from which the model will try to minimize with each iteration. It can also handle missing values in the data, something scikit-learn cannot do. Finally, and perhaps most importantly, XGBoost has some *great* computational benefits. It automatically parallels on multi-threaded CPUs which is pretty cool, especially considering the fact that boosted trees are an iterative process, meaning sequential models can't be built simultaneously.

However, while many data scientists and Kaggle competitors use XGBoost to build their models, we must not forget that no one model works best for every problem. The assumptions of a great model for one problem may not hold for for another, so it's usually best practice to try several models to see which one works best for a particular problem. This theorem that no one models works best for every problem is called the No Free Lunch theorem and it's important to keep in mind that although XGBoost is really efficient, gradient boosting may not always be the best algorithm to use for any problem.
