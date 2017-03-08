Title: Why Are There So Many Machine Learning Classifiers?
Date: 2016-12-23
Category: Data Science
Tags: Data Science
Author: Avinash TAMBY

I'm sure that at some point, most data scientists must ask the question: Why are there so many classifiers? There's k-Nearest Neighbors, Logistic Regression, SVMs, Decision Trees, Naive Bayes, Random Forests, Neural Networks, and the list goes on! But do we really need all of these classifiers?

Well, I'd say it probably *isn't* really necessary to have all of these classifiers, but they are nice to have. There are some algorithms which arguably usually don't perform as well others for many datasets. These usually include KNN and Decision Trees. These are usually considered "weak learners", but they are useful because they are easily interpretable and they can help you make sense of the data and maybe see some relationships that are not as obvious when using other classifiers.

In data science, there appear to be a lot of tradeoffs and it is up to the data scientist to minimize the losses due to these tradeoffs and balancing these tradeoffs are not an exact science. For the bias-variance tradeoff, it's difficult -- near impossible -- to find the exact point which minimizes both error due to variance and error due to bias without wasting computational resources. The best a data scientist can do is *try* to minimize both types of error.

Returning to the classifiers question, another important tradeoff in data science is complexity vs. interpretability. Very complex models may be very accurate and classify things incredibly well, but those models usually come at the cost of low interpretability. The models become black boxes and we get excited about our high accuracy, but we don't actually know how the model actually classifies observations. It also becomes difficult to explain how the model works and it's difficult to just tell someone to trust your model without them knowing how it works. KNN and DTs are usually easily interpetable, but they are also not very complex models. There are research papers that purport that Random Forests and Gradient Boosting Trees are the *best* models (http://www.jmlr.org/papers/volume15/delgado14a/delgado14a.pdf), and there are other papers that refute this claim (http://jmlr.org/papers/volume17/15-374/15-374.pdf).

When approaching a data science classification problem, I'd suggest trying a variety of classifiers then seeing which ones work best. The popular ones right now seem to be Random Forests and Gradient Boosting Trees, but I'm sure that data scientists change their opinions over time and develop new algorithms with high predictive power *and* high interpretability. A lot of research is going into further developing neural networks, especially in the field of artificial intelligence, but we'll see which ones come out on top.
