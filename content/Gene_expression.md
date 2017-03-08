Title: Relapse Classification from Gene Expressions
Date: 2017-02-10
Category: Data Science
Tags: Data Science
Author: Avinash TAMBY

One really interesting problem in computational molecular medicine, and one that is heavily researched is the prediction of phenotypes (e.g. properties of cancer growths) based on gene expression profiles.

I worked on a project where I try to predict the time of relapse among breast cancer patients diagnosed with a malignant tumor. In particular, I consider patients with ER+ (estrogen receptor positive) disease who were treated with surgery or surgery and radiation, and untreated with systemic hormonal therapy and/or chemotherapy. Of these, I consider 2 categories of patients: 'NoRelapse' patients who did not relapse for the duration of the 6.5 year study and 'Relapse' patients who did. The researchers did post-study follow-ups, and there were indeed patients in the study who relapsed after 6.5 years, but they have been excluded from the dataset.

The data in the gene expression profiles come from DNA microarrays. The data compose a matrix where each row represents an anonymized patient and each column represents the expression level for a particular gene.

Without diving too deep into the biochemical methods behind gathering the data, I just wanted to provide a brief overview of what 'DNA microarray data' actually is. A microarray is a collection of small DNA spots attached to a solid surface. In microarray experiments, the signal collected from each spot is used to estimate the expression level of a gene. A microarray contains thousands of DNA spots, covering almost every gene in a genome.
To make the gene expression levels more comparable between genes, the data have been normalized. This makes it so the numbers in the dataframe are not the true expression levels, but are slighly altered so we can interpret and compare these numbers across genes (the I gathered the normalized data, so I did not have to do any extra preprocessing beforehand).

I was particularly lucky because the data I gathered was already clean. There were no missing variables, and as I mentioned before, the were already normalized for interpretability.

I then split the data into 2 categories: a training set (the set of patients I use to build my predictive model) and a testing set (the set of patients I use to evaluate how well my model performs). The training set contain 22,215 genes for 212 patients. Of these, 152 patients are categorized as 'NoRelapse' and 60 are categorized as 'Relapse'. The test set also contain 22,215 genes for 212 patients, and of these, 137 are categorized as 'NoRelapse' and 75 as 'Relapse'.

The first issue I ran into when trying to work on this problem was high dimensionality. I had 22,215 genes for 424 patients (number of variables (p) << number of observations (n)). I needed a way to filter out some genes that I thought wouldn’t be important. To do this, I run the Wilcoxon Rank-Sum test to see for which genes is it **least likely** that the NoRelapse and Relapse patients come from the same population and I kept the 1,000 genes for which it was least likely for my Random Forest model and my Gradient Boosted Tree model.

When I ran the Random Forest and the XGBoost (and gridsearched through different parameters to find the optimal ones), both yielded relatively similar results at about 67-68% accuracy. But I wanted to try a method that was a little less “black-box”. So I implemented an algorithm that I read about in a research paper called “Top Scoring Pairs” which find gene pairs for which the gene expression levels typically invert from one class to the other. This is a **very** computationally expensive process because I’m looking at every gene pair, so for 22,215 genes, that’s about 248 million gene pairs (22,215 Choose 2).

The Top Scoring Pairs classifier then assigns a score to each pair for which pair discriminates between the 2 classes the best and I then use the pair (or pairs) with the highest scores for my classification. Note: I didn’t actually run this on all 248 million gene pairs, I ran it on about 5,000 genes, so about 12 million gene pairs. My goal is to MapReduce this problem so that to speed it up so that my computer can handle running this on all possible gene pairs.

I’ll discuss the algorithm in a bit more detail in my next post.