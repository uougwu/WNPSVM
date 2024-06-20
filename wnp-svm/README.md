This paper develops optimization and Machine Learning (ML) algorithms to analyze gene
expression datasets from the lungs and spleen of mice, infected intranasally, with two bacterial
strains, Francisella tularensis - Schu4 and Live Vaccine Strain (LVS). We propose and utilize
Weighted ℓ1-norm Generalized Eigenvalue-type Problems (ℓ1-WGEPs) to determine a small
set of biomarkers (genes) that promote bacterial dissemination of Schu4 and LVS from the
lungs to spleen tissues. The optimal solutions of ℓ1-WGEPs determine the direction onto which
the datasets are projected for dimensionality reduction, with the projection scores computed
and ranked for gene selection. The top k-ranked projection scores correspond to the top k most
informative genes. The top k genes selected from the lungs data are employed to train ML
models, with uninfected controls and Schu4 or LVS samples as classes. The trained models
are validated on the spleen data to incorporate transfer learning. Baseline ML algorithms
such as ANN, XGBoost, AdaBoost, AdaGrad, KNN, SVM, Naive Bayes, Random Forest,
Logistic Regression, and Decision Tree are compared with our Weighted ℓ1-norm Non-Parallel
Proximal Support Vector Machine (ℓ1-WNPSVM) that is based on two non-parallel separating
hyperplanes. We report average balanced accuracy scores of the methods over multiple folds.
Gene ontology is performed on the most significant genes in both tissues to examine for
relevant biological pathways that may aid in preemptive vaccine development and effective
therapeutics.
