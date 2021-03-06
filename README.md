# Final project - Data Mining (Elective subject)

This project replicates and extends experiments carried out in the following paper:

[NETtalk: a parallel network that learns to read aloud](https://www.semanticscholar.org/paper/NETtalk%3A-a-parallel-network-that-learns-to-read-Sejnowski-Rosenberg/406033f22b6a671b94bcbdfaf63070b7ce6f3e48
)
T. Sejnowski, C. Rosenberg

Several machine learning techniques (Neural Networks, Random Forests, Support Vector Machines, Gaussian Processes and KNN) are used in order to learn to pronounce words in English, given their written representation.
Even though the experiments carried out in this work are limited to English language, an outstanding advantage of using machine learning rather than a set of fixed rules to approach this problem is that the pronunciation of any language could be learnt, provided a reasonable set of examples is given.

![Results](https://i.ibb.co/BKFVKFr/image.png)

Approximately 85% of individual sounds (phonems) were predicted correctly in test, being ANN and Random Forests the top performers claassifiers. Naturally, vowel soulds were the most frequent type of misclassification due to the fact that English has 20 vowel sounds.

A complete report can be found in Informe/Informe.pdf (in Spanish)

# Index of folders and files:
```
rawDataset/                    
  |-> netTalk.data             Original NetTalk dataset
  |-> netTalk.names            Documentation of the original dataset

encodedDs/                     [Encoded NetTalk dataset & code used to generate it]
  |-> articulatoryFeatures/    (folder) Tables that store the encoded representation of phonems as articulatory features
  |-> generate_dataset.R       Preprocess original nettalk dataset, encode it as a matrix of numbers
  |-> datasets/                (folder) Encoded NetTalk dataset (created using scripts/generate_dataset.R)

ann/                           [All code related to neural networks] 
  |-> ann_impl\                
       |-> bp.c                Modified version of the code used in Machine Learning course to train ANNs          
  |-> optimize_parameters.ssh  Train several ANNs to find the best hyperparameters
  |-> ann_best_calculate.ssh   Calculate errors for different configurations (Using optimized hyperparameters)

rf/                            [All code calculating the errors for RF]
  |-> train_and_predict.R      Train RF and save the predictions
  |-> calculate_error.R        Load a file with predictions along with a test file, and calculate the discrete error
  |-> variable_importance.R    Calculate the importance of each input variable

svm/
  |-> train_optimize_and_predict.R  Optimize SVM hyperparameters, train and predict in test.
  |-> calculate_error.R             Calculate discrete error of predictions in test dataset

gp/
  |-> train_and_predict.R      Train GPs and save the predictions
  |-> calculate_error.R        Load a file with predictions along with a test file, and calculate the discrete error

knn/                           [All code implementing KNN and KNN-syll]

Informe/                       (folder) .tex and resulting .pdf file
```
** all scripts must be executed from their own subfolders **

[1] https://www.google.com/search?q=nettalk+paper&oq=nettalk+paper&aqs=chrome..69i57.2265j0j7&sourceid=chrome&ie=UTF-8
