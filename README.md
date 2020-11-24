# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains personal data about various people, using this data we seek to predict whether or not the "y" comlumn is equal to 1. This is a classification problem and the "y" column could be mean anything, because we didn't receive any details on the dataset. 

For myself the best performing model was the a Scikit-learn Pipeline model with a max-iter of 190 and a dropout value of 0.74.

## Scikit-learn Pipeline
For the Scikit-learn Pipeline we began with train.py. The module did the following:
* Download the data into a tabular dataset converting it to a dataframe, then cleaned it, converting all string values to numeric, and one-hot encoded categical variables.
* The main() was setup to receive max_iter and C parameters for the hyperdrive then complete a logistic regression algorithm model from the dataset to be scored on "Accuracy"
Next we udacity-project notebook as follows:
* "Register" the workspace in Azure
* Create the compute cluster
* Specify the Parameter Sampling and Early Stopping Policy
* Create an estimator using the SKLearn Class linked to our train.py file
* Create a HyperDriveConfig class with all the parameters we want for the Pipeline
* Submit the experiment with the HyperDriveConfig
* Save the best run by the "Accuracy"
* Download the best run model
* Save the best run model

I chose Random Paramter Sampling with a uniform distriution for both Max Iterations (I was unable to use .choice() for some reason, so I had to convert the max_iter to an Int in the train.py) and the L1 dropping factor. I beleive Random Sampling is a good choice since I would guessing at these hyperparameters anyways. The benefits using Random Sampling with uniform distibution were that I had a wide selection of varying runs to choose from.

I used the Bandit Policy Early Stopping Policy because it was already called in the notebook. I've never heard of it, but after reading the documentation I set the delay interval to 200 and the evalustion interval to 50. Since my Max Iteration was capped at 200 it meant that the early stopping policy did nothing.

## AutoML
The AutoML model that had the best accuracy was a voting ensemble, I was unable to decipher what the hyperparameters used were, but I did find that the two most important factors (columns) were Duration and Variable Rate.

For the preproccessing we imported and used the function from train.py, converting all "str"s to "Int"s where appropriate and categorical variables using one-hot encoding. This split the into 2 dataframes which we had to recombine and then transform into a tabular dataset even though the documentation shows I should be to pass a dataframe to the automl framework.

As noted above the classification technique that had the best results was a voting ensemble. Documentation shows the Azure uses a soft voting ensemble, which means that the probabilities for each "y" column of each model are added up, and if the sum of probabilities that it is 1 is higher than the sum of probabilities that it is 0 it chooses a vlue of 1 in that instance. There were about 27 models used in the voting ensemble, each model itself took less than 1 minute to run. Some examples of the classification techniques are: LightGBM, XGBoostClassifier, Stochastic Gradient Descent, Random Forest, Extreme RandomTrees. Voting ensemble is effective in improving models where most models have mostly the same predictions as was the case in this experiment.

## Pipeline comparison
Both models scored near identical, 0.91697 for the Scikit-learn Pipeline and 0.91667 for AutoML. I consider these to essentially be the same score. As for the architecture, there are some slight difference, after cleaning the data the you set the AutoMLConfig and it does it's work, testing various models and hyperparameters. The Scikit-learn Pipeline requires a little more coding in developing the model yourself, but you have greater control over the model. I feel like for the most part AutoML is sort of a grab all pipeline, where as Scki-Kit learn would be used for more in depth specialized models.

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**
For the Sci-Kit Learn pipeline, I blundered with the Early Termination policy. Going back I would definitely revise that. I noticed some of the higher scoring models generally had a higher max iteration as well, so I would increase that. I would probably narrow down the L1 dropping score because I had a pretty wide distibution.

For the AutoML there is not much you can do once you plug in the data. Maybe I could have let it run longer or change the number of cross folds, but I'm happy with that's models output.

## Proof of cluster clean up
![alt text](https://github.com/ilknish/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/Compute_Delete.PNG)
