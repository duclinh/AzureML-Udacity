# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
#### Dataset
Our dataset contains data about the bank marketing. 
RangeIndex: 32950 entries, 0 to 32949, Data columns (total 21 columns).

The classification goal is to predict whether the client will subscribe a bank term deposit (column y).

#### the compute cluster info:

- vm_size: Standard_D2_V2
- min_nodes: 0
- max_nodes: 4

#### Solution and Result

For this classification problem to approaches were used:

- Apply a Scikit-learn Logistic Regression model, optimizing its hyperparameters using HyperDrive. 
    + Best Run Id:  HD_5d3e03e5-7aad-4e92-90ec-7238be70d355_7
    + Accuracy: 0.9171471927162367
- Use Azure Auto ML to build and optimize a model on the same dataset

## Scikit-learn Pipeline
#### Parameter sampler

I specified the parameter sampler as such:

ps = RandomParameterSampling(
    {
    '--C': choice(0.01, 0.1, 0.2, 0.5, 0.7, 1.0),
    '--max_iter': choice(range(10,110,10))
    }
)

I chose discrete values with choice for both parameters, C and max_iter.

C is the Regularization while max_iter is the maximum number of iterations.

##### Early stopping policy

An early stopping policy is used to automatically terminate poorly performing runs thus improving computational efficiency. I chose the BanditPolicy which I specified as follows:

policy = BanditPolicy(evaluation_interval=2, 
                      slack_factor=0.1)

- evaluation_interval: This is optional and represents the frequency for applying the policy. Each time the training script logs the primary metric counts as one interval.

- slack_factor: The amount of slack allowed with respect to the best performing training run. This factor specifies the slack as a ratio.

#### Our best model in the Scikit-learn Pipeline is:
- Best Run Id: HD_5d3e03e5-7aad-4e92-90ec-7238be70d355_7
- Accuracy: 0.9171471927162367

## AutoML

the following configuration for the AutoML run:

automl_config = AutoMLConfig(
    experiment_timeout_minutes=15,
    task='classification',
    primary_metric='accuracy',
    compute_target=cpu_cluster,
    max_concurrent_iterations=4,
    training_data=train_clean_data,
    test_data=test_clean_data,
    label_column_name='y',
    n_cross_validations=2
)

- experiment_timeout_minutes=15. This is an exit criterion and is used to define how long, in minutes, the experiment should continue to run. To help avoid experiment time out failures, I used the minimum of 15 minutes.

- task='classification'. This defines the experiment type which in this case is classification.

- primary_metric='accuracy'. accuracy as the primary metric.

- enable_onnx_compatible_models=True. to enable enforcing the ONNX-compatible models.

- n_cross_validations=2. to sets how many cross validations to perform, based on the same number of folds (number of subsets). As one cross-validation could result in overfit, in my code I chose 2 folds for cross-validation; thus the metrics are calculated with the average of the 2 validation metrics.

My Result is:
- Accuracy: 0.91077
- AUC macro: 0.94005
- AUC micro: 0.97811
- AUC weighted: 0.94005

## Pipeline comparison
Comparison of the two models and their performance. Differences in accuracy & architecture - comments

#### HyperDrive Model	
- Id: HD_5d3e03e5-7aad-4e92-90ec-7238be70d355_7
- Accuracy: 0.9171471927162367

#### AutoML Model	
- id:	AutoML_94820f21-4bcc-442d-bd9c-95130ddedb92	
- Accuracy: 0.91077
- AUC macro: 0.94005
- AUC micro: 0.97811
- AUC weighted: 0.94005
- Algortithm: VotingEnsemble

## Future work
I'll try other parameter samplers to compare the Bayesian sampling, such as Grid Sampling, Random Sampling and others.

## Proof of cluster clean up

Compute name: udacity-project
Compute type: Compute instance
Subscription ID: cdbe0b43-92a0-4715-838a-f2648cc7ad21
Resource group: aml-quickstarts-267698
Workspace: quick-starts-ws-267698
Region: southcentralus
Created by: ODL_User 267698
Assigned to: ODL_User 267698
