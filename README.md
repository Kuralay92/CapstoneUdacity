# Capstone Project Udacity Machine Learning Engineer Nanodegree
This project I have the opportunity to use the knowledge obtained from this Nanodegree to solve an interesting problem. In this project, I will create two models: one using Automated ML (denoted as AutoML from now on) and one customized model whose hyperparameters are tuned using HyperDrive. I will then compare the performance of both the models and deploy the best performing model.

In this  will demonstrate my ability to use an external dataset in your workspace, train a model using the different tools available in the AzureML framework as well as your ability to deploy the model as a web service.
<img width="514" alt="image" src="https://user-images.githubusercontent.com/49708694/169706551-79b726fd-21e5-489e-b625-d0aa6fc6c732.png">

## Dataset

This is the classic marketing bank dataset uploaded originally in the UCI Machine Learning Repository. The dataset gives you information about a marketing campaign of a financial institution in which you will have to analyze in order to find ways to look for future strategies in order to improve future marketing campaigns for the bank.
https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset

### Task
This is a binary classification problem, where the outcome 'Y' will either be 'yes' or 'no'. In this experiment, we will use `Hyperdrive` and `AutoML' to train models on the dataset based on the `AUC Weighted` metric. We will then deploy the model with the best performance and interact with the deployed model.

### Access
The data is hosted in this repository.[here]https://github.com/Kuralay92/CapstoneUdacity/blob/607123a40a3d05d1c2043edf5fb9bcb18dd46f86/bankmarketing_train%20(2).csv 
I will use the `Tabular Dataset Factory's Dataset.Tabular.from_delimited_files()` operation to get the data from the url and save it to the datastore by using dataset.register().

## Automated ML
AutoML or Automated ML is the process of automating the task of machine learning model development. Using this feature, you can predict the best ML model, and its hyperparameters suited for your problem statement.

For this experiment in Azure ML Studio we train a model using the on the bank marketing dataset with Automated ML, we create a new compute cluster, and run the AutoML experiment.

![](https://github.com/Kuralay92/CapstoneUdacity/blob/4bbe8b552bbb7981fa9c26da6cf0eb47c5521b1d/screenshot/Screenshot%202022-05-22%20214450.png)
And above you can see all screenshots how Auto Ml model is created and successfully deployed 
### Results
The best performing model is the `VotingEnsemble` with an AUC_weighted value of **0.89**. A voting ensemble (or a “majority voting ensemble“) is an ensemble machine learning model that combines the predictions from multiple other models. It is a technique that may be used to improve model performance, ideally achieving better performance than any single model used in the ensemble. This balances out the individual weaknesses of the considered classifiers.

**Run Details**

![](https://github.com/Kuralay92/CapstoneUdacity/blob/2924f921e48f9192ca6c81e621f369d97640ac8c/screenshot/Screenshot%202022-05-22%20104140.png)
 
**Best Model**

![](https://github.com/Kuralay92/CapstoneUdacity/blob/4633c5629a2dac8ac69acc30535ca3b51fa8d1ed/screenshot/Screenshot%202022-05-22%20105131.png)

![](https://github.com/Kuralay92/CapstoneUdacity/blob/f53dd0b4075a06fc4a9ec410c02e4887028cf4d7/screenshot/accuracy.png)

### Improve AutoML Results
* Increase experiment timeout duration: This would allow for more model experimentation, but might be costly.
* Try a different primary metric: We can explore other metrics like `f1 score`, `log loss`, `precision - recall`, etc. depending on the nature of the data you are working with.
* Engineer new features that may help improve model performance..
* Explore other AutoML configurations.

Project's dataset contains marketing data about individuals. The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict whether the client will subscribe a bank term deposit (column y).

The best performing model was the **HyperDrive model** with  Run Id:  HD_fd4947e4-5b87-414b-a592-6c5d6cfe4496_10. It derived from a Scikit-learn pipeline and had an accuracy of **0.91**. In contrast, for the **AutoML model** with IAutoML_e8ea9e9e-8768-44c2-bcf1-8165282894b7_36, the accuracy was **0.89** and the algorithm used was VotingEnsemble.

## Hyperparameter Tuning

**Parameter sampler**

I specified the parameter sampler as such:

```
ps = RandomParameterSampling(
    {
        '--C' : choice(0.001,0.01,0.1,1,10,20,50,100,200,500,1000),
        '--max_iter': choice(50,100,200,300)
    }
)
```

I chose discrete values with _choice_ for both parameters, _C_ and _max_iter_.

_C_ is the Regularization while _max_iter_ is the maximum number of iterations.

_RandomParameterSampling_ is one of the choices available for the sampler and I chose it because it is the faster and supports early termination of low-performance runs. If budget is not an issue, we could use _GridParameterSampling_ to exhaustively search over the search space or _BayesianParameterSampling_ to explore the hyperparameter space. 

**Early stopping policy**

An early stopping policy is used to automatically terminate poorly performing runs thus improving computational efficiency. I chose the _BanditPolicy_ which I specified as follows:
```
policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1)
```
_evaluation_interval_: This is optional and represents the frequency for applying the policy. Each time the training script logs the primary metric counts as one interval.

_slack_factor_: The amount of slack allowed with respect to the best performing training run. This factor specifies the slack as a ratio.

Any run that doesn't fall within the slack factor or slack amount of the evaluation metric with respect to the best performing run will be terminated. This means that with this policy, the best performing runs will execute until they finish and this is the reason I chose it.

### HyperDrive Results
The best performing model using HyperDrive had Parameter Values as '--C', '0.001', '--max_iter', '100'

![](https://github.com/Kuralay92/CapstoneUdacity/blob/38965f48442c254abe3a33f5f42576808437dddc/screenshot/Best%20model.png)

![](https://github.com/Kuralay92/CapstoneUdacity/blob/ce64559b448e0bcbc9a0b607743e4564eb4f9479/screenshot/HdSubmit.png)

**Run Details**

![](https://github.com/Kuralay92/CapstoneUdacity/blob/94a27f2fa57df0371143ee23de5577ec8b99b16f/screenshot/Result.png)

### Improve HyperDrive Results
* Choose a different algorithm to train the dataset on like `Logistic Regression`, `xgboost`, etc.
* Choose a different clssification metric to optimize for
* Choose a different termination policy like `No Termination`, `Median stopping`, etc.
* Specify a different sampling method like `Bayesian`, `Grid`, etc.

## Model Deployment
The AutoML model outperforms the HyperDrive model so it will be deployed as a web service. Below is the workflow for deploying a model in Azure ML Studio;

* **Register the model**: A registered model is a logical container for one or more files that make up your model. After you register the files, you can then download or deploy the registered model and receive all the files that you registered.
* **Prepare an inference configuration (unless using no-code deployment)**: This involves setting up the configuration for the web service containing the model. It's used later, when you deploy the model.
* **Prepare an entry script (unless using no-code deployment)**: The entry script receives data submitted to a deployed web service and passes it to the model. It then takes the response returned by the model and returns that to the client.
* **Choose a compute target**: This involves choosing a compute target to host your model. Some common ones include `Azure Kubernetes Service (AKS)`, and `Azure Container Instances (ACI)`.
* **Deploy the model to the compute target**: You will need to define a deployment configuration and this depends on the compute target you choose. After this step, you are now ready to deploy your model.
* **Test the resulting web service**: After deployment, you can interact with your deployed model by sending requests (input data) and getting responses (predictions).

**Healthy Deployed State**

![](https://github.com/Kuralay92/CapstoneUdacity/blob/c3feb4ca31aa58ffb59c8bc064a89315d7d84ba3/screenshot/endpoint.png)

## Standout Suggestions
**Enable Logging**

Application Insights is a very useful tool to detect anomalies, and visualize performance. It can be enabled before or after a deployment. To enable Application Insights after a model is deployed, you can use the python SDK.

![](https://github.com/Kuralay92/CapstoneUdacity/blob/a9a9a22eba3719b77d1add8c6ee06aeef19e271d/screenshot/appenabled.png)
**Application Insights Enabled**

![](https://github.com/Kuralay92/CapstoneUdacity/blob/477c098728f03ad3800b08576bd2405e0e258dea/screenshot/appenabler.png)

## Screen Recording

An overview of this project can be found [here](https://www.youtube.com/watch?v=Y377_pp3kNw)


