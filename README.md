# Capstone Project Udacity Machine Learning Engineer Nanodegree

This repository hosts files for the Capstone Project of Udacity's Machine Learning Nanodegree with Microsoft Azure.

In this project I created two experiments; one using Microsoft Azure Machine Learning [Hyperdrive package](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive?view=azure-ml-py), and another using Microsoft Azure [Automated Machine Learning](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train?view=azure-ml-py) feature (referred to as AutoML going forward) with the [Azure Python SDK](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-ml-py).

The best models from the two experiments were compared based on the primary metric (AUC weighted score) and the best performing model was deployed and consumed using a web service.

## Project Workflow
![](https://github.com/ObinnaIheanachor/Capstone-Project-Udacity-Machine-Learning-Engineer/blob/master/images/capstone-diagram.png)

## Dataset

I made use of [IBM HR Analytics Employee Attrition & Performance Dataset](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset).
Employee Attrition affects every organization. The IBM HR Attrition Case Study is aimed at determining factors that lead to employee attrition and predict those at risk of leaving the company.

The Dataset consists of 35 columns, which will help us predict employee attrition. More information about the dataset can be found [here](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset)

### Task
This is a binary classification problem, where the outcome 'Attrition' will either be 'true' or 'false'. In this experiment, we will use `Hyperdrive` and `AutoML' to train models on the dataset based on the `AUC Weighted` metric. We will then deploy the model with the best performance and interact with the deployed model.

### Access
The data is hosted [here](https://github.com/ObinnaIheanachor/Capstone-Project-Udacity-Machine-Learning-Engineer/blob/master/data/attrition-dataset.csv) in this repository. 
I will use the `Tabular Dataset Factory's Dataset.Tabular.from_delimited_files()` operation to get the data from the url and save it to the datastore by using dataset.register().

## Automated ML
Automated machine learning picks an algorithm and hyperparameters for you and generates a model ready for deployment. There are several options that you can use to configure automated machine learning experiments.
This is a binary classification problem with label column 'Attrition' having output as 'true' or 'false'. 25 mins is the experiment_timeout_duration, a maximum of 5 concurrent iterations take place together, and the primary metric is AUC_weighted.

The AutoML configurations used for this experiment are:

| Auto ML Configuration | Value | Explanation |
|    :---:     |     :---:      |     :---:     |
| experiment_timeout_minutes | 25 | Maximum duration in minutes that all iterations can take before the experiment is terminated |
| max_concurrent_iterations | 5 | Represents the maximum number of iterations that would be executed in parallel |
| primary_metric | AUC_weighted | This is the metric that the AutoML runs will optimize for when selecting the best performing model |
| compute_target | cpu_cluster(created) | The compute target on which we will run the experiment |
| task | classification | This is the nature of our machine learning task |
| training_data | dataset(imported) | The training data to be used within the experiment |
| label_column_name | Attrition | The name of the label column |
| path | ./capstone-project | This is the path to the project folder |
| enable_early_stopping | True | Enable early termination if the score is not improving |
| featurization | auto | Indicates if featurization step should be done automatically or not, or whether customized featurization should be used |
| debug_log | automl_errors.log | The log file to write debug information to |

![](https://github.com/ObinnaIheanachor/Capstone-Project-Udacity-Machine-Learning-Engineer/blob/master/images/auto%20ml%20config.PNG)


### Results
The best performing model is the `VotingEnsemble` with an AUC_weighted value of **0.823**. A voting ensemble (or a “majority voting ensemble“) is an ensemble machine learning model that combines the predictions from multiple other models. It is a technique that may be used to improve model performance, ideally achieving better performance than any single model used in the ensemble. This balances out the individual weaknesses of the considered classifiers.

**Run Details**

![](https://github.com/ObinnaIheanachor/Capstone-Project-Udacity-Machine-Learning-Engineer/blob/master/images/automl%20run%20details.PNG)

**Best Model**

![](https://github.com/ObinnaIheanachor/Capstone-Project-Udacity-Machine-Learning-Engineer/blob/master/images/automl%20best%20model.PNG)

![](https://github.com/ObinnaIheanachor/Capstone-Project-Udacity-Machine-Learning-Engineer/blob/master/images/auto%20ml%20best%20run.PNG)

### Improve AutoML Results
* Increase experiment timeout duration: This would allow for more model experimentation, but might be costly.
* Try a different primary metric: We can explore other metrics like `f1 score`, `log loss`, `precision - recall`, etc. depending on the nature of the data you are working with.
* Engineer new features that may help improve model performance..
* Explore other AutoML configurations.


## Hyperparameter Tuning
Decision tree builds classification or regression models in the form of a tree structure. It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. Decision trees can also handle both categorical and numerical data and was suitable for our classification task.
I also used a Decision Tree model because it is simple to understand, interpret, and visualize.


### Results
Hyperparameters are adjustable parameters that let you control the model training process. Azure Machine Learning lets you automate hyperparameter tuning and run experiments in parallel to efficiently optimize hyperparameters as the prrocess can be computationally expensive and manual.
HyperDrive configuration 
| Configuration | Value | Explanation |
|    :---:     |     :---:      |     :---:     |
| hyperparameter_sampling | Value | Explanation |
| policy | early_termination_policy | The early termination policy to use |
| primary_metric_name | AUC_weighted | The name of the primary metric reported by the experiment runs |
| primary_metric_goal | PrimaryMetricGoal.MAXIMIZE | One of maximize / minimize. It determines if the primary metric has to be minimized/maximized in the experiment runs' evaluation |
| max_total_runs | 12 | Maximum number of runs. This is the upper bound |
| max_concurrent_runs | 4 | Maximum number of runs to run concurrently. |
| estimator | 4 | An estimator that will be called with sampled hyper parameters |

The Hyperparameters for the Decision Tree are:
| Hyperparameter | Value | Explanation |
|    :---:     |     :---:      |     :---:      |
| criterion | choice("gini", "entropy") | The function to measure the quality of a split. |
| splitter | choice("best", "random") | The strategy used to choose the split at each node. |
| max_depth | choice(3,4,5,6,7,8,9,10) | The maximum depth of the tree. |

### HyperDrive Results
The best performing model using HyperDrive had Parameter Values as `criterion` = **gini**, `max_depth` = **8**, `splitter` = **random**. The AUC_weighted of the Best Run is **0.730**.

![](https://github.com/ObinnaIheanachor/Capstone-Project-Udacity-Machine-Learning-Engineer/blob/master/images/best%20hyperdrive%20model.PNG

![](https://github.com/ObinnaIheanachor/Capstone-Project-Udacity-Machine-Learning-Engineer/blob/master/images/hyperdrive%20runs.PNG)

**Run Details**

![](https://github.com/ObinnaIheanachor/Capstone-Project-Udacity-Machine-Learning-Engineer/blob/master/images/hyperdrive%20run%20completed.PNG)

**Visualization of Runs**

![](https://github.com/ObinnaIheanachor/Capstone-Project-Udacity-Machine-Learning-Engineer/blob/master/images/hyperdrive%20output.PNG)

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

![](https://github.com/ObinnaIheanachor/Capstone-Project-Udacity-Machine-Learning-Engineer/blob/master/images/automl%20webservice.PNG)

## Standout Suggestions
**Enable Logging**

Application Insights is a very useful tool to detect anomalies, and visualize performance. It can be enabled before or after a deployment. To enable Application Insights after a model is deployed, you can use the python SDK.
I enabled logging by running the [logs.py](https://github.com/ObinnaIheanachor/Capstone-Project-Udacity-Machine-Learning-Engineer/blob/master/logs.py) script.

**Running logs.py**

![](https://github.com/ObinnaIheanachor/Capstone-Project-Udacity-Machine-Learning-Engineer/blob/master/images/automl%20enable%20logging.PNG)

**Application Insights Enabled**

![](https://github.com/ObinnaIheanachor/Capstone-Project-Udacity-Machine-Learning-Engineer/blob/master/images/automl%20enable%20logging2.PNG)

## Screen Recording

An overview of this project can be found [here](https://youtu.be/JC7_Anw-Fa8)


