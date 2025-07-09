# Predictive modelling: travel insurance

This repo contains an example of applying a predictive model on a dataset to solve a business problem. A tour & travels company from India launched a new travel insurance offer with some of their customers in 2019. Since that time, the company has collected data about various customers who bought – or didn't buy – this new travel insurance. The dataset that is being analysed here contains some of this data from the period of the introductory offering of this service in 2019.

The goal of the analysis contained here is to build a predictive classifier model that would identify customers likely to be interested in the aforementioned travel insurance offering, based on customers' data such as age, income, and education. Such a model can be applied in many ways to solve various business problems, like building a target audience for a marketing compaign of the insurance service, or for measuring the actual versus expected sales of the service, among others.

Our analysis consists of 2 main steps:
1. Exploration and statistical analysis of the customers data
2. Building a pipeline and a predictive model that classifies customers

You can find all of this and a summary of what was achieved in the notebook file: `Travel_insurance_analysis_and_model.ipynb`.

## The dataset

The dataset is publicly accessible on [Kaggle](https://www.kaggle.com/datasets/tejashvi14/travel-insurance-prediction-data). You can find more info about the dataset there, as well as in the notebook provided here.

In short, it's a small dataset with just under 2,000 instances and 9 variables. The response is a binary variable that indicates whether the customer has bought the new travel insurance or not. The predictors describe various features of the customers. Some of them are discrete, such as age, income, and family size. The rest are binary, such as employment type (private,self-employed/government), graduation (yes/no), chronic diseases (yes/no), frequent flyer (yes/no), and ever travelled abroad (yes/no).

## Contents of this repo

- `Travel_insurance_analysis_and_model.ipynb` is the Jupyter notebook file presenting the analysis, modelling, and the results
- `TravelInsurancePrediction.csv` is the source dataset
- `utilities.py` is a python file containing various helper functions used in the analysis

## Replicating the contents of this repo

You may replicate this repo on your local environment or continue building upon it. Just make sure to follow the prerequisites.

**Prerequisites:**
1. Python 3.9.6 or higher
2. [Jupyter notebook](https://jupyter.org/install)
3. Install the most recent version of python libraries used in the project so far via your preferred package manager such as conda or pip.

The following command line argument example for pip will install all the necessary libraries:
```
pip install numpy pandas matplotlib seaborn scipy statsmodels imblearn sklearn
```
Feel free to use `git clone` to download the contents to your machine. If you have any questions or suggestions, contact me at jackunui@gmail.com