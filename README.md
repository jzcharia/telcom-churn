# Telcom Customer Churn Model

  * Built model that estimates customer churn (leaving after one month) with 80% accuracy to help company to optimize marketing efforts and increase customer retention. 
  * Built three models: decision tree (base), stochastic gradient descent, and linear SVC classification model and optimized using GridSearchCV
  * Looked at feature importance to understand most impactful features in predicting churn

## Resources

**Major packages used:** Pandas, Numpy, Scikit-Learn, Matplotlib, Seaborn

**Source of Data:** https://www.kaggle.com/blastchar/telco-customer-churn

**Tools:** Jupyter Notebook, Anaconda Prompt

## Exploratory Data Analysis

Looked at various relationships between factors within dataset. Below are some highlight visuals that highlight the analysis:

![alt text] (https://github.com/jzcharia/telcom-churn/blob/master/EDA%20Images/ChurnFrequency.png "Churn Frequency")

## Model Buidling



# Future Improvements
 * **Feature Engineering -** the sample was not great. The data was sourced from Bangladesh which has a population that does not represent the USA. Specifically, we can see that in the dataset when looking at obesity and genital thrush rates. 
 * **Build API using Flask -** the web application is pretty plain. It allows you to answer some questions and it returns a prediction. Additional features to be added:
    1. A window to show actual probablity of event.
    2. A submit button -- currently it updates every time a change is made
    3. Resources for diabetes help to come up when the prediction is yes
 * **Class Imbalance-** options for at high risk or low risk would be a great way to show more subtly to the app. This can be based on probability or a new dataset to train for that. 

