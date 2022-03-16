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

![alt text](https://github.com/jzcharia/telcom-churn/blob/master/EDA%20Images/ChurnFrequency.png "Churn Frequency")
![alt text](https://github.com/jzcharia/telcom-churn/blob/master/EDA%20Images/CorrMatrix.png "Correlation Matrix")
![alt text](https://github.com/jzcharia/telcom-churn/blob/master/EDA%20Images/CustomerTenure.png "Customer Tenure")
![alt text](https://github.com/jzcharia/telcom-churn/blob/master/EDA%20Images/MonthlyCharges.png "Monthly Charges")

## Model Buidling

Built three models:
 1. **Decision Tree**- baseline model and decision trees are very easy to explain
 2. **Stochastic Gradient Descent**- powerful modeling technique but did not generalize well with the dataset. Model was overfitting. 
 3. **Linear Support Vector** - best model with 80 percent accuracy. 

# Future Improvements
 * **Feature Engineering -** there were features that could have been engineered to potentially create a better predictor. Ideas included:
   1. **Family** - if they were partnered and had kids, this probably means they were a family unit.
   2. **Count of Services** - self explainatory but amount of services (TV, phone, internet...)
   3. **Average cost of services** - amount of services/monthly charges
 * **Build API using Flask -** Pickle the model and create an endpoint API that programs can call and utilize
 * **Class Imbalance-** the target variable has a pretty evident class imbalance so techniques like upweighting could be beneficial

