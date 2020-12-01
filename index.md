## Portfolio

---

### Unsupervised Learning Project: Airplane Crashed
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/apham15/Airplane_Crashed) 
[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1EcLvcgNmkrYZBPraaOk8EKw9X_KMONN2?usp=sharing)

I am a traveler, who always use airplane as the main transportation to discover the world. And My favorite desination is Southest Asia, where I can do snokerling, kayaking, and diving. As a result, when I found this dataset, all my curiousities increase. Let's learn what was the airplane crashing from the past, I may know some insightful that would help me to decide for my future trips. Also, as an Asian, Chinese Zodiac is something very interesting. I did some fun stuff with that to learn more about the data.

![airports-world-network](https://user-images.githubusercontent.com/63126292/100527674-00053d00-319a-11eb-8c97-93025d1dc632.png)


### Outcome

#### 1. Airplane Crashing

* USA has the highest fatility cases in the world
* There are over 85% of airplan crashed which are from commercial flights
* Don't fly with Aeroflot Operator, there is 68% chance of you dying.
* Althout it is a military airplane, Don't fly in a Douglas DC-3 aircraft, you are 5 times more chance to dir and it had highest fatalities percentage.
* Don't take any flight that flies Tenerife - Las Palmas / Tenerife - Las Palmas route or Tokyo - Osaka.
* Avoid going to Sao Paulo, Brazil ; Moscow, Russia ; Rio de Janeiro, Brazil ; they had highest plane crash location.
* It is so much safer to take flight now-a-days as compared to 1970-80, 1972 was the worst years for airline industry.
* Peole who are born in year of Ox have more chance to die, and people who are born in year of Horse have high chance to survive
![download (9)](https://user-images.githubusercontent.com/63126292/100661775-9a3fbf00-3319-11eb-8103-5b35c5bc2d9e.png)
![download (10)](https://user-images.githubusercontent.com/63126292/100661776-9ad85580-3319-11eb-8108-c98376f7e609.png)

#### 2. Airplane Fatalities Clustering
* The best clustering algorithms for clustering the fatilities is DBSCAN with eps = 0.01, min sample is 1, and metric is euclidean, which has 2 clusters

#### 3. Text Clustering
* It's hard to determine which is the best method. Overall, The best model which has the highest Silhoette Coefficient Score (0.065) is DBSCAN with eps = 0.01, min_sample = 1, and metric etric='euclidean', which has 4627 clusters
* Personally, I like K-mean because it returns 16 cluster, which is easy to understand and visuallize. 
![download (12)](https://user-images.githubusercontent.com/63126292/100666428-9b271f80-331e-11eb-95e2-25bdff6ff931.png)
![download (13)](https://user-images.githubusercontent.com/63126292/100666430-9c584c80-331e-11eb-99c1-60147de4b9ad.png)


#### 4. Learning from this project
* There is no absolute right answer for clustering. To bring the best results for business solution, I would recomend data scientists consider the best model that fit the business need and apply their business acumen to suggest the best strategy for the whole team.
* My data analysis above is good for travelers around the world to have an insider about traviling with airplane. And it is a good suggestion for operators or aerospace company to look up the disavantages of the past to improve their future businesses.

---
### Unsupervised Learning Project: Fashion-MNIST
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/apham15/Fashion_MNIST) 
[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1cdfu56A7ze5n3v5mvs-dneJaZT4jtGmo?usp=sharing)

`Fashion-MNIST` is a dataset of [Zalando](https://jobs.zalando.com/tech/)'s article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. `Fashion-MNIST` serves as a direct **drop-in replacement** for the original [MNIST dataset](http://yann.lecun.com/exdb/mnist/) for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

![Screen Shot 2020-11-19 at 12 50 04 AM](https://user-images.githubusercontent.com/63126292/99631413-2f30e700-2a01-11eb-84aa-6ce1fdc6ddbd.png)

#### Outcome
* The best Clustering Model is KMeans with the number of cluster is 4

* The best dimensional reduction to visualize is UMAP with n_neighbors = 5 and min_dist = 0.3 or n_neighbors = 7 and min_dist = 1

***3D graph for UMAP***
![newplot](https://user-images.githubusercontent.com/63126292/100052122-e69f7200-2de2-11eb-8253-c4cad0dcb0d8.png)

#### Learning from this project
* There is no absolute right answer for clustering. To bring the best results for business solution, I would recomend data scientists consider the best model that fit the business need and apply their business acumen to suggest the best strategy for the whole team.
* My work is very helpful for business solution teams in fashion industry. They can apply it to build the similar filters to select out what customers need/want. Also, if they do have some numerical dataset, my work can be an add-on for predicting the outcome base of the business model need.

---
### Supervised Learning Project: Loan Prediction
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/apham15/Loan_Prediction) 
[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1AinDO8z5Ud7OVTSx6KS3yGUauJqzzGhq?usp=sharing)
[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://github.com/apham15/Loan_Prediction/blob/main/Supervised_Learning_Project_Loan_Prediction_final_version.ipynb)

Lending Club is "the world's largest online marketplace connecting borrowers and investors." It is a peer-to-peer lending network that open sources some of its loan data to the community.

I use the loan.csv file that contains real-world, historical data on loans organized by Lending Club between 2007 and 2011.

![Screen Shot 2020-10-13 at 11 01 14 PM](https://user-images.githubusercontent.com/63126292/95942130-01280980-0da8-11eb-88a8-cfa750120469.png)

1. Business Understanding

You work for a consumer finance company which specialises in lending various types of loans to urban customers. When the company receives a loan application, the company has to make a decision for loan approval based on the applicant’s profile. Two types of risks are associated with the bank’s decision:

If the applicant is likely to repay the loan, then not approving the loan results in a loss of business to the company

If the applicant is not likely to repay the loan, i.e. he/she is likely to default, then approving the loan may lead to a financial loss for the company

The dataset given contains the information about past loan applicants and whether they ‘defaulted’ or not. The aim is to identify patterns which indicate if a person is likely to default, which may be used for taking actions such as denying the loan, reducing the amount of loan, lending (to risky applicants) at a higher interest rate, etc.

![Screen Shot 2020-10-13 at 11 33 17 PM](https://user-images.githubusercontent.com/63126292/95943851-89101280-0dac-11eb-85c6-39e7db5ef5b4.png)

2. Business Concept
When a person applies for a loan, there are two types of decisions that could be taken by the company:

Loan accepted: If the company approves the loan, there are 3 possible scenarios described below:

a. Fully paid: Applicant has fully paid the loan (the principal and the interest rate)

b. Current: Applicant is in the process of paying the instalments, i.e. the tenure of the loan is not yet completed. These candidates are not labelled as 'defaulted'.

c. Charged-off: Applicant has not paid the instalments in due time for a long period of time, i.e. he/she has defaulted on the loan

Loan rejected: The company had rejected the loan (because the candidate does not meet their requirements etc.). Since the loan was rejected, there is no transactional history of those applicants with the company and so this data is not available with the company (and thus in this dataset)


### Outcome
* The best performance models are Gradient Boosting Classifier (87.57% accuracy) and Random Forest Classifier (87.29% accuracy)
* Apply GridSearchCV for tunning and find out the best GBM parameter is {'max_depth': 15, 'max_features': 'auto', 'n_estimators': 15} and the best Random Forest parameter is {'max_depth': 38, 'max_features': 'sqrt', 'n_estimators': 50}
* The optimized model achieved an ROCs are 99.99% (GBM) and 99.03% (Random Forest)
![download (1)](https://user-images.githubusercontent.com/63126292/97929804-97a87480-1d2f-11eb-8692-e167f0c4e26e.png)
![download](https://user-images.githubusercontent.com/63126292/97929852-b9096080-1d2f-11eb-9c42-036d103db5a1.png)

---
### Supervised Learning Project: Wine Quality Prediction
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/apham15/Loan_Prediction) 
[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1AinDO8z5Ud7OVTSx6KS3yGUauJqzzGhq?usp=sharing)
[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://github.com/apham15/Loan_Prediction/blob/main/Supervised_Learning_Project_Loan_Prediction_final_version.ipynb)

![wine-images-high-resolution-free-download](https://user-images.githubusercontent.com/63126292/97619738-96eda680-19ee-11eb-8b50-323cd9900bca.jpg)

1. Objectives: 
* The goal of this kernel is to find the best approach to identify the quality of wines. We will go through the basic EDA and visually identify the quality 
Moreover, I also applied multiple ML models to make the prediction respectively. Each of the models would have its own strength.
* There are two different notebooks. One is for red wine, and another one is for white wine
* My research is good for alcohol beverage companies and customers who love to learn more about wine

2. Dataset
* The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine. For more details, consult the reference [Cortez et al., 2009]. Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).
* Acknowledgements
 This dataset is also available from the [UCI machine learning repository](https://archive.ics.uci.edu/ml/datasets/wine+quality) , I just shared it to kaggle for convenience. (I am mistaken and the public license type disallowed me from doing so, I will take this down at first request. I am not the owner of this dataset.
* P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

3. Outcome
* The best model for Red Wine is Logistic Regression (AUC 56.9%)
* The best model for White Wine is Random Forest Classifier (AUC 70.2%)

---
<center>© 2020 Anh (Andrew) Pham. Powered by Jekyll and the Minimal Theme.</center>
