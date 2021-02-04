# Portfolio

---
## Amazon Fine Food Review 
### Using Statistical Analysis (Sparse Matrix, Descriptive Analysis), Extract Component with TruncatedSVD, Text Processing (NLTK stopword, TF-IDF, Bag of Word, Word to Vector), Machine Learning Algorithms (BernoulliNB, Logistic Regression), K-means Clustering Method, Visualization with Plotly, Deep Learning with Kera (Tokenizer, ANN, RNN - LSTM)

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/apham15/Amazon_Fine_Food_Review) 
[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/152Bde9BARD6lZczGvFlDGL91nATMPRRl?usp=sharing)
[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://github.com/apham15/Amazon_Fine_Food_Review/blob/main/amazon%20food%20review.ipynb)

[Amazon.com, Inc.](https://www.amazon.com/) is an American multinational technology company based in Seattle, Washington, which focuses on e-commerce, cloud computing, digital streaming, and artificial intelligence. The fine food data span a period of more than 10 years, including all ~500,000 reviews up to October 2012. Reviews include product and user information, ratings, and a plaintext review.
The dataset belongs to [Stanford Network Analysis Project](https://snap.stanford.edu/data/web-FineFoods.html)

![31d13c99ee841869ca44ef54ba956272](https://user-images.githubusercontent.com/63126292/106423731-a550c180-6426-11eb-84a7-ca4f738f5ae2.png)

* After understanding the Amazon Fine Food review dataset, as a data scientist, I have few questions to set up the outline that helps me dive into the project:

> What is the connection between the food review score with the reviews and the products?

> Any correlation between products and top users who often write reviews?

> Can I extract the top product based on users’ recommendation?

> What are the top words that help business to understand whether it is a good review or not?

> Can I predict the positive and the negative reviews?

> For building a better prediction, should I choose machine learning algorithms or deep learning model?


* Those questions help me to separate the dataset into two parts:

> One is the correlation of userID, producID, review score to bring up the business solution: recommend food product item

> Another is the correlation of plaintext reviews with the sentiment analysis

### 1. Outcome
Words in Positive Reviews

![download (6)](https://user-images.githubusercontent.com/63126292/106503106-d82ca100-648a-11eb-85bb-b7b8486e0e17.png)

Word in Negative Reviews

![download (7)](https://user-images.githubusercontent.com/63126292/106503136-e24e9f80-648a-11eb-9649-70ffc90acbad.png)


a. Recommendation system with Sparse Matrix ```scipy.sparse.linalg.svds```

a.1. Building Popularity Recommender system

![Screen Shot 2021-02-01 at 12 18 47 AM](https://user-images.githubusercontent.com/63126292/106421861-1c845680-6423-11eb-806f-18718e28b9ff.png)

* Since this is a popularity-based recommender model, recommendations remain the same for all users
* We predict the products based on the popularity. It is not personalized to particular user

a.2. Building Collabrating Filtering
* Model-based Collaborative Filtering is a personalised recommender system, the recommendations are based on the past behavior of the user and it is not dependent on any additional information.

![Screen Shot 2021-02-01 at 12 13 01 AM](https://user-images.githubusercontent.com/63126292/106421595-9a942d80-6422-11eb-8261-6f57cfae269b.png)
![Screen Shot 2021-02-01 at 12 12 54 AM](https://user-images.githubusercontent.com/63126292/106421671-bc8db000-6422-11eb-9855-b83303eafce4.png)

* Based on the real value and the predict value, it is clear to see that the predictive recomendation system is great
* The Popularity-based recommender system is non-personalised and the recommendations are based on frequecy counts, which may be not suitable to the user.You can see the differance above for the user id 70 and 100, The Popularity based model has recommended the same set of 5 or 6 products to both but Collaborative Filtering based model has recommended entire different list based on the user past purchase history
* To evaluate the model, I apply RMSE score to test whether it is a good one or not. For RMSE, the lower of the value is, the better the performance of the model is. ```RMSE SVD Model = 0.0069574634757268656```


b. Sentiment Analysis

b.1 Machine Learning Algorithms

* Accuracy scores

![Screen Shot 2021-02-01 at 12 20 48 AM](https://user-images.githubusercontent.com/63126292/106421972-5fdec500-6423-11eb-9edf-4e5de62c1a93.png)

* Log Loss

![Screen Shot 2021-02-01 at 12 20 55 AM](https://user-images.githubusercontent.com/63126292/106422031-79800c80-6423-11eb-816b-c85973174c77.png)

* TF-IDF is the best model that has the highest accuracy score for both NernoulliNB and Logistic regression

![Screen Shot 2021-02-01 at 12 26 21 AM](https://user-images.githubusercontent.com/63126292/106422417-2a86a700-6424-11eb-9090-13ce94a73037.png)

* Logistic Regresison is the best model that fit in this dataset because it bring the highest accuracy score with the lowest log loss
* Tuning model, the best parameters set fo Logistic Regression is ```lr__C: 100.0, 	lr__penalty: 'none', lr__solver: 'saga'``` with the highest accuracy score is 93%

b.2 Clustering the top words with K-mean
* The best number of cluster is 3 with the highest Silhoutte Score is 0.002
* Top 10 words are ['strong', 'br', 'cup coffee', 'tea',  'taste',  'like', 'cups', 'flavor', 'coffee', 'love', 'one', 'food', 'good', 'cup', 'great', 'bold', 'br br', 'product']

![newplot](https://user-images.githubusercontent.com/63126292/106422475-438f5800-6424-11eb-8abb-936c32d58c9f.png)

b.3 Deep Learning
* Developing both ANN and RNN-LSTM, LSTM is the best one with the best accuracy score is 96%

![download](https://user-images.githubusercontent.com/63126292/106423195-961d4400-6425-11eb-8ad6-291f80cf2f04.png)
![download (1)](https://user-images.githubusercontent.com/63126292/106423224-a6352380-6425-11eb-8ad2-b924ba4855b8.png)

### 2. Conclusion
* Amazon Fine Food Review dataset is the incredible one. It allows me to utilize all my skills: statistical analysis, supervised learning methods, unsupervised learning methods, natural language processing, machine learning algorithm, and deep learning. 
* I used to confuse about the Sparse Matrix, but with this dataset, I completely understand how the matrix works by researching and applying it into my personal recommendation system. Furthermore, it is priceless to bring the relevence of statistic to the real world by utilizing it into the hand-on project.
* From this dataset, I learn that with deep learning, everything is so simple. Kereas class with tokenzie to vectorize word is faster than the TF-IDF traditional method. Also, RNN-LSTM maximized the plaintext review vectorized to bring up the better accuracy score for future sentiment analysis
* My work is useful for all type of e-commerce because it can apply for both strategy team and customer service team to help the business to be better.

---
## NLP Project: Twitter Sentiment Analysis
### Using Text Processing (NLTK stopword, TF-IDF, Bag of Word, Word to Vector), Extract Component with TruncatedSVD, and Machine Learning Algorithms (BernoulliNB, Logistic Regression, Random Forest, XGBoost)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/apham15/Twitter_Sentiment_Analysis) 
[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1Cz0w8z4IWivmNacHSOBPnD_x7A6gXIca?usp=sharing)
[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://github.com/apham15/Twitter_Sentiment_Analysis/blob/main/NLP%20-%20sentiment%20analysis.ipynb)

[Twitter](https://twitter.com/) is an American microblogging and social networking service on which users post and interact with messages known as "tweets". Registered users can post, like and retweet tweets, but unregistered users can only read them
The dataset belongs to Standford University, Since I cannot find the original source but research papers, I use the dataset that public on [Kaggle](https://www.kaggle.com/kazanova/sentiment140)

![Twitter](https://user-images.githubusercontent.com/63126292/106425944-51e07280-642a-11eb-888d-ad13efa902c3.jpg)

### 1. Outcome
* From the dataset, the amount of positive and negative tweets are almost the same
* Words in Positive Tweets 

![download (2)](https://user-images.githubusercontent.com/63126292/106426389-21e59f00-642b-11eb-9a18-75585935121a.png)

* Words in Negative Tweets 

![download (3)](https://user-images.githubusercontent.com/63126292/106426363-14c8b000-642b-11eb-905a-149dcbcf0940.png)

* The best vectorized method is TF-IDF, which has the lowest log loss and highest accuracy scores in all machine learning algorithms

[Bernoulli Naive Bayers] ![Screen Shot 2021-02-01 at 1 21 25 AM](https://user-images.githubusercontent.com/63126292/106426829-e7303680-642b-11eb-965d-34d12dd1f973.png)

[Logistic Regression] ![Screen Shot 2021-02-01 at 1 21 14 AM](https://user-images.githubusercontent.com/63126292/106426898-0333d800-642c-11eb-934e-7c19428f314a.png)

* The best machine learning algorithms is Logistic Regression and the second ond is Bernoulli Naive Bayers
Accuracy

![download (5)](https://user-images.githubusercontent.com/63126292/106426940-18a90200-642c-11eb-8022-0780b24b60c1.png)

Log Loss 

![download (4)](https://user-images.githubusercontent.com/63126292/106427012-3a09ee00-642c-11eb-9b91-1a2d6c3cde8c.png)

* For tunning the best parameter sets: 
  ** The best one for Logistic Regression are  ```lr__C: 1.0, lr__penalty: 'l2', lr__solver: 'newton-cg'```
  ** The best one for Bernoulli Naive Bayers are ```nb__alpha: 4, nb__binarize: 0```
  
### 2. Conclusion
* My research is a good practice for understand NLP and predict a model for sentiment analysis
* It is useful for News, Economics, Journalists, Data collectors for predict futre twitters, or other social media platform to predict sentiment.

---

## Unsupervised Learning Project: Airplane Crashed 
### Using Clustering methods( K-Means, Hierachical Clsutering, DBSCAN), Dimensionality Reduction (PCA, t-SNE, UMAP), and Text Processing with TF-IDF
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/apham15/Airplane_Crashed) 
[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1EcLvcgNmkrYZBPraaOk8EKw9X_KMONN2?usp=sharing)

I am a traveler, and airplanes are my primary means of transportation to discover the world. My favorite desination is Southest Asia, an ideal destination for snokerling, kayaking, and diving. As a result, when I found this dataset, it piqued my curiosity. Learning from historical airplane crash data could offer insight that would assist me in making future travel decisions. Also, as an Asian, the Chinese Zodiac is something very interesting to me. I did some fun stuff with that to learn more about the data.
![airports-world-network](https://user-images.githubusercontent.com/63126292/100527674-00053d00-319a-11eb-8c97-93025d1dc632.png)


### Outcome

#### 1. Airplane Crashing

* USA has the highest fatility cases in the world
* There are over 85% of airplane crashed which are from commercial flights
* Don't fly with Aeroflot Operator, there is 68% chance of death.
* Although it is a military airplane, don't fly in a Douglas DC-3 aircraft, you are 5 times more likely to die and it had the highest fatalities percentage.
* Don't take any flight that flies Tenerife - Las Palmas / Tenerife - Las Palmas route or Tokyo - Osaka.
* Avoid going to Sao Paulo, Brazil ; Moscow, Russia ; Rio de Janeiro, Brazil ; they had highest plane crash location.
* It is so much safer to take flight now-a-days as compared to 1970-80, 1972 was the worst years for the airline industry.
* Peole who are born in the year of the Ox have a higher risk of death, and people who are born in the year of the Horse have high chance to survive

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
* There is no absolute right answer for clustering. To bring the best results for a business solution, I would recommend data scientists consider the best model that fits the business need and apply their business acumen to suggest the best strategy for the whole team.
* My data analysis above is good for travelers around the world to gain insight about traviling with via airplane. Finally, it is a good suggestion for operators of aerospace companies to research the disavantages of the past to improve their future businesses.

---
## Unsupervised Learning Project: Fashion-MNIST 
### Using Clustering methods (K-Means, Hierarchical clustering, Gaussian Mixture Models, DBSCAN), apply Dimentional reduction to plot (PCA, t-SNE, LDA, UMAP), and build a 3d plot with UMAP
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
## Supervised Learning Project: Loan Prediction 
### Using Logistic, Decision Tree, Random Forest, K-Nearest Neighbor, Support Vector, Naive Bayes, and GBM
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/apham15/Loan_Prediction) 
[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1AinDO8z5Ud7OVTSx6KS3yGUauJqzzGhq?usp=sharing)
[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://github.com/apham15/Loan_Prediction/blob/main/Supervised_Learning_Project_Loan_Prediction_final_version.ipynb)

Lending Club is "the world's largest online marketplace connecting borrowers and investors." It is a peer-to-peer lending network that open sources some of its loan data to the community.

I used the loan.csv file that contains real-world, historical data on loans organized by Lending Club between 2007 and 2011.

![Screen Shot 2020-10-13 at 11 01 14 PM](https://user-images.githubusercontent.com/63126292/95942130-01280980-0da8-11eb-88a8-cfa750120469.png)

1. Business Understanding

You work for a consumer finance company which specializes in lending various types of loans to urban customers. When the company receives a loan application, the company has to make a decision for loan approval based on the applicant’s profile. Two types of risks are associated with the bank’s decision:

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
## Supervised Learning Project: Wine Quality Prediction 
### Using Logistic, K-Nearest Neighbor, Decission Tree, Random Forest, Support Vector
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/apham15/Loan_Prediction) 
[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1AinDO8z5Ud7OVTSx6KS3yGUauJqzzGhq?usp=sharing)
[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://github.com/apham15/Loan_Prediction/blob/main/Supervised_Learning_Project_Loan_Prediction_final_version.ipynb)

![wine-images-high-resolution-free-download](https://user-images.githubusercontent.com/63126292/97619738-96eda680-19ee-11eb-8b50-323cd9900bca.jpg)

1. Objectives: 
* The goal of this kernel is to find the best approach to identify the quality of wines. We will go through the basic EDA and visually identify the quality 
Moreover, I also applied multiple ML models to make the prediction respectively. Each of the models would have its own strength.
* There are two different notebooks. One is for red wine, and another one is for white wine
* My research is good beneficial for alcohol beverage companies and customers who love to learn more about wine.

2. Dataset
* The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine. For more details, consult the reference [Cortez et al., 2009]. Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).
* Acknowledgements
 This dataset is also available from the [UCI machine learning repository](https://archive.ics.uci.edu/ml/datasets/wine+quality) , I just shared it to Kaggle for convenience. (I am mistaken and the public license type disallowed me from doing so, I will take this down at first request. I am not the owner of this dataset.)
* P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

3. Outcome
* The best model for Red Wine is Logistic Regression (AUC 56.9%)
* The best model for White Wine is Random Forest Classifier (AUC 70.2%)

---
<center>© 2021 Anh (Andrew) Pham. Powered by Jekyll and the Minimal Theme.</center>
