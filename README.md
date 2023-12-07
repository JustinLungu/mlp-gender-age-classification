# Evaluation 🛠️

predicting/storing the predicitons from the test data in the same dataset (.csv file)

- adjusted R^2 --> use y_age_pred (compare to the average age)
- f1 score --> use y_gen_pred (should just be better than random)

- evaluation - predict on test data and perform adjusted R$^$2 and F1 score, confusion matrices, and threshold stuff ( from sklearn.metrics import classification_report classification\_report(y\_test, y\_pred))

y_age_pred (for adjusted r squared, as well as true/false with confidence intervals)
y_gen_pred (for F1, etc.)
compare to true labels

cross validation plots 
