import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import metrics
 
 
 
ldata = pd.read_csv('lungcancer.csv')
ldata.head(5)
ldata.tail()
ldata.info()
ldata.columns
ldata.shape
ldata.describe()
ldata.isna().sum()

ldata.Level.value_counts()
sns.countplot(x = 'Level', data = ldata, palette = 'hls')
plt.show()
le = preprocessing.LabelEncoder()






  
# Encode labels in column 'species'. 
ldata['Level']= le.fit_transform(ldata['Level']) 
  
ldata['Level'].unique() 

list(le.classes_)


X = ldata[['Air Pollution', 'Alcohol use', 'Dust Allergy',
       'OccuPational Hazards', 'Genetic Risk', 'chronic Lung Disease',
       'Balanced Diet', 'Obesity', 'Smoking', 'Passive Smoker', 'Chest Pain',
       'Coughing of Blood', 'Fatigue', 'Weight Loss', 'Shortness of Breath',
       'Wheezing', 'Swallowing Difficulty', 'Clubbing of Finger Nails',
       'Frequent Cold', 'Dry Cough', 'Snoring']]

Y = ldata[['Level']]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)

logreg = LogisticRegression()
logreg.fit(X_train,Y_train)


Y_pred=logreg.predict(X_test)

cnf_matrix = metrics.confusion_matrix(Y_test, Y_pred)
cnf_matrix

fig, ax = plt.subplots()

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="viridis" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))

sns.countplot(x="Level",data=ldata)





