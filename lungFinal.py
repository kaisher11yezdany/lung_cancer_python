 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
df = pd.read_csv('lungcancer.csv')
print(df.head())


#X = ldata[['Air Pollution', 'Alcohol use', 'Dust Allergy',
#       'OccuPational Hazards', 'Genetic Risk', 'chronic Lung Disease',
#        'Balanced Diet', 'Obesity', 'Smoking', 'Passive Smoker', 'Chest Pain',
#        'Coughing of Blood', 'Fatigue', 'Weight Loss', 'Shortness of Breath',
#        'Wheezing', 'Swallowing Difficulty', 'Clubbing of Finger Nails',
 #       'Frequent Cold', 'Dry Cough', 'Snoring']]

#X_train = df.loc[:,'Air Pollution':'Snoring']
#Y_train = df.loc[:,'Level']

X_train = df[['Alcohol use','chronic Lung Disease',]]

Y_train = df[['Level']]



tree = DecisionTreeClassifier(max_leaf_nodes=6, random_state=0)

tree.fit(X_train, Y_train)

#prediction = tree.predict([[4,5,6,5,5,4,6,7,2,3,4,8,8,7,9,2,1,4,6,7,2]])
prediction = tree.predict([[5,6]])



#2 4 5 4 3 2 2 4 3 2 2 4 3 4 2 2 3 1 2 3 4

#4,5,6,5,5,4,6,7,2,3,4,8,8,7,9,2,1,4,6,7,2
#air pollution+alcohol use +