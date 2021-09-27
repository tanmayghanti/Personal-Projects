import pandas as pd 
import pickle as pkl

A=pd.read_excel("Attrition Case Study.xlsx")


X = A[['RND','MKT']]  #change X and Y
Y = A[['PROFIT']]

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
model = lm.fit(X,Y)

pkl.dump(model,open("model.pkl","wb"))
model = pkl.load(open("model.pkl","rb"))
