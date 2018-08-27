from sklearn.metrics import f1_score
import numpy as np
#y_true = [0, 1, 2, 0, 1, 2]
#y_pred = [0, 2, 1, 0, 0, 1]

y_true=np.ones((20,5))
y_true[0]=[0,1,0,0,0]
y_pred=np.ones((20,5))
print(y_pred.dtype)
#a=f1_score(y_true, y_pred, average='macro')  
a=f1_score(y_true, y_pred)  

print(a)
