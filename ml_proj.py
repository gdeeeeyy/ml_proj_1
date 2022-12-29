import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import math
data=pd.read_csv(r"student-mat.csv", sep=";")
data=data[["X5 latitude", "Y house price of unit area"]]
predict="Y house price of unit area"
X_train=np.array(data.drop([predict], 1))
y_train=np.array(data[predict])
m=X_train.shape[0]
def model(xi, w,b):
    m=xi.shape[0]
    f_wb=np.zeros(m)
    for i in range(m):
        f_wb=w*xi[i]+b
    return f_wb
def compute_cost(xi,yi, w, b):
    m=xi.shape[0]
    cost_sum=0
    f_wb=model(X_train, w,b)
    for i in range(m):
        cost=(f_wb-yi[i])**2
        cost_sum+=cost
    total_cost=(1/(2*m))*cost_sum
    return total_cost
def compute_gradient(x, y, w, b): 
    m = x.shape[0]    
    dj_dw = 0
    dj_db = 0
    for i in range(m):  
        f_wb = w * x[i] + b 
        dj_dw_i = (f_wb - y[i]) * x[i] 
        dj_db_i = f_wb - y[i] 
        dj_db += dj_db_i
        dj_dw += dj_dw_i 
    dj_dw = dj_dw / m 
    dj_db = dj_db / m 
    return dj_dw, dj_db
def grad_desc(x,y,w_in, b_in, alpha,  num_iter, cost_fn, grad_fn):
    W=copy.deepcopy(w_in)
    J_hist=[]#hist of cost_fn
    w_hist=[]#hist of w,b
    b=b_in
    w=copy.deepcopy(w_in)
    for i in range(num_iter):
        dj_dw, dj_db=grad_fn(x,y,w,b)
        b=b-alpha*dj_db
        w=w-alpha*dj_dw
        if i<10000:
            J_hist.append(cost_fn(x, y, w , b))   
            w_hist.append(w)
        return w, b, J_hist, w_hist
w_init=0
b_init=0
iter=1000000
tmp_alpha = 1.0e-2
W_final, b_final, J_hist, p_hist=grad_desc(X_train, y_train,w_init, b_init, tmp_alpha, iter, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({W_final},{b_final})")
predict1=24.98746 * W_final + b_final
print(predict1)
m = X_train.shape[0]
predicted = np.zeros(m)
for i in range(m):
    predicted[i] = W_final * X_train[i] + b_final
# Plot the linear fit
plt.plot(X_train, predicted, c = "b")

plt.scatter(X_train, y_train, marker='x', c='r') 

# Set the title
plt.title("latitude vs. house price per unit area")
# Set the y-axis label
plt.ylabel('house price per unit area')
# Set the x-axis label
plt.xlabel('latitude')
plt.show()
