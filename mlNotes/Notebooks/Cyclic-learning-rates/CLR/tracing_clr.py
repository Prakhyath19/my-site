import math
import matplotlib.pyplot as plt
import numpy as np

#variables
base_lr = 0.001
max_lr = 0.006
step_size = 2000
lr_list = []

def traingular_clr(iteration, step_size, base_lr, max_lr):
    cycle = math.floor(1+iteration/(2*step_size))
    x = abs(iteration/step_size - (2*cycle)+1)
    lr = base_lr + ((max_lr-base_lr) * max(0,(1-x)))
    return lr


for iteration in range(10000):
    lr = traingular_clr(iteration, step_size, base_lr,max_lr)
    lr_list.append(lr)
    print(f"Iteration: {iteration}, Learning Rate: {lr: .6f}")
    
# # itr_list = np.linspace(0,1000,retstep=1,num=1000,dtype="int")
itr_list = []
for i in range(10000): itr_list.append(i)
# # print(itr_list)
# # print(lr_list)
plt.grid(True)
plt.xlabel("Iterations")
plt.ylabel("Learning Rate")
plt.title("CLR policy: Triangular")
plt.plot(itr_list, lr_list,
            #  xlabel = "iteration",
            #  ylabel = "lr",
            "b-"
            )
plt.show()
