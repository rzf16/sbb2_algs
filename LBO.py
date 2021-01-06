import numpy as np

def single_optimize(size,value,eta,zeta):
    a = 0.1083869
    b = 0.99837249
    c = 0.02535

    if value == 0.0:
        return 0
    action = -a/(np.log(2)*(zeta/eta)*value) + 1/b
    if action <=0 :
        action = 0
    elif action >= 1:
        action = 1
    return action
