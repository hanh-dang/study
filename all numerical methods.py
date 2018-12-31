#all methods on 1 plot
import matplotlib.pyplot as plt
import numpy as np
import math
from math import log

def feval(funcName, *args):
    return eval(funcName)(*args)

def mult(vector, scalar):
    newvector = [0]*len(vector)
    for i in range(len(vector)):
        newvector[i] = vector[i]*scalar
    return newvector

#ForwardEuler
def ForwardEuler(func, vinit, t_range, h):
    numOfODEs = len(vinit) 
    sub_intervals = int((t_range[-1] - t_range[0])/h)     
    t = t_range[0] 
    v = vinit 
    tsol = [t]
    vsol = [v[0]]
    
    for i in range(sub_intervals):
        vprime = feval(func, t, v)        
        for j in range(numOfODEs):
            v[j] = v[j] + h*vprime[j]             
        t += h 
        tsol.append(t)         
        for r in range(len(v)):
            vsol.append(v[r])             
    return [tsol, vsol]

#BackwardEuler
def BackwardEuler(func, vinit, t_range, h):
    numOfODEs = len(vinit)
    sub_intervals = int((t_range[-1] - t_range[0])/h)
    t = t_range[0]
    v = vinit
    tsol = [t]
    vsol = [v[0]]

    for i in range(sub_intervals):
        vprime = feval(func, t+h, v)
        vp = mult(vprime, (1/(1+h)))
        for j in range(numOfODEs):
            v[j] = v[j] + h*vp[j]
        t += h
        tsol.append(t)
        for r in range(len(v)):
            vsol.append(v[r])  
    return [tsol, vsol]

#2ND order Runge Kutta
def RK2(func, vinit, t_range, h):
    m = len(vinit)
    n = int((t_range[-1] - t_range[0])/h)    
    t = t_range[0]
    v = vinit
    tsol = np.empty(0)
    tsol = np.append(tsol, t)
    vsol = np.empty(0)
    vsol = np.append(vsol, v)
    
    for i in range(n):
        k1 = feval(func, t, v)
        vpredictor = v + k1 * (h/2)
        k2 = feval(func, t+h/2, vpredictor)
        for j in range(m):
            v[j] = v[j] + h*k2[j]
        t = t + h
        tsol = np.append(tsol, t)
        for r in range(len(v)):
            vsol = np.append(vsol, v[r])  
    return [tsol, vsol]

#3rd order Runge Kutta
def RK3(func, vinit, t_range, h):
    m = len(vinit)
    n = int((t_range[-1] - t_range[0])/h)    
    t = t_range[0]
    v = vinit    
    tsol = np.empty(0)
    tsol = np.append(tsol, t)
    vsol = np.empty(0)
    vsol = np.append(vsol, v)
    
    for i in range(n):
        k1 = feval(func, t, v)
        vp1 = v + k1 * (h/2)
        k2 = feval(func, t+h/2, vp1)
        vp2 = v - (k1 * h) + (k2 * 2*h)
        k3 = feval(func, t+h, vp2)
        for j in range(m):
            v[j] = v[j] + (h/6)*(k1[j] + 4*k2[j] + k3[j])
        t = t + h
        tsol = np.append(tsol, t)
        for r in range(len(v)):
            vsol = np.append(vsol, v[r])  
    return [tsol, vsol]

#4th order Runge Kutta
def RK4(func, vinit, t_range, h1):
    m = len(vinit)
    h=float(h1)
    n = int((t_range[-1] - t_range[0])/h)
    t = t_range[0]
    v = vinit    
    tsol = np.empty(0)
    tsol = np.append(tsol, t)
    vsol = np.empty(0)
    vsol = np.append(vsol, v)
    
    for i in range(n):
        k1 = feval(func, t, v)
        vp2 = v + k1*(h/2)
        k2 = feval(func, t+h/2, vp2)
        vp3 = v + k2*(h/2)
        k3 = feval(func, t+h/2, vp3)
        vp4 = v + k3*h
        k4 = feval(func, t+h, vp4)
        for j in range(m):
            v[j] = v[j] + (h/6)*(k1[j] + 2*k2[j] + 2*k3[j] + k4[j])
        t = t + h
        tsol = np.append(tsol, t)
        for r in range(len(v)):
            vsol = np.append(vsol, v[r])  
    return [tsol, vsol]

#skydiver problem
def skydiver(t, v):
    dv = np.zeros((len(v)))
    dv[0] = 0.4875/50*(v[0]**2)-9.81
    return dv

#input parameters
      
#BackwardEuler
h = 4
t = [0, 100]
vinit = [0.0]
[tB, yB] = BackwardEuler('skydiver', vinit, t, h)
#convergence order 
a=float(1/log(2)*log(abs((yB[4]-yB[2])/(yB[2]-yB[1]))))
print('The order of convergence of Backward Euler is',float('%.5f' % a))

#ForwardEuler
h =4 
t = [0, 100]
vinit = [0.0]
[tF, yF] = ForwardEuler('skydiver', vinit, t, h)
#convergence order 
a=float(1/log(2)*log(abs((yF[4]-yF[2])/(yF[2]-yF[1]))))
print('The order of convergence of Forward Euler is',float('%.5f' % a))
#RK2
h = 4
t = [0, 100]
vinit = [0.0]
[tRK2, yRK2] = RK2('skydiver', vinit, t, h)
#convergence order 
a=float(1/log(2)*log(abs((yRK2[4]-yRK2[2])/(yRK2[2]-yRK2[1]))))
print('The order of convergence of 2nd Order Runge-Kutta is',float('%.5f' % a))

#RK3
h =4 
t = [0, 100]
vinit = [0.0]
[tRK3, yRK3] = RK3('skydiver', vinit, t, h)
#convergence order 
a=float(1/log(2)*log(abs((yRK3[4]-yRK3[2])/(yRK3[2]-yRK3[1]))))
print('The order of convergence of 3rd Order Runge-Kutta is',float('%.5f' % a))

#RK4
h = 4
t = [0, 100]
vinit = [0.0]
[tRK4, yRK4] = RK4('skydiver', vinit, t, h)
#convergence order 
a=float(1/log(2)*log(abs((yRK3[4]-yRK3[2])/(yRK3[2]-yRK3[1]))))
print('The order of convergence of 4th Order Runge-Kutta is',float('%.5f' % a))

#plot all methods
plt.xlim(t[0], t[1])
plt.plot(tB, yB, 'r')
plt.plot(tF, yF, 'b')
plt.plot(tRK2, yRK2, 'm')
plt.plot(tRK3, yRK3, 'y')
plt.plot(tRK4, yRK4, 'g')

plt.legend(["Backward Euler",
            "Forward Euler", "2nd Order Runge-Kutta", 
            "3rd Order Runge-Kutta", "4th Order Runge-Kutta"], loc=0)
plt.xlabel('time (s)', fontsize=14)
plt.ylabel('velocity (m/s)', fontsize=14)
plt.title("Velocity of skydiver as a function of time", fontsize=14)


 



    

