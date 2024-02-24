#!/usr/bin/env python
# coding: utf-8

# In[173]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sc
import time
from scipy.integrate import solve_ivp, solve_bvp


# # problem 1
# #### Use the 4th-order Runge-Kutta method to solve the ODE. Provide plots of 𝑦 and 𝑑𝑦/𝑑𝑡 as a function of 𝑡 for the first 10 seconds. 

# In[174]:


def rk(y,t,dt,f):
    k1 = f(y,t)*dt
    k2 = f(y+k1/2,t+dt/2)*dt
    k3 = f(y+k2/2,t+dt/2)*dt
    k4 = f(y+k3,t+dt)*dt
    return y + (k1+2*k2+2*k3+k4)/6

def ode(y,t):
    return np.array([y[1],(-2*(9.8))*y[0]/L])

yi=np.array([.05,0])
ti=0
L=.2
dt=.01

times=[ti]
heights=[yi]

while ti <= 10:
    yi=rk(yi,ti,dt,ode)
    ti+=dt
    times.append(ti)
    heights.append(yi)
    
hs=np.array(heights)

fig=plt.figure()
gs = fig.add_gridspec(2,1)
y = fig.add_subplot(gs[0, 0])
dydt = fig.add_subplot(gs[1, 0])
fig.tight_layout()

y.plot(times,hs[:,0],label='y')
y.set_xlabel('time (s)')
y.set_ylabel('y (height in m)')
y.set_ylim(-.6,.6)

dydt.plot(times,hs[:,1])
dydt.set_xlabel('time (s)')
dydt.set_ylabel('dy/dt (m/s)')
dydt.set_ylim(-.6,.6)

plt.tight_layout()


# # problem 2
# #### (a) What are the initial condition(s) that fully define the system?
# 
# the inital conditions that fully define the system are the cat's speed, dog's speed, dog's intial position, and tree's position 
# 
# #### (b) Integrate this differential equation to prepare a plot of the dog’s position y as a function of its x position. Assume that the cat’s speed is 𝑎 = 20 𝑓𝑡/𝑠, the dog’s speed is 𝑏 = 30 𝑓𝑡/𝑠, and 𝑐 = 150 𝑓𝑡. Feel free to use any of the tools at your disposal to integrate

# In[214]:


a=20
b=30
c=150

def f(t,v):
    return[-b*((1+(v[2]**2))**-.5),v[2],(a/(b*v[0]))*((1+(v[2]**2))**.5)]
soln=solve_ivp(f,(0,5),[c,0,0],method='LSODA',max_step=.01)
plt.plot(soln.y[0],soln.y[1])
plt.xlabel('x (ft)')
plt.ylabel('y (ft)')
plt.show()


# #### (c) What is the maximum distance that the cat can start from the tree to ensure that it can reach safety?

# In[217]:


soln.t[-1] * a


# #### (d) Plot the cat’s maximum safe distance from the tree (on the y-axis) as a function of the dog’s speed (on the x-axis). Assume again that the cat’s speed is 𝑎 = 20 𝑓𝑡/𝑠 and 𝑐 = 150 𝑓𝑡

# In[236]:


v=np.linspace(20,1000)
td=[]
for i in v:
    soln=solve_ivp(f,(0,5),[c,0,0],method='LSODA',max_step=.01)
    td.append(soln.t[-1]*a)
plt.plot(v,td)
plt.xlabel('dog speed (ft/s)')
plt.ylabel('max safe dist from tree (ft)')
plt.show()


# #### (e)  Clearly the dog could be more successful in catching the cat if it always ran toward a position 10 feet in front of the cat rather than directly at it. Where will the dog catch the cat under these conditions? 

# In[ ]:





# #### (f) The dog could do even better if it anticipated that the cat was headed to the tree and therefore chose itself to run straight toward the tree rather than the cat. This type of pursuit path is the approach taken frequently by humans and relies upon our intelligence and predictive/integrative prowess. Doing some research in the literature and thinking about natural systems, do predator/prey systems in nature exhibit pursuit paths described by these limiting behaviors? 

# yes, some predators do exhibit those pursuit paths.

# # problem 3
# $Adapted from https://www.mathworks.com/company/newsletters/articles/stiff-differential-equations.html$
# 
# Stiffness is a subtle, complicated, but important concept in the numerical solution of ordinary differential equations.
# 
# An ordinary differential equation problem is stiff if the solution being sought is varying slowly, but there are nearby solutions that vary rapidly, so the numerical method must take small steps to obtain satisfactory results.
# 
# A model of flame propagation from O'Malley (1991) and Shampine et al. (2003) provides an example. When you light a match, the ball of flame grows rapidly until it reaches a critical size. Then it remains at that size because the amount of oxygen being consumed by the combustion in the interior of the ball balances the amount available through the surface. The simple model is
# $$\frac{dy}{dt}=y^2(1-y)$$
# $$y(0)=\delta$$
# $$0<t<\frac{2}{\delta}$$
# 
# The scalar variable $y(t)$  represents the radius of the ball. The $y^2$ and $y^3$ terms come from the surface area and the volume. The critical parameter is the initial radius, $\delta$, which is "small." We seek the solution over a length of time that is inversely proportional to $\delta$.

# #### (a) Try $\delta = 0.5$. Choose appropriate method and step size. Plot $y(t)$ and $\frac{dy}{dt}$ in the same figure.

# In[237]:


def f(t, y):
    return y**2-y**3

delta = 0.5
soln = solve_ivp(f,[0,2/delta],[delta],method='Radau')

plt.plot(soln.t, soln.y.flatten(), label = 'y(t)')
plt.plot(soln.t, f(soln.t, soln.y.flatten()), label = 'dy/dt')
plt.legend()
plt.show()


# #### (b) Try $\delta = 2$. Choose appropriate method and step size. Plot $y(t)$ and $\frac{dy}{dt}$ in the same figure.

# In[238]:


def f(t, y):
    return y**2-y**3

delta = 2
soln = solve_ivp(f,[0,2/delta],[delta],method='Radau')

plt.plot(soln.t, soln.y.flatten(), label = 'y(t)')
plt.plot(soln.t, f(soln.t, soln.y.flatten()), label = 'dy/dt')
plt.legend()
plt.show()


# #### (c) Now try $\delta = 0.001$. Choose an appropriate method and step size. Plot $y(t)$ and $\frac{dy}{dt}$ in the same figure. Compared with your result from part (a), what is the difference?

# In[374]:


def f(t, y):
    return y**2-y**3

delta = .001
soln = solve_ivp(f,[0,2/delta],[delta],method='Radau')

plt.plot(soln.t, soln.y.flatten(), label = 'y(t)')
plt.plot(soln.t, f(soln.t, soln.y.flatten()), label = 'dy/dt')
plt.legend()
plt.show()

print('compared to a, y(t) has a large spike as well as dy/dt. not very curve-like')


# #### (d) Now create a graph that provides a zoomed in view to the solution of y versus t for y between 0.998 and 1.002.  To your plot, add the solutions achieved using the RK23, RK45, DOP853, Radau, LSODA, and BDF methods from the scipy.integrate library.  
# 
# What do you observe?  Which methods work best?  Why?

# In[281]:


times = []
delta=.001
def f(t, y):
    return y**2-y**3
t0=0
RK23 = solve_ivp(f,[0,2/delta],[delta],method = 'RK23')
times.append(time.time()-t0)
t0=0
RK24 = solve_ivp(f,[0,2/delta],[delta],method = 'RK45')
times.append(time.time()-t0)
t0=0
DOP853 = solve_ivp(f,[0,2/delta],[delta],method = 'DOP853')
times.append(time.time()-t0)
t0=0
Radau = solve_ivp(f,[0,2/delta],[delta],method = 'Radau')
times.append(time.time()-t0)
t0=0
LSODA = solve_ivp(f,[0,2/delta],[delta],method = 'LSODA')
times.append(time.time()-t0)
t0=0
BDF = solve_ivp(f,[0,2/delta],[delta],method = 'BDF')
times.append(time.time()-t0)

plt.ylim(0.988,1.002)
plt.plot(RK23.t, RK23.y.flatten(),label='RK23')
plt.plot(RK24.t, RK24.y.flatten(),label='RK45')
plt.plot(DOP853.t, DOP853.y.flatten(),label='DOP853')
plt.plot(Radau.t, Radau.y.flatten(),label='Radau')
plt.plot(LSODA.t, LSODA.y.flatten(),label='LSODA')
plt.plot(BDF.t, BDF.y.flatten(),label='BDF')
plt.legend()
plt.ylabel('y')
plt.xlabel('t')
plt.xlim(995,1195)
plt.show()


# LSODA, BDF, and Radau seem the smoothest

# #### (e) For each of the methods that you compared in part (d), determine (i) the number of steps taken and (ii) the CPU time for the integration.  Is the computational efficiency of the library methods correlated with their ability to accurately solve stiff ODEs?

# In[376]:


stepRK23 = [len(RK23.t)]
stepRK24 = [len(RK24.t)]
stepDOP853 = [len(DOP853.t)]
stepRadau = [len(Radau.t)]
stepLSODA = [len(LSODA.t)]
stepBDF = [len(BDF.t)]

timeRK23 = [times[0]]
timeRK24 = [times[1]]
timeDOP853 = [times[2]]
timeRadau = [times[3]]
timeLSODA = [times[4]]
timeBDF = [times[5]]

plt.plot(stepRK23,timeRK23,'s',label='RK23')
plt.plot(stepRK24,timeRK24,'s',label='RK45')
plt.plot(stepDOP853,timeDOP853,'s',label='DOP853')
plt.plot(stepRadau,timeRadau,'s',label='Radau')
plt.plot(stepLSODA,timeLSODA,'s',label='LSODA')
plt.plot(stepBDF,timeBDF,'s',label='BDF')
plt.xlabel('step size')
plt.ylabel('CPU time')
plt.legend()
plt.show()


# larger step sizes tend to take longer and are less accuate, seen with RK23&RK24 and part c. Raudu seems to be the most accurate, with the smallest step size while not compromising as much on time, and BDF and LSODA both have small step sizes, which can explain the smoothness observed in c.

# ## Problem 4 - Application of implicit methods for solving IVPs
# #### The implicit Euler method is unconditionally stable. Let's explore this further under different situations.

# (a) Given the non-linear ODE $$\frac{dy}{dt}=30(cos(t)-y)+3sin(t)$$
# 
# If y(0) = 1, use the implicit Euler method to obtain a solution from t = 0 to 4 using a step size of 0.4.
# 

# In[192]:


def f(t,yi,h):
    return (yi+30*h*np.cos(t)+3*h*np.sin(t))/(1+30*h)
def e(f,xi,xf,yi,h):
    a=[(xi,yi)]
    x=a[-1][0]
    y=a[-1][1]
    while x+h <= xf:
        yi = f(x+h,y,h)
        a.append((x+h, yi))
        x=a[-1][0]
        y=a[-1][1]
    return a
x=[]
y=[]
for i in e(f,0,4,1,.4):
    x.append(i[0])
    y.append(i[1])
plt.plot(x,y)
plt.xlabel('t')
plt.ylabel('y')
plt.show()

def eexplicit(f,xi,xf,yi,h):
    a=[(xi,yi)]
    x=a[-1][0]
    y=a[-1][1]
    x0=[xi]
    y0=[yi]
    while x+h <= xf:
        yi=f(x,y,h)*h+y
        a.append((x+h,yi))
        x=a[-1][0]
        y=a[-1][1]
        x0.append(x+h)
        y0.append(yi)
    return x0,y0
plt.plot(eexplicit(f,0,4,1,.4)[0],eexplicit(f,0,4,1,.4)[1])
plt.title('part d')
plt.xlabel('t values')
plt.ylabel('y values')
plt.show()


# (b) Given the non-linear ODE $$\frac{dy}{dt}=30(cos(t)-y^2)+3sin(t)$$
# 
# If y(0) = 1, use the implicit Euler method to obtain a solution from t = 0 to 4 using a step size of 0.4.
# 

# In[193]:


def f(t,yi,h):
    return (yi+30*h*np.cos(t)+3*h*np.sin(t))/(1-30*yi*h)
def e(f,xi,xf,yi,h):
    a=[(xi,yi)]
    x=a[-1][0]
    y=a[-1][1]
    while x+h <= xf:
        yi = f(x+h,y,h)
        a.append((x+h, yi))
        x=a[-1][0]
        y=a[-1][1]
    return a
x=[]
y=[]
for i in e(f,0,4,1,.4):
    x.append(i[0])
    y.append(i[1])
plt.plot(x,y)
plt.xlabel('t')
plt.ylabel('y')
plt.show()

def eexplicit(f,xi,xf,yi,h):
    a=[(xi,yi)]
    x=a[-1][0]
    y=a[-1][1]
    x0=[xi]
    y0=[yi]
    while x+h <= xf:
        yi=f(x,y,h)*h+y
        a.append((x+h,yi))
        x=a[-1][0]
        y=a[-1][1]
        x0.append(x+h)
        y0.append(yi)
    return x0,y0
plt.plot(eexplicit(f,0,4,1,.4)[0],eexplicit(f,0,4,1,.4)[1])
plt.title('part d')
plt.xlabel('t values')
plt.ylabel('y values')
plt.show()


# (c) Given a higher-order of ODE $$\frac{d^2y}{dx^2}=-1001\frac{dy}{dx}-1000y$$
# 
# If y(0) = 1 and y'(0) = 0, use the implicit Euler approach with h = 0.5 to solve for x = 0 to 5.

# In[202]:


def f(t,yi,dyi,h):
    return (dyi-1000*h*yi)/(1+1001*h+1000*(h**2)) 
def e(f,xi,xf,yi,dyi,h):
    a=[(xi,yi,dyi)]
    x=a[-1][0]
    y=a[-1][1]
    dy=a[-1][2]
    while x+h <= xf:
        dy=f(x,y,dy,h)
        y= y+h*dy
        a.append((x+h,y,dy))
        x=a[-1][0]
        y=a[-1][1]
        dy=a[-1][2]
    return a
x=[]
y=[]
for i in e(f,0,5,1,0,.5):
    x.append(i[0])
    y.append(i[1])
plt.plot(x,y)
plt.xlabel('t')
plt.ylabel('y')
plt.show()

def eexplicit(f,xi,xf,yi,dyi,h):
    a=[(xi,yi,dyi)]
    x=a[-1][0]
    y=a[-1][1]
    dy=a[-1][2]
    while x+h <= xf:
        dy0=dy=f(x,y,dy,h)*h
        y=y+dy*h
        a.append((x+h,y,dy))
        x=a[-1][0]
        y=a[-1][1]
        dy=a[-1][2]
    return a
x=[]
y=[]
for i in eexplicit(f,0,5,1,0,.5):
    x.append(i[0])
    y.append(i[1])
plt.plot(x,y)
plt.title('part d')
plt.xlabel('t')
plt.ylabel('y')
plt.show()


# (d) For each of the prior examples, solve the ODEs using the explicit Euler method with the same step size.  Are the solutions stable?  What is the critical step size at which the explicit Euler method transitions from being stable to unstable?   

# it seems that the step size of .5 is stable, where as the h=.4 is not. part c is stable, but not a and b with the explicit euler method. 

# # problem 5
# #### (a) Recast this differential equation as a system of two first-order differential equations

# d^2(T)/dx^2 = 10^-7(T+273)^4-4(150-T)
# 
# set dT/dx=k so...
# 
# dk/dx = d^2(T)/dx^2 = 10^-7(T+273)^4-4(150-T)
# 
# dT/dx=k and dk/dx = 10^-7(T+273)^4-4(150-T) are the two first-order differential equations

# #### (b) Implement the shooting method to solve the equation given the boundary conditions of 𝑇(0) = 200 and 𝑇(0.5) = 100. Provide a plot of your solution and discuss whether it satisfies the boundary conditions.

# In[373]:


def dTdx(x,T,k):
    return k
def dkdx(x,T,k):  
    return (10**-7)*(T+273)**4+4*(T-150)

def e(xi,xf,Ti,ki,h):
    a=[(xi,Ti,ki)]
    x=a[-1][0]
    T=a[-1][1]
    k=a[-1][2]
    X=[]
    t=[]
    K=[]
    while x+h <= xf:
        T2 = T+h*dTdx(x,T,k)
        k2 = k+h*dkdx(x,T,k) 
        a.append((x+h,T2,k2))
        x=a[-1][0]
        T=a[-1][1]
        k=a[-1][2]  
    for i in a:
        x,T,k = i
        X.append(x)
        t.append(T)
        K.append(k)
    return X,t,K

def shooter(xi,xf,Ti,Tf,h):  
    guess=[-2,2]
    guess1=guess[0]
    guess2=guess[1]
    x1,T1,k1 = e(xi,xf,Ti,guess1,h)
    x2,T2,k2 = e(xi,xf,Ti,guess2,h)
    g1=k1[-1]
    g2=k2[-1]
    error=abs((Tf-g2)/Tf)
    while error > .001:
        guessi = guess2 - (g2-Tf)*((guess2-guess1)/(g2-g1))
        guess.append(guessi)
        guess1 = guess[-2]
        guess2 = guess[-1]
        x1,T1,k1 = e(xi,xf,Ti,guess1,h)
        x2,T2,k2 = e(xi,xf,Ti,guess2,h)
        g1 = T1[-1]
        g2 = T2[-1]
        error = abs((Tf-g2)/Tf)
    return x2,T2,k2
x,T,k = shooter(0,.5,200,100,.05)
plt.plot(x,T)
plt.axvline(0,c='grey',linewidth=.5)
plt.axvline(.5,c='grey',linewidth=.5)
plt.axhline(200,c='grey',linewidth=.5)
plt.axhline(100,c='grey',linewidth=.5)
plt.show()


# the  solution satisifes the bcs.

# # problem 6
# #### (a) Solve using a finite difference method based on the central difference approximation for both the first and second derivatives (both have a truncation error of 𝑂(ℎ^2)). Adjust the number of sub-intervals (or your mesh-size) into which you subdivide the domain to balance solution accuracy and computational efficiency. Show your work in setting up the problem, and then use python to solve for 𝑦(𝑥) for points between 0 ≤ 𝑥 ≤ 1 .

# In[128]:


n=200
h=.00000000001
x=np.linspace(0,1,n+1)
y=np.zeros(n+1)
y[0]=0
y[1]=.2
for i in range(1,n):
    y[i+1] = (6*h**2 + 2*h*y[i] - 2*y[i-1])/(2*h+1)
plt.plot(x,y)
plt.show()


# #### (b) Solve using the scipy.integrate.solve_bvp method in the python libraries. Plot 𝑦 versus 𝑥 and 𝑑𝑦/𝑑𝑥 versus 𝑥.

# In[144]:


def f(x, y):
    return np.vstack([y[1],6*x-y[0]-x*y[1]])
def bc(y1, y2):
    return np.array([y1[1]+1,y2[0]-1])

n=100
x=np.linspace(0,1,n)
y1=np.zeros((2,n))

y=solve_bvp(f,bc,x,y1)
plt.plot(x, y.sol(x)[0], label='y')
plt.plot(x, y.sol(x)[1], label='dy/dx')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

