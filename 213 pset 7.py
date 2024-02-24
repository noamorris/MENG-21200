#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sc


# # problem 1
# Use the golden-section search method to determine the length of the shortest ladder that reaches from the ground over the fence to touch the building’s wall (see figure at right). 
# #### (a) Write your own code for the golden-section search method. Demonstrate your code to determine the shortest ladder possible for the case where ℎ = 𝑑 = 6 𝑚.

# In[2]:


tol=0.00001

def f(x,d=6,h=6):
    return (x+d)*np.sqrt(x**2 + h**2)/x
gr = (np.sqrt(5)-1)

def gs(low,up):
    while abs(up-low) > tol:
        x=(up-low)*0.5*gr
        low1=low+x
        up1=up-x
        low2=f(low1)
        up2=f(up1)
        if low2 < up2:
            low=up1
        else:
            up=low1
    return (up+low)/2

print(f'shortest ladder possible for h=d=6m: {gs(1,100)}')


# #### (b)  Demonstrate the application of the scipy.optimize.golden function in python to find the minimum.

# In[3]:


from scipy.optimize import golden

print(f'shortest ladder possible for h=d=6m: {golden(f, brack=(0.0000000001,10))}')


# #### (c) Note that the golden function is a ‘legacy’ function still available in Python, but is no longer recommended for use. Instead, please demonstrate the application of the scipy.optimize.minimize_scalar function.

# In[4]:


from scipy.optimize import minimize_scalar as ms

print(f'shortest ladder possible for h=d=6m: {ms(f,bracket=(0.0000000001,10),method="Golden").x}')


# #### (d) Prepare a plot of the length of the shortest ladder versus 𝑑 for the case where ℎ = 6 𝑚. Consider distances between the fence and wall up to 20 𝑚 (i.e., 0 ≤ 𝑑 ≤ 20 𝑚). 

# In[5]:


lens=[]
ds=[]
for d in range(21):
    ds.append(d)
    length=gs(1,100)
    lens.append(f(length,d))
plt.plot(ds,lens)
plt.title('length of shortest ladder vs. d when h=6')
plt.xlabel("d (m)")
plt.ylabel('shorest ladder (m)')
plt.show()


# # problem 2
# A finite-element model of a cantilever beam subject to loading and moments is given by optimizing $𝑓(𝑥,𝑦)=5𝑥^2−5𝑥𝑦+2.5𝑦^2−𝑥− 1.5𝑦$ where 𝑥 = end displacement and 𝑦 = end moment. Find the values of x and y that minimize 𝑓(𝑥, 𝑦).
# 
# #### (a) Starting with an initial guess of x = y = 1, compute by hand (or in a markdown box of Jupyter notebooks) the first steps taken using 1) the steepest descent optimization method and 2) the optimal steepest descent optimization method.
# 
# 1) $x_0=y_0=1: \frac{df}{dx}=10x-5y-1, \frac{df}{dy}=5y-5x-1.5$
# $x=x_0-\frac{df}{dx}h=1-(10-5-1)h=1-4h$
# $y=y_0-\frac{df}{dy}h=1-(5-5-1.5)h=1+1.5h$
# 
# if h = 0.1, the first step would be (0.6, 1.15)
# 
# 2) f(1-4h, 1+1.5h)=g(h),
# $g(h)=5(1-4h)^2-5(1-4h)(1+1.5h)+2.5(1+1.5h)^2-(1-4h)-1.5(1+1.5h)$
# 
# $g(h)'=231.25h - 18.25$
# 
# if g(h)'=0, h would need to be 0.0789, making the first step (0.684, 1.119)

# #### (b) Now implement in python your own versions of the 1) steepest descent and 2) optimal steepest descent optimization methods. Apply to find the minimize of 𝑓(𝑥, 𝑦). 

# In[6]:


def f(x,y):
    return 5*x**2 - 5*x*y + 2.5*y**2 - x - 1.5*y
def dfdx(x,y):
    return 10*x - 5*y - 1
def dfdy(x,y):
    return 5*y - 5*x - 1.5
def dgdh(h):
    return 231.25*h - 18.25

tol=0.00001
#1
def sd(x0,y0,h=0.1):
    itr=0
    x=x0
    y=y0
    xs=[x0]
    ys=[y0]
    
    while itr < 100000:
        itr+=1
        x=x-h*dfdx(x,y)
        y=y-h*dfdy(x,y)
        xs.append(x)
        ys.append(y)
        if abs(dfdx(x, y)) < tol and abs(dfdy(x, y)) < tol:
            itr+=10000
    return xs,ys,itr-100000

#2
from scipy.optimize import brentq
h=(brentq(dgdh,0,1))
def osd(x0,y0):
    itr=0
    x=x0
    y=y0
    xs=[x0]
    ys=[y0]
    
    while itr < 100000:
        itr+=1
        x=x-h*dfdx(x,y)
        y=y-h*dfdy(x,y)
        xs.append(x)
        ys.append(y)
        if abs(dfdx(x, y)) < tol and abs(dfdy(x, y)) < tol:
            itr+=10000
    return xs,ys,itr-100000

print(f'1) {f(sd(1,1)[0][-1],sd(1,1)[1][-1])}, at {(sd(1,1)[0][-1],sd(1,1)[1][-1])}, with {(sd(1,1)[2])} iterations')
print(f'2) {f(osd(1,1)[0][-1],osd(1,1)[1][-1])}, at {(osd(1,1)[0][-1],osd(1,1)[1][-1])}, with {(osd(1,1)[2])} iterations')

        


# #### (c) Construct a plot comparing the path taken by each method in their search for the maximum position. 

# In[7]:


fig,ax=plt.subplots()
ax.plot(sd(1,1)[0],sd(1,1)[1],label='sd')
ax.plot(osd(1,1)[0],osd(1,1)[1],label='osd')

x=np.arange(0.45,1.1,0.01)
y=np.arange(0.75,1.2,.01)
levels=np.arange(-0.9,0,0.05)
X, Y = np.meshgrid(x, y)
z = f(X, Y)
c = ax.contour(x, y, z, levels=levels)
plt.colorbar(c,ax=ax,label='f(x,y)')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


# # problem 3
# #### (a) Set up a linear programming problem that specifies the flows in the channels that maximize profit.
# 
# channel 1 flow rate: $x_1$, channel 2 flow rate: $x_2$
# linearly maximizing profit: = $(3.2\times(1−0.3)+4−1.1)x_1+(3.2\times(1−0.2)+3−1.4)x_2=5.14x_1+4.16x_2$ while $x_1+x_2\le1.4\times10^6,|x_1-x_2|\le0.56\times10^6,and x_1,x_2\ge0$

# #### (b) Solve the linear programming problem with the Simplex method (i.e., by hand, using pen and paper and calculator, or equivalent; do not code).

# In[ ]:





# #### (c) Solve the problem using the linprog function in python.

# In[8]:


from scipy.optimize import linprog
c = [-5.14,-4.16]
A = [[1, 1],[1, -1],[-1, 1]]
b = [1400000,560000,560000]
res = linprog(c, A_ub=A, b_ub=b)
print(f'max profit: {-res.fun} when x1 = {res.x[0]} and x2 = {res.x[1]}')


# #### (d) If the Splish County Water Management Board wanted to increase the profit further, what would you advise them to do? Justify your response.
# 
# I would advise to emphasize channel 1, as there is more than double the profit per unit of flow rate, they would need to lobby against the 40% difference in diverted flow rates cap.

# #### (e) Consider the challenges being faced by the southwestern states and US government as they decide how to manage the rapidly diminishing water resources provided by the Colorado River. Reviewing the recent news, create a list of some of the key parameters and resources that are highlighted as key to the decision-making process and part of the optimization problem. Good starting points for learning more about this issue are: https://www.washingtonpost.com/climate-environment/2023/02/05/colorado-river-drought-explained/ https://www.latimes.com/environment/story/colorado-river-in-crisis https://www.watereducationcolorado.org/
# 
# - how much water we consume (overuse)
# - climate change!!
# - snow runoff
# 

# ## Problem 4 - Random search for global optimization
# 

# Write your own code to implement the random search method and test your code to find the global minimum of the following  function $f(x, y)$.  Report your solution as the $x^*$ and $y^*$ position of the mimimum, as well as the value of function at the mimimum $f(x^*, y^*)$. 

# In[9]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
plt.rcParams['figure.figsize'] = [8,6]
plt.rcParams.update({'font.size':18})


# In[10]:


def gauss(x, y):
    height = 5
    width = 1
    return -height * np.exp(-1/width*(x**2 + y**2))

def parabola(x, y):
    width = 20
    return 1/width*(x**2 + y**2)

def wave(x, y):
    omega = 3
    amplitude = 1
    return amplitude*np.cos(omega*x)*np.cos(omega*y)

def f(x,y):
    return  wave(x,y) + gauss(x,y) + parabola(x,y)


# In[11]:


w = 1.5
h = 1.5

xmin = -w*np.pi
xmax = w*np.pi
ymin = -h*np.pi
ymax = h*np.pi
xvec = np.linspace(xmin, xmax, 100)
yvec = np.linspace(ymin, ymax, 100)

X, Y = np.meshgrid(xvec, yvec)

plt.pcolormesh(X, Y, f(X,Y), cmap='coolwarm')
plt.axis('square')
plt.show()


# In[12]:


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X,Y,f(X,Y),cmap='coolwarm')
plt.show()


# In[13]:


def random_search(N, obj_fn):
    '''
    implement random search
    N: number of random guesses
    obj_fn: (callable), function which to sample'''
        
    Min=0
    for i in range(N):
        rx=xmin+xmax*np.random.uniform(0,2)
        ry=ymin+ymax*np.random.uniform(0,2)
        if obj_fn(rx,ry) < Min:
            x=rx
            y=ry
            Min = obj_fn(rx,ry)

    return x,y,Min

x,y,Min=random_search(100000,f)
print(f'x* = {x}, y* = {y}, f(x*,y*) = {Min}')

