import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,1,100)
h = np.heaviside(x-1,1)
y = x
y_0 = x**(2*0) # y = x^2k, k = 0
y_1 = x**(2*1) # y = x^2k, k = 1
y_2 = x**(2*2) # y = x^2k, k = 2
y_3 = x**(2*3) # y = x^2k, k = 3
y_10 = x**(2*10) # y = x^2k, k = 10
y_100 = x**(2*100) # y = x^2k, k = 100
# Derivatives
dy_0 = np.zeros(len(x)) # y = 2k*x^(2k-1), k = 0
dy_1 = 2*1*x**(2*1-1) # y = 2k*x^(2k-1), k = 1
dy_2 = 2*2*x**(2*2-1) # y = 2k*x^(2k-1), k = 2
dy_3 = 2*3*x**(2*3-1) # y = 2k*x^(2k-1), k = 3
dy_10 = 2*10*x**(2*10-1) # y = 2k*x^(2k-1), k = 10
dy_100 = 2*100*x**(2*100-1) # y = 2k*x^(2k-1), k = 100

e = np.linalg.norm(h - y)
e_0 = np.linalg.norm(h - y_0)
e_1 = np.linalg.norm(h - y_1)
e_2 = np.linalg.norm(h - y_2)
e_3 = np.linalg.norm(h - y_3)
e_10 = np.linalg.norm(h - y_10)
e_100 = np.linalg.norm(h - y_100)

fig, ax = plt.subplots()
ax.set_title('Communication Weight Function')
ax.plot(x,h,'-',color='black',linewidth=2,label= r'$H(x)$')
plt.annotate(r'y=x', xy=(x[45],y[50]), rotation=40)
ax.plot(x,y,'--',color='black',linewidth=2,label= r'$y=x$')
plt.annotate(r'k=0', xy=(x[50],y_0[50]), rotation=0)
ax.plot(x,y_0,'-',color='blue',linewidth=1,label= r'$y=x^{2k}$')
plt.annotate(r'k=1', xy=(x[50],y_1[55]), rotation=45)
ax.plot(x,y_1,'-',color='blue',linewidth=1)
plt.annotate(r'k=2', xy=(x[60],y_2[65]), rotation=45)
ax.plot(x,y_2,'-',color='blue',linewidth=1)
plt.annotate(r'k=3', xy=(x[65],y_3[70]), rotation=45)
ax.plot(x,y_3,'-',color='blue',linewidth=1)
plt.annotate(r'k=10', xy=(x[80],y_10[85]), rotation=45)
ax.plot(x,y_10,'-',color='blue',linewidth=1)
plt.annotate(r'k=100', xy=(x[89],y_100[90]), rotation=45)
ax.plot(x,y_100,'-',color='blue',linewidth=1)
ax.set_ylabel(r'$y$', color = 'black', fontweight = 'bold')
ax.set_xlabel(r'$x$', color = 'black', fontweight = 'bold')
ax.grid()
ax.legend()
fig, ax2 = plt.subplots()
ax2.set_title('Communication Weight Function (Derivative)')
plt.annotate(r'k=0', xy=(x[50],dy_0[50]), rotation=0)
ax2.plot(x,dy_0,'-',color='blue',linewidth=1,label= r'$\frac{\delta y}{\delta x}=2kx^{2k-1}$')
plt.annotate(r'k=1', xy=(x[50],y_1[55]), rotation=45)
ax2.plot(x,dy_1,'-',color='blue',linewidth=1)
plt.annotate(r'k=2', xy=(x[60],y_2[65]), rotation=45)
ax2.plot(x,dy_2,'-',color='blue',linewidth=1)
plt.annotate(r'k=3', xy=(x[65],y_3[70]), rotation=45)
ax2.plot(x,dy_3,'-',color='blue',linewidth=1)
plt.annotate(r'k=10', xy=(x[80],y_10[85]), rotation=45)
ax2.plot(x,dy_10,'-',color='blue',linewidth=1)
plt.annotate(r'k=100', xy=(x[89],y_100[90]), rotation=45)
ax2.plot(x,dy_100,'-',color='blue',linewidth=1)
ax2.set_ylabel(r'$y$', color = 'black', fontweight = 'bold')
ax2.set_xlabel(r'$x$', color = 'black', fontweight = 'bold')
ax2.set_xlim([-0.05, 1.05])
ax2.set_ylim([-0.05, 1.05])
ax2.grid()
ax2.legend()
fig, ax3 = plt.subplots()
ax3.set_title(r'Heaviside Approximation Error ($L2$ Norm)')
ax3.plot([0,0.5,1,2,3,10,100],[e_0,e,e_1,e_2,e_3,e_10,e_100],'-',color='red',linewidth=2,label=r'$L2$ Norm '+r'$H(x)-x^{2k}$')
ax3.set_ylabel('Error', color = 'black', fontweight = 'bold')
ax3.set_xlabel(r'$k$', color = 'black', fontweight = 'bold')
ax3.grid()
ax3.legend()
plt.show()

