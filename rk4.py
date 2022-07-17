import numpy as np
import matplotlib.pyplot as plt

def rk4(t,x, f, h = 0.1):
	k1 = f(t,x)
	k2 = f(t+h/2, x+h*k1/2)
	k3 = f(t+h/2, x+h*k2/2)
	k4 = f(t + h, x+h*k3)
	return x + 1/6*h*(k1+2*k2+2*k3+k4)

def solve(t0, x0, f, tf = 10, step = 0.01):
    x0 = np.float64(x0)
    t = np.arange(t0, tf, step)
    n = t.size
    solution = np.array(n*[x0])
    for i in range(n-1):
        solution[i+1] = rk4(t[i], solution[i], f, step)
    return np.reshape(t,(n)),solution

def f(t,x):
    #lorents attractor, you can change here the equation to solve in the form dx/dt = f(t,x)
    sigma = 10
    rho = 28
    beta = 8/3
    x1 = sigma*(x[1]-x[0])
    x2 = x[0]*(rho-x[2])-x[1]
    x3 = x[0]*x[1]-beta*x[2]
    return np.array([x1,x2,x3])


def main():
    t,x = solve(0,[0,1,0],f, 100)
    #plot a 3d curve, if you change the dimensionaltiy of x and f(), you will need to change the plot
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(x[:,0], x[:,1], x[:,2], 'gray')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    plt.show()


if __name__ == '__main__':
    main()