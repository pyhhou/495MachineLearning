from pylab import *
from mpl_toolkits.mplot3d import Axes3D

###### ML Algorithm functions ######
def gradient_descent(w0):
    w = w0
    g_path = []
    w_path = []
    w_path.append(w)
    #g_path.append(-cos(2*pi*dot(w.T,w)) + 2*dot(w.T,w))       # 2.13
    g_path.append(np.log(1 + np.e ** dot(w.T,w)))
    # start gradient descent loop
    grad_1 = 1
    grad_2 = 1
    iter = 1
    max_its = 10
    while iter <= max_its:
        # take gradient step
        #grad = 2*w + 4*pi*sin(2*pi*dot(w.T,w))*w   # 2.13
        I = np.eye(2)
        grad_1 = (1 / (1 + np.e ** dot(w.T,w))) * 2 * (np.e ** (dot(w.T,w))) * w
        grad_2 = 4*(np.e**dot(w.T,w))*dot(w,w.T) / (1 + np.e**dot(w.T,w))**2 + (2 * np.e**dot(w.T,w)) / (1 + np.e**dot(w.T,w)) * I
        w = w - dot(linalg.inv(grad_2),grad_1)
        print w        
        # update path containers
        w_path.append(w)
        #g_path.append(-cos(2*pi*dot(w.T,w)) + 2*dot(w.T,w))    # 2.13
        g_path.append(np.log(1 + np.e ** dot(w.T,w)))           # 2.17
        iter+= 1
    g_path = asarray(g_path)
    g_path.shape = (iter,1)
    w_path = asarray(w_path)
    w_path.shape = (iter,2)

# show final average gradient norm for sanity check
    # s = dot(grad.T,grad)/2
    # s = 'The final average norm of the gradient = ' + str(float(s))
    # print(s)
    
    
    # # for use in testing if algorithm minimizing/converging properly
    # plot(asarray(obj_path))
    # show()
    return w_path,g_path

###### plotting functions #######
def make_function():
    global fig,ax1
    
    # prepare the function for plotting
    r = linspace(-1.15,1.15,300)
    s,t = meshgrid(r,r)
    s = reshape(s,(size(s),1))
    t = reshape(t,(size(t),1))
    h = concatenate((s,t),1)
    h = dot(h*h,ones((2,1)))
    #b = -cos(2*pi*h) + 2*h
    b = np.log(1 + np.e ** h)
    s = reshape(s,(int(sqrt(size(s))),int(sqrt(size(s)))))
    t = reshape(t,(int(sqrt(size(t))),int(sqrt(size(t)))))
    b = reshape(b,(int(sqrt(size(b))),int(sqrt(size(b)))))
    
    # plot the function
    fig = plt.figure(facecolor = 'white')
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot_surface(s,t,b,cmap = 'Greys',antialiased=False) # optinal surface-smoothing args rstride=1, cstride=1,linewidth=0
    ax1.azim = 115
    ax1.elev = 70
    
    # pretty the figure up
    ax1.xaxis.set_rotate_label(False)
    ax1.yaxis.set_rotate_label(False)
    ax1.zaxis.set_rotate_label(False)
    ax1.get_xaxis().set_ticks([-1,1])
    ax1.get_yaxis().set_ticks([-1,1])
    ax1.set_xlabel('$w_0$   ',fontsize=20,rotation = 0,linespacing = 10)
    ax1.set_ylabel('$w_1$',fontsize=20,rotation = 0,labelpad = 50)
    ax1.set_zlabel('   $g(\mathbf{w})$',fontsize=20,rotation = 0,labelpad = 20)

def plot_steps(w_path,g_path):
    # colors for points
    ax1.plot(w_path[:,0],w_path[:,1],g_path[:,0],color = [1,0,1],linewidth = 5)   # add a little to output path so its visible on top of the surface plot
    ax1.plot(w_path[-8:-1,0],w_path[-8:-1,1],g_path[-8:-1,0],color = [1,0,0],linewidth = 5)   # add a little to output path so its visible on top of the surface plot


def main():
    make_function()                                 # plot objective function
    
    # plot first run on surface
    #alpha = 10**-2
    w0 = array([1.0,1.0])
    w0.shape = (2,1)
    w_path,g_path = gradient_descent(w0)            # perform gradient descent
    plot_steps(w_path,g_path)
    
    # plot second run on surface
    # w0 = array([4.0,4.0])
    # w0.shape = (2,1)
    # w_path,g_path = gradient_descent(w0)    # perform gradient descent
    # plot_steps(w_path,g_path)
    show()
main()
