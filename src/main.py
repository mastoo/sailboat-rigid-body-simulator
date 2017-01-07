import numpy as np
import numpy.matlib as ml
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.integrate import ode

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

VERSION = "v0.1"
STATE_SIZE  = 13

def quaternion_to_matrix(quaternion):
    #r i j k
    q = ml.matrix(quaternion,dtype=np.float64,copy=True)
    n = (q.T).dot(q)
    if n < np.finfo(float).eps:
        return ml.identity(4)
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    return ml.matrix([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0]],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0]],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2]]])

def matrix_to_quaternion(m):
    q = ml.zeros((4,1))
    tr = np.trace(m)
    
    if tr>=0:
        s = np.sqrt(tr + 1);
        q[0] = 0.5 * s;
        s = 0.5 / s;
        q[1] = (m[2,1] - m[1,2]) * s;
        q[2] = (m[0,2] - m[2,0]) * s;
        q[3] = (m[1,0] - m[0,1]) * s;
    else:
        i = 0;
        if(m[1,1] > m[0,0]):
            i = 1;
        if(m[2,2] > m[i,i]):
            i = 2;
            
        if i == 0:        
            s = np.sqrt((m[0,0] - (m[1,1] + m[2,2])) + 1);
            q[1] = 0.5 * s;
            s = 0.5 / s;
            q[2] = (m[0,1] + m[1,0]) * s;
            q[3] = (m[2,0] + m[0,2]) * s;
            q[0] = (m[2,1] - m[1,2]) * s;

        elif i == 1:
            s = np.sqrt((m[1,1] - (m[2,2] + m[0,0])) + 1);
            q[2] = 0.5 * s;
            s = 0.5 / s;
            q[3]= (m[1,2] + m[2,1]) * s;
            q[1] = (m[0,1] + m[1,0]) * s;
            q[0] = (m[0,2] - m[2,0]) * s;
        elif i == 2:

            s = np.sqrt((m[2,2] - (m[0,0] + m[1,1])) + 1);
            q[3] = 0.5 * s;
            s = 0.5 / s;
            q[1] = (m[2,0] + m[0,2]) * s;
            q[2] = (m[1,2] + m[2,1]) * s;
            q[0] = (m[1,0] - m[0,1]) * s;    
    return q

    

        
#rigid body model of a boat
class boat_rb:
    
    #mass: totale mass of the boat,
    #I: Body Frame inertia Tensor (3x3 Matrix)
    def __init__(self, mass, I):
        self.mass = mass
        self.Ibody    = I
        self.Ibodyinv = inv(I)
        
        #current position of center mass in the world reference
        self.x = ml.zeros((3,1)) 
        
        #Rotation matrix
        self.R = ml.identity(3)
        #quaternion representig the rotation
        self.q    = matrix_to_quaternion(self.R)
        print(self.q)
        
        #linear momentum
        self.P    = ml.zeros((3,1))
        #angular momentum
        self.L    = ml.zeros((3,1))
        
        self.Iinv  = ml.zeros((3,3))
        self.v     = ml.zeros((3,1))
        self.omega = ml.zeros((3,1))
        
        self.force     = ml.zeros((3,1))
        self.torque    = ml.zeros((3,1))

def array_to_boat(y, boat):
    boat.x[0] = y[0]
    boat.x[1] = y[1]
    boat.x[2] = y[2]
    
    boat.q[0] = y[3]
    boat.q[1] = y[4]
    boat.q[2] = y[5]
    boat.q[3] = y[6]
    
    boat.P[0] = y[7]
    boat.P[1] = y[8]
    boat.P[2] = y[9]    
    
    boat.L[0] = y[10]
    boat.L[1] = y[11]
    boat.L[2] = y[12]
    
    boat.q = boat.q / norm(boat.q) #normalize quaternion
    boat.R = quaternion_to_matrix(boat.q)
    boat.v = boat.P/boat.mass
    boat.Iinv = boat.R*boat.Ibodyinv*np.transpose(boat.R)
   
    boat.omega = boat.Iinv*boat.L
    

def boat_to_array(boat, y):
    y[0] = boat.x[0]
    y[1] = boat.x[1]
    y[2] = boat.x[2]
    
    y[3] = boat.q[0]
    y[4] = boat.q[1]
    y[5] = boat.q[2]
    y[6] = boat.q[3]
             
    y[7] = boat.P[0]
    y[8] = boat.P[1]
    y[9] = boat.P[2]
    
    y[10] = boat.L[0]
    y[11] = boat.L[1]
    y[12] = boat.L[2]

def star(a):

    return ml.matrix([
        [0     ,-a[2],  a[1]],
        [a[2]  ,    0, -a[0]],
        [ -a[1], a[0],    0]])    
        

def compute_force_and_torque(boat):
    boat.force  = np.zeros((3,1))
    boat.force[0,0] = 1
    
    boat.torque = np.zeros((3,1))

    boat.torque[1,0] = 0.5

    boat.torque[2,0] = 0.5

def q_mult(q1, q2):
    s1 = q1[0,0]
    s2 = q2[0,0]
    
    v1 = q1[1:,0]
    v2 = q2[1:,0]
    r = ml.zeros((4,1))
    
    r[0,0] = s1*s2 - (v1.T).dot(v2)
    r[1:,0] = s1*v2+s2*v1 + np.cross(v1,v2,axis=0)

    return r
    
    
def compute_dydt(t, y, boat):
    array_to_boat(y, boat)
    compute_force_and_torque(boat)
    z = y*0
    #dx/dt = v 
    z[0] = boat.v[0]
    z[1] = boat.v[1]
    z[2] = boat.v[2]
    

    atmp = ml.zeros((4,1))
    atmp[0] = 0
    atmp[1] = boat.omega[0]
    atmp[2] = boat.omega[1]
    atmp[3] = boat.omega[2]
    
    qdot = 0.5*q_mult(atmp,boat.q);
    
    z[3] = qdot[0]
    z[4] = qdot[1]
    z[5] = qdot[2]
    z[6] = qdot[3]
    
    z[7] = boat.force[0]
    z[8] = boat.force[1]
    z[9] = boat.force[2]
    
    z[10] = boat.torque[0]
    z[11] = boat.torque[1]
    z[12] = boat.torque[2]
    
    return z

def to_ssv(y):   
    return  " ".join("{:.7f}".format(n) for n  in y.flat)

    
def run_simulation():
    # Yvector = [X , q , P, L]   13 
    y0      = ml.zeros((STATE_SIZE,1))
    t0      = 0
    
    boat = boat_rb(10,ml.identity(3))
    boat_to_array(boat,y0)
    
    print("y0")
    print(y0)
    print("end y0")
    r = ode(compute_dydt).set_integrator('vode').set_f_params(boat)
    r.set_initial_value(y0, t0)
    dt = 0.1
    t1 = 10
    #saving to file
    outfile = open("simulation.dat", 'w')
    outfile.write("#t x y z q.s q.x q.y q.z P.x P.y P.z L.x L.y L.z\n") 
    outfile.write("{:.7f}".format(t0)+" "+to_ssv(y0)+"\n")
    
    y_data = ml.zeros((STATE_SIZE,10))
    
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    i = 0
    while r.successful() and i < 10:
        y_new = r.integrate(r.t+dt)
        y_data[:,i] = y_new
        print(r.t+dt, y_new)
        outfile.write("{:.7f}".format(r.t+dt)+" "+to_ssv(y_new)+"\n")
        i += 1

    xd = y_data[0,:]
    yd = y_data[1,:]
    zd = y_data[2,:]
    
    xd = np.reshape(np.array(xd), xd.size)
    yd = np.reshape(np.array(yd), yd.size)
    zd = np.reshape(np.array(zd), zd.size)
    
    
    u =  np.zeros (xd.size)
    v =  np.ones(yd.size)
    w =  np.zeros(zd.size)
    
    u1 =  np.zeros (xd.size)
    v1 =  np.zeros(yd.size)
    w1 =  np.ones(zd.size)
    
    for i in range(xd.size):
        q = ml.zeros((4,1)) 
        q[0,0] = y_data[3,i]
        q[1,0] = y_data[4,i]
        q[2,0] = y_data[5,i]
        q[3,0] = y_data[6,i]
        R = quaternion_to_matrix(q)
        
        ax1 = ml.zeros((3,1))
        ax1[0] = u[i]
        ax1[1] = v[i]
        ax1[2] = w[i]
        ax1 = R*ax1
        u[i] = ax1[0]
        v[i] = ax1[1]
        w[i] = ax1[2]
        
        ax2 = ml.zeros((3,1))
        ax2[0] = u1[i]
        ax2[1] = v1[i]
        ax2[2] = w1[i]
        ax2 = R*ax2
        u1[i] = ax2[0]
        v1[i] = ax2[1]
        w1[i] = ax2[2]        
    
    ax.plot(xd, yd, zs=zd, zdir='z', label='zs=0, zdir=z')
    ax.quiver(xd, yd,zd,u,v,w,length=0.01,pivot="tail")
    
    ax.quiver(xd, yd,zd,u1,v1,w1,length=0.01,pivot="tail")
    plt.show()


def main():
    print("SAILBOAT SIMULATOR: "+VERSION)
    print("START SIMULATION")
    run_simulation()
    
    
if __name__ == "__main__":
    main()