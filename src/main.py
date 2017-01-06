import numpy as np
from numpy.linalg import inv
from scipy.integrate import ode


VERSION = "v0.1"
STATE_SIZE  = 13

def quaternion_to_matrix(quaternion):
    #r i j k
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0]],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0]],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2]]])

def matrix_to_quaternion(m):
    q = np.empty((4, ))
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
        self.x    = np.zeros(3) 
        
        #Rotation matrix
        self.R = np.identity(3)
        #quaternion representig the rotation
        self.q    = matrix_to_quaternion(self.R)
        print(self.q)
        
        #linear momentum
        self.P    = np.zeros(3)
        #angular momentum
        self.L    = np.zeros(3)
        
        self.Iinv  = np.zeros((3,3))
        self.v     = np.zeros(3)
        self.omega = np.zeros(3)
        
        self.force     = np.zeros(3)
        self.torque    = np.zeros(3)

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
    
def compute_dydt(t, y):
    return 0*y
    
def run_simulation():
    # Yvector = [X , q , P, L]   13 
    y0      = np.array(STATE_SIZE)
    t0      = 0
    
    boat = boat_rb(10,np.identity(3))
    boat_to_array(boat,y0)
    r = ode(compute_dydt).set_integrator('vode')
    r.set_initial_value(y0, t0)
    dt = 0.1
    
    while r.successful() and r.t < t1:
        print(r.t+dt, r.integrate(r.t+dt))
    
    
        
        


def main():
    print("SAILBOAT SIMULATOR: "+VERSION)
    
    
    
if __name__ == "__main__":
    main()