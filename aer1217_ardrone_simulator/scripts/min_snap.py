import numpy as np
from cvxopt import matrix, solvers


def min_snap_trajectory(state_init, state_final, int_points, mpc_dt, speed=1.0):
    """Function that generates minimum snap trajectories
    Args:
        state_init: initial quadcopter state
        state_final: final quadcopter state
        int_points: intermediate waypoints
        mpc_dt: delta t required for mpc
        speed: speed needed to calculate the timing for each segment

    Returns:
        total_dist: total distance covered by the trajectory
        X_final: an array containing all the states forming the trajectory

    """
    
    #Sum number of points to be fitted
    n_points = len(int_points)+2

    #create points list
    x = [state_init[0]]
    y = [state_init[1]]
    z = [state_init[2]]
    for point in int_points:
        x.append(point[0])
        y.append(point[1])
        z.append(point[2])
    x.append(state_final[0])
    y.append(state_final[1])
    z.append(state_final[2])

    #create a list of times for each trajectory segment
    delta_t = []
    for i in range(n_points-1):
        p1 = np.array([x[i], y[i], z[i]])
        p2 = np.array([x[i+1], y[i+1], z[i+1]])
        tim = np.linalg.norm(p2 - p1) / speed
        delta_t.append(tim)

    print('Total_time = ', np.sum(delta_t))

    #Construct P matrix (penalizes the snap)
    P = np.zeros((30*(n_points-1),30*(n_points-1)))

    dt = 1/50.0
    ti = np.arange(0,1,dt)
    t10 = np.sum(ti**10*dt)
    t9 = np.sum(ti**9*dt)
    t8 = np.sum(ti**8*dt)
    t7 = np.sum(ti**7*dt)
    t6 = np.sum(ti**6*dt)
    t5 = np.sum(ti**5*dt)
    t4 = np.sum(ti**4*dt)
    t3 = np.sum(ti**3*dt)
    t2 = np.sum(ti**2*dt)
    t = np.sum(ti*dt)

    Pb = np.zeros((10,10))

    Pb[0:6,0:6] = np.array([[3024**2*t10, 3024*1680*t9, 3024*840*t8, 
                             3024*360*t7,3024*120*t6, 3024*24*t5],
                            [3024*1680*t9, 1680**2*t8, 1680*840*t7, 
                             1680*360*t6, 1680*120*t5, 1680*24*t4],
                            [3024*840*t8, 1680*840*t7, 840**2*t6, 
                             840*350*t5, 840*120*t4, 840*24*t3],
                            [3024*360*t7,1680*360*t6, 840*360*t5, 
                            360**2*t4, 360*120*t3, 360*24*t2],
                            [3024*120*t6, 1680*120*t5, 840*120*t4,
                             360*120*t3, 120**2*t2, 120*24*t],
                            [3024*24*t5, 1680*24*t4, 840*24*t3,
                             360*24*t2, 120*24*t, 24**2]])

    for i in range(n_points-1):        
        P[i*30:i*30+10, i*30:i*30+10] = Pb/delta_t[i]**8
        P[i*30+10:i*30+20, i*30+10:i*30+20] = Pb/delta_t[i]**8
        P[i*30+20:i*30+30, i*30+20:i*30+30] = Pb/delta_t[i]**8


    #Construct A and b matrices
    A = np.zeros((18*(n_points)-6, 30*(n_points-1)))

    b = np.zeros((18*(n_points)-6,1))

    b[0,0] = state_init[0]
    b[1,0] = state_init[3]
    b[2,0] = state_init[6]
    b[5,0] = x[1]

    b[6,0] = state_init[1]
    b[7,0] = state_init[4]
    b[8,0] = state_init[7]
    b[11,0] = y[1]

    b[12,0] = state_init[2]
    b[13,0] = state_init[5]
    b[17,0] = z[1]


    for i in range(n_points-1):
        dt1 = delta_t[i]

        Ab0 =np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1/dt1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 2/dt1**2, 0, 0],
                  [0, 0, 0, 0, 0, 0, 6/dt1**3, 0, 0, 0],
                  [0, 0, 0, 0, 0, 24/dt1**4, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

        A0 = np.zeros((18,30))

        A0[0:6, 0:10] = Ab0
        A0[6:12, 10:20] = Ab0
        A0[12:18, 20:30] = Ab0

        A[i*18:i*18+18, i*30:i*30+30] = A0
        if i >0:
            dt2 = delta_t[i-1]

            ab1_vec = np.array([1, 1/dt2, 1/dt2**2, 1/dt2**3, 1/dt2**4, 1]).reshape(-1,1)

            Ab1 = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                        [72, 56, 42, 30 , 20, 12, 6, 2, 0, 0],
                        [504, 336, 210, 120, 60, 24, 6, 0, 0, 0],
                        [3024, 1680, 840, 360, 120, 24, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
             
            Ab1 = np.multiply(Ab1, ab1_vec)

            A1 = np.zeros((18,30))

            A1[0:6, 0:10] = Ab1
            A1[6:12, 10:20] = Ab1
            A1[12:18, 20:30] = Ab1
            
            A[i*18:i*18+18, (i-1)*30:(i-1)*30+30] = -A1
            b[i*18+5, 0] = x[i+1]
            b[i*18+11, 0] = y[i+1]
            b[i*18+17, 0] = z[i+1]
        
    dt2 = delta_t[-1]
    ab1_vec = np.array([1, 1/dt2, 1/dt2**2, 1/dt2**3, 1/dt2**4, 1]).reshape(-1,1)
    Ab1 = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                [72, 56, 42, 30 , 20, 12, 6, 2, 0, 0],
                [504, 336, 210, 120, 60, 24, 6, 0, 0, 0],
                [3024, 1680, 840, 360, 120, 24, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    Ab1 = np.multiply(Ab1, ab1_vec)
        
    A[(i+1)*18:(i+1)*18+4, i*30:i*30+10] = Ab1[1:5, 0:10]
    A[(i+1)*18+4:(i+1)*18+8, i*30+10:i*30+20] = Ab1[1:5, 0:10]
    A[(i+1)*18+8:(i+1)*18+12, i*30+20:i*30+30] = Ab1[1:5, 0:10]


    b[(i+1)*18, 0] = state_final[3]
    b[(i+1)*18+1, 0] = state_final[6]

    b[(i+1)*18+4, 0] = state_final[4]
    b[(i+1)*18+5, 0] = state_final[7]

    b[(i+1)*18+8, 0] = state_final[5]

    q = np.zeros((30*(n_points-1),1))
    G = np.zeros((30*(n_points-1),30*(n_points-1)))
    h = np.zeros((30*(n_points-1),1))


    P = matrix(P)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    q = matrix(q)
    b = matrix(b)

    #silent solver 
    solvers.options['show_progress'] = False
    #solvers.options['show_progress'] = False

    sol = solvers.qp(P, q, G, h, A, b)
    params = np.array(sol['x'])

    total_dist = 0
    for i in range(n_points-1):
        step = delta_t[i]/mpc_dt
        t = np.arange(0, 1, 1/step)
        N = t.shape[0]
        T = np.zeros((10, N))
        T[0,:] = t**9
        T[1,:] = t**8
        T[2,:] = t**7
        T[3,:] = t**6
        T[4,:] = t**5
        T[5,:] = t**4
        T[6,:] = t**3
        T[7,:] = t**2
        T[8,:] = t
        T[9,:] = np.ones(N)

        x = params[i*30:i*30+10].reshape(1,-1).dot(T)
        y = params[i*30+10:i*30+20].reshape(1,-1).dot(T)
        z = params[i*30+20:i*30+30].reshape(1,-1).dot(T)

        dx = x[0,1:N] - x[0,0:N-1]
        dy = y[0,1:N] - y[0,0:N-1]
        dz = z[0,1:N] - z[0,0:N-1]
        
        ds2 = dx**2 + dy**2 + dz**2
        ds = np.sqrt(ds2)
        s = np.sum(ds)

        total_dist += s

        #calculate velocities
        T = np.zeros((10, N))
        T[0,:] = t**8
        T[1,:] = t**7
        T[2,:] = t**6
        T[3,:] = t**5
        T[4,:] = t**4
        T[5,:] = t**3
        T[6,:] = t**2
        T[7,:] = t**1
        T[8,:] = np.ones(N)
        T[9,:] = np.zeros(N)

        T = T/delta_t[i]

        c_vel = np.diag([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
        
        vx = params[i*30:i*30+10].reshape(1,-1).dot(c_vel).dot(T)
        vy = params[i*30+10:i*30+20].reshape(1,-1).dot(c_vel).dot(T)
        vz = params[i*30+20:i*30+30].reshape(1,-1).dot(c_vel).dot(T)
    
        
        #calculate accelerations
        T = np.zeros((10, N))
        T[0,:] = t**7
        T[1,:] = t**6
        T[2,:] = t**5
        T[3,:] = t**4
        T[4,:] = t**3
        T[5,:] = t**2
        T[6,:] = t**1
        T[7,:] = np.ones(N)
        T[8,:] = np.zeros(N)
        T[9,:] = np.zeros(N)

        T = T/delta_t[i]**2

        c_acc = np.diag([72, 56, 42, 30, 20, 12, 6, 2, 0, 0])
        
        ax = params[i*30:i*30+10].reshape(1,-1).dot(c_acc).dot(T)
        ay = params[i*30+10:i*30+20].reshape(1,-1).dot(c_acc).dot(T)
        az = params[i*30+20:i*30+30].reshape(1,-1).dot(c_acc).dot(T)
        
        #Create a matrix of all points
        X = np.zeros((9,N))
        X[0,:] = x.flatten()
        X[1,:] = y.flatten()
        X[2,:] = z.flatten()
        X[3,:] = vx.flatten()
        X[4,:] = vy.flatten()
        X[5,:] = vz.flatten()
        X[6,:] = ax.flatten()
        X[7,:] = ay.flatten()
        X[8,:] = az.flatten()

        if i == 0:
            X_final = X
        else:
            X_final = np.concatenate((X_final, X), axis=1)

    return total_dist, X_final









