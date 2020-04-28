import numpy as np
import matplotlib.pyplot as plt

def min_snap_stable(state_init, state_final, int_points, mpc_dt):

    
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

    delta_t = []
    for i in range(n_points-1):
        p1 = np.array([x[i], y[i], z[i]])
        p2 = np.array([x[i+1], y[i+1], z[i+1]])
        tim = np.linalg.norm(p2 - p1) / 0.5
        delta_t.append(max(tim,0.001))

    P = np.zeros((10*(n_points-1),10*(n_points-1)))
                 
    dt = 1/100.0
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
        
        P[i*10:i*10+10, i*10:i*10+10] = Pb/delta_t[i]**8


    A_inv = np.zeros((10*(n_points - 1),5*n_points))

    A_big = np.zeros((5*n_points, 10*(n_points-1)))

    for i in range(n_points-1):
        dt =  delta_t[i]
        
        Ab0 =np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1/dt, 0],
                      [0, 0, 0, 0, 0, 0, 0, 2/dt**2, 0, 0],
                      [0, 0, 0, 0, 0, 0, 6/dt**3, 0, 0, 0],
                      [0, 0, 0, 0, 0, 24/dt**4, 0, 0, 0, 0]])
        
        Ab1 = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                        [72, 56, 42, 30 , 20, 12, 6, 2, 0, 0],
                        [504, 336, 210, 120, 60, 24, 6, 0, 0, 0],
                        [3024, 1680, 840, 360, 120, 24, 0, 0, 0, 0]])
        
        ab1_vec = np.array([1, 1/dt, 1/dt**2, 1/dt**3, 1/dt**4]).reshape(-1,1)
        Ab1 = np.multiply(Ab1, ab1_vec)
        
        A = np.zeros((10,10))
        A[0:5,0:10] = Ab0
        A[5:10,0:10] = Ab1
        Ain = np.linalg.inv(A)
        #A_big[i*5:i*5+10, i*10:i*10+10] = A
        A_inv[i*10:i*10+10, i*5:i*5+10] = Ain

    #A_inv = np.linalg.inv(A_big)
    #Construct C matrix

    C = np.identity(5*n_points)

    for i in range(n_points-2):
        row = C[i+5+1:i+5+5,:]
        C = np.concatenate((C[:i+5+1,:],C[i+10:,:]), axis=0)
        C = np.concatenate((C,row), axis=0)
    C = np.linalg.inv(C)

    R = C.T.dot(A_inv.T).dot(P).dot(A_inv).dot(C)


    Rpp = R[R.shape[0]-4*(n_points-2):,R.shape[0]-4*(n_points-2):]
    Rfp = R[R.shape[0]-4*(n_points-2):,:R.shape[0]-4*(n_points-2)]

    #construct Df
    df_list = []
    dpstar_list = []
    for i in range(3):
        x1 = state_init[i]
        xd1 = state_init[i+3]
        xdd1 = state_init[i+6]
        x2 = state_final[i]
        xd2 = state_final[i+3]
        xdd2 = state_final[i+6]
        df = [x1, xd1, xdd1, 0,0]
        for point in int_points:
            df.append(point[i])
        df.append(x2)
        df.append(xd2)
        df.append(xdd2)
        df.append(0)
        df.append(0)
        
        df = np.array(df).reshape(-1,1)
        df_list.append(df)
        dpstar = -np.linalg.inv(Rpp).dot(Rfp).dot(df)
        dpstar_list.append(dpstar)

    p_list = []
    for i in range(3):
        d = df_list[i]
        for j in range(n_points-2):
            d = np.concatenate((d[:(5+j*5+1),:], dpstar_list[i][j*4:j*4+4,0:1],d[(6+j*5):,:]),axis=0)

        p = A_inv.dot(d)
        p_list.append(p)
        
    
    params= p_list

    for i in range(n_points-1):
        N = int(delta_t[i]/mpc_dt) -1
        t = np.linspace(0, 1 -1/(N+1), N)
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

        x = params[0][i*10:i*10+10].reshape(1,-1).dot(T)
        y = params[1][i*10:i*10+10].reshape(1,-1).dot(T)
        z = params[2][i*10:i*10+10].reshape(1,-1).dot(T)
        

        dx = x[0,1:N] - x[0,0:N-1]
        dy = y[0,1:N] - y[0,0:N-1]
        dz = z[0,1:N] - z[0,0:N-1]

        ds2 = dx**2 + dy**2 + dz**2
        ds = np.sqrt(ds2)
        s = np.sum(ds)

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

        vx = params[0][i*10:i*10+10].reshape(1,-1).dot(c_vel).dot(T)
        vy = params[1][i*10:i*10+10].reshape(1,-1).dot(c_vel).dot(T)
        vz = params[2][i*10:i*10+10].reshape(1,-1).dot(c_vel).dot(T)


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

        ax = params[0][i*10:i*10+10].reshape(1,-1).dot(c_acc).dot(T)
        ay = params[1][i*10:i*10+10].reshape(1,-1).dot(c_acc).dot(T)
        az = params[2][i*10:i*10+10].reshape(1,-1).dot(c_acc).dot(T)

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


    return s, X_final



# if __name__ == '__main__':
#     state_init = np.array([0,0,0,0,0,0,0,0,0])

#     state_final = np.array([1,1,0,0,0,0,0,0,0])

#     int_points = []
#     t = np.linspace(0.1,0.9,10)
#     t2 = t**2
#     for i in range(10):
#         int_points.append(np.array([t[i],t2[i],0]))


#     mpc_dt = 1/1000.0

#     s, traj = min_snap_stable(state_init, state_final, int_points, mpc_dt)

#     plt.figure()
#     plt.plot(traj[0,:], traj[1,:])
#     plt.figure()
#     plt.title('x-pos')
#     plt.plot(traj[0,:])
#     plt.figure()
#     plt.title('y-pos')
#     plt.plot(traj[1,:])
#     plt.figure()
#     plt.title('x-vel')
#     plt.plot(traj[3,:])
#     plt.figure()
#     plt.title('y-vel')
#     plt.plot(traj[4,:])
#     plt.figure()
#     plt.title('x-acc')
#     plt.plot(traj[6,:])
#     plt.figure()
#     plt.title('y-acc')
#     plt.plot(traj[7,:])

#     plt.show()
#     