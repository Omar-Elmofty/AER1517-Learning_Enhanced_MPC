#!/usr/bin/env python2

"""ROS Node for publishing desired positions."""

from __future__ import division, print_function, absolute_import

# Import ROS libraries
import roslib
import rospy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Import class that computes the desired positions
# from aer1217_ardrone_simulator import PositionGenerator

from aer1217_ardrone_simulator.msg import FlatState
from std_msgs.msg import Int16
from narrow_window import Window
from helper_functions import nearby_nodes, get_total_cost, truncate, min_snap_trajectory
from min_snap_stable import min_snap_stable
from steer import steer
import matplotlib.animation as animation
from matplotlib import style
import time


class RRT_star_planner(object):
    def __init__(self):

        # Publishers
        self.pub_rrt = rospy.Publisher('/aer1217/rrt_path', 
                                      FlatState, queue_size=32)

        self.pub_send_comp = rospy.Publisher('/aer1217/rrt_send_complete', 
                                    Int16,
                                    queue_size=32)

        self.rrt_msg = FlatState()

        #define global variables
        self.G = {0:{'parent':None,'state':np.array([0, 0, 2, 0, 0, 0, 0, 0, 0]), 'cost': 0, 'path':None, 'free':True}}
        self.state_goal = np.array([12, 3, 2, 0, 0, 0, 0, 0, 0])
        self.r_search = 2
        self.max_iter = 400
        self.dt_des = 1/10.0

        #steer function order
        self.steer_order = 5

        #define search range
        self.x_range = [0,13]
        self.y_range = [-2,4]
        self.z_range = [0,6]

        #initiate narrow window
        T1= np.identity(4)
        T1[0:3,3] = np.array([4,-1,2])
        win1 = Window(1,1,1,3.5,T1)

        T2= np.identity(4)
        T2[0:3,3] = np.array([4,3,2])
        win2 = Window(1,1,1,-4,T2)

        T3= np.identity(4)
        T3[0:3,3] = np.array([8,1,2])
        win3 = Window(1,1,1,2.7,T3)

        T4= np.identity(4)
        T4[0:3,3] = np.array([8,3,2])
        win4 = Window(1,1,1,-3.1,T4)

        self.windows = [win1, win2, win3, win4]

        #define circular obstacles
        self.obs_rad = 0.8
        self.obs_locs = [np.array([2,0.5]), np.array([6,2]), np.array([10,1.5])]

    def check_in_window(self, p_rand):

        #check if random point lies within window range
        for win in self.windows:
            if win.check_in_window(p_rand):
                return True, win

        return False, None

    def check_collision(self, p_rand, p_nearest):

        #check collision with walls
        for win in self.windows:
            if win.check_collision(p_rand, p_nearest):
                return True

        #check collision with circular obstacles
        t = np.linspace(0,1,20)
        for i in range(20):
            p = p_nearest + t[i] * (p_rand - p_nearest)
            for obs_loc in self.obs_locs:
                if np.linalg.norm(p[0:2] - obs_loc) < self.obs_rad:
                    return True

        return False

    def check_path_collision(self, X):

        #check collision with circular obstacles
        for i in range(X.shape[1]):
            p = X[0:3,i].flatten()

            for obs_loc in self.obs_locs:
                if np.linalg.norm(p[0:2] - obs_loc) < self.obs_rad:
                    return True
            if i>0:
                p2 = X[0:3,i-1].flatten()
                #check collision with walls
                for win in self.windows:
                    if win.check_collision(p, p2):
                        return True

        return False

    def rrt_plan(self):

        #Iterate to get converges state
        for it in range(1,self.max_iter):
            if it%100 == 0:
                print('iteration = ', it)

            #Random sample random point
            state_rand = np.zeros(9)
            state_rand[0] = np.random.uniform(self.x_range[0],self.x_range[1])
            state_rand[1] = np.random.uniform(self.y_range[0],self.y_range[1])
            state_rand[2] = 2 #np.random.uniform(self.z_range[0],self.z_range[1])
            angle_rand = np.random.uniform(-np.pi/2., np.pi/2.)
            state_rand[3:5] = np.array([np.cos(angle_rand), np.sin(angle_rand)])

            #randomly sample a window every 10 iterations, else random rest of space
            if it%10 == 0:
                win = np.random.choice(self.windows)
                state_rand, next_node = win.generate_parabolic_nodes(it, self.dt_des)
                in_win = True
            else:
                #check if in window
                in_win, win = self.check_in_window(state_rand[0:3])
                if in_win:
                    state_rand, next_node = win.generate_parabolic_nodes(it, self.dt_des)
             
            #look for nearby nodes
            Near_nodes, Nearest_node, key_Nearest_node = nearby_nodes(state_rand,self.G,self.r_search)
            
            #Connect to best node
            min_cost = float('inf')
            best_node = None
            for key in Near_nodes.keys():
                node = self.G[key]
                
                # if self.check_collision(state_rand[0:3], node['state'][0:3]):
                #     continue
                stage_cost, X = steer(node['state'], state_rand, self.dt_des, self.steer_order)

                if self.check_path_collision(X):
                    continue

                total_cost = stage_cost + get_total_cost(self.G, key)
                
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_node = key
                    best_path = X
                    best_cost = stage_cost
            
            #Continue if node is not found
            if best_node == None:
                continue

            #Wire new node
            self.G[it] = {'parent':best_node,'state':state_rand, 'cost':best_cost, 'path':best_path, 'free': True}  

            #Wire next node if in window
            if in_win:
                self.G[-it] = next_node

            #rewire close nodes to reduce cost
            for key in Near_nodes.keys():
                node = self.G[key]
                
                # if self.check_collision(node['state'][0:3], state_rand[0:3]):
                #     continue
                    
                stage_cost, X = steer(state_rand,node['state'], self.dt_des, self.steer_order)

                if self.check_path_collision( X):
                    continue

                total_cost = stage_cost + get_total_cost(self.G, it)
                
                if total_cost < node['cost']:
                    self.G[key]['parent'] = it
                    self.G[key]['cost'] = stage_cost
                    self.G[key]['path'] = X
        

        #find best node to connect to goal
        min_cost = float('inf')
        best_node = None
        Near_nodes,Nearest_node, key_Nearest_node = nearby_nodes(self.state_goal,self.G,self.r_search)
        for key in Near_nodes.keys():
            node = self.G[key]
            
            # if self.check_collision(self.state_goal[0:3], node['state'][0:3]):
            #     continue
                    
            stage_cost, X = steer(node['state'],self.state_goal, self.dt_des, self.steer_order)

            if self.check_path_collision(X):
                continue

            total_cost = stage_cost + get_total_cost(self.G, key)
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_node = key
                best_path = X
                    
        #wire goal state
        self.G['goal'] = {'parent':best_node,'state':self.state_goal, 'cost':min_cost, 'path':best_path, 'free':True}  

        #generate best path
        best_path = [self.G['goal']]
        parent = best_node
        while parent != None:
            best_path.append(self.G[parent])
            parent = self.G[parent]['parent']

        return best_path
    
    def plot_path(self, best_path, traj):

        #Plotting Results:

        fig, ax = plt.subplots()
        for obs in self.obs_locs:
            circle = plt.Circle((obs[0], obs[1]), self.obs_rad, color='r')
            ax.add_artist(circle)
        for key in self.G.keys():
            pos = self.G[key]['state'][0:3]
            ax.plot([pos[0]],[pos[1]],'ro')
            parent_key = self.G[key]['parent']
            if parent_key != None:
                parent_pos = self.G[parent_key]['state'][0:3]
                ax.plot([pos[0],parent_pos[0]],[pos[1],parent_pos[1]],'b')

        plt.xlim(self.x_range)
        plt.ylim(self.y_range)

        #plot the shortest path
        #fig, ax = plt.subplots()
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        # for obs in self.obs_locs:
        #     circle = plt.Circle((obs[0], obs[1]), self.obs_rad, color='r')
        #     ax.add_artist(circle)
        for i in range(len(best_path)-1):
            x = best_path[i]['path'][0,:]
            y = best_path[i]['path'][1,:]
            z = best_path[i]['path'][2,:]
            ax.plot(x,y,z,'b')

        #fig, ax = plt.subplots()
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        # for obs in self.obs_locs:
        #     circle = plt.Circle((obs[0], obs[1]), self.obs_rad, color='r')
        #     ax.add_artist(circle)
        ax.plot(traj[0,:], traj[1,:],traj[2,:], 'b.')
        ax.set_xlim3d(0, 12)
        ax.set_ylim3d(0, 12)
        ax.set_zlim3d(0, 12)


        # plt.figure()
        # plt.title('x-pos')
        # plt.plot(traj[0,:])
        # plt.figure()
        # plt.title('y-pos')
        # plt.plot(traj[1,:])
        # plt.figure()
        # plt.title('x-vel')
        # plt.plot(traj[3,:])
        # plt.figure()
        # plt.title('y-vel')
        # plt.plot(traj[4,:])
        # plt.figure()
        # plt.title('x-acc')
        # plt.plot(traj[6,:])
        # plt.figure()
        # plt.title('y-acc')
        # plt.plot(traj[7,:])
        
        plt.show()


    def lazy_states_contraction(self, best_path):
        #lazy states contraction
        curr_idx = 0
        mid_idx = 1
        next_idx = 2
        while next_idx < len(best_path):
            node1 = best_path[curr_idx]
            node2 = best_path[next_idx]
            if self.check_collision(node1['state'][0:3],node2['state'][0:3]):
                curr_idx += 1
                mid_idx = curr_idx + 1
                next_idx = curr_idx + 2
                continue
            
            _, X = steer(node2['state'],node1['state'], self.dt_des, self.steer_order)
            
            if self.check_path_collision( X):
                curr_idx += 1
                mid_idx = curr_idx + 1
                next_idx = curr_idx + 2
                continue
            
            best_path.pop(mid_idx)
            best_path[curr_idx]['path'] = X

        #total distance
        s_t = 0
        for node in best_path:
            if node['state'][0]==0:
                continue
            X = node['path']

            x = X[0,:]
            y = X[1,:]
            z = X[2,:]

            N = X.shape[1]

            dx = x[1:N] - x[0:N-1]
            dy = y[1:N] - y[0:N-1]
            dz = z[1:N] - z[0:N-1]
            ds2 = dx**2 + dy**2 + dz**2
            ds = np.sqrt(ds2)
            s = np.sum(ds)
            s_t += s

        return best_path, s_t


    def min_snap_trajectory(self,best_path):

        traj = None
        i = 0
        solution_found = True
        while i < (len(best_path)-1):

            #if node[i] has no path
            if best_path[i]['free']:
                state_final = best_path[i]['state'] 
                int_points = [] 
                int_nodes = []
                for j in range(i+1, len(best_path)):
                    if best_path[j]['free']:
                        if j+1 == len(best_path):
                            state_init = best_path[j]['state'] 
                            break
                        int_points.append(best_path[j]['state'][0:3])
                        int_nodes.append(best_path[j])
                        continue
                    else:
                        state_init = best_path[j]['state'] 
                        break
                n_int = len(int_points)
                #s, X = min_snap_stable(state_init, state_final, int_points, self.dt_des)
                s, X = min_snap_trajectory(state_init, state_final, int_points, self.dt_des)

                #Check min snap trajectory collision
                div = 2
                while self.check_path_collision(X) and div < 10:

                    #add intermediate points
                    int_points = []
                    N = len(int_nodes)
                    for j in range(N-1,-1,-1):
                        for k in range(1,div):
                            factor = k/float(div)
                            int_idx = int(int_nodes[j]['path'].shape[1] * factor)
                            p_mid = int_nodes[j]['path'][0:3,int_idx]
                            int_points.append(p_mid)
                        int_points.append(int_nodes[j]['state'][0:3])

                    for k in range(1,div):
                        factor = k/float(div)
                        int_idx = int(best_path[i]['path'].shape[1] * factor)
                        p_mid = best_path[i]['path'][0:3,int_idx]
                        int_points.append(p_mid)
                        
                    print('iterating')
                    print(int_points)
                    #recalculate path using intermediate points
                    #s, X = min_snap_stable(state_init, state_final, int_points, self.dt_des)
                    s, X = min_snap_trajectory(state_init, state_final, int_points, self.dt_des)
                    div += 1

                if div == 10:
                    solution_found = False

                print('done')


                i+=1 + n_int
                if traj is None:
                    traj = X
                else:
                    traj = np.concatenate((X, traj), axis=1)
            
            else:
                X =  best_path[i]['path'] 
                i+=1
                if traj is None:
                    traj = X
                else:
                    traj = np.concatenate((X, traj), axis=1)
        #return traj path
        if traj is not None:
            N = traj.shape[1]
            x  = traj[0,:]
            y  = traj[1,:]
            z  = traj[2,:]

            dx = x[1:N] - x[0:N-1]
            dy = y[1:N] - y[0:N-1]
            dz = z[1:N] - z[0:N-1]
            
            ds2 = dx**2 + dy**2 + dz**2
            ds = np.sqrt(ds2)
            s = np.sum(ds)
        else:
            s = None
        return traj, solution_found, s

    def publish_path(self, best_path):
        rate = rospy.Rate(10)

        for node in best_path[-2::-1]:
            N = node['path'].shape[1]

            for i in range(N):
                rrt_point = node['path'][:,i].flatten()
                self.rrt_msg.flat_state = list(rrt_point)
                self.pub_rrt.publish(self.rrt_msg)
                rate.sleep()

        print('RRT_star_done')
        comp_rrt = 1
        self.pub_send_comp.publish(comp_rrt)


if __name__ == '__main__':
    #rospy.init_node('rrt_star_planner')coll
    times = []
    found_sol = []
    lengths = []
    for i in range(100):
        start = time.time()
        rrt = RRT_star_planner()
        best_path = rrt.rrt_plan()
        best_path, length = rrt.lazy_states_contraction(best_path)
        #traj, solution_found, length = rrt.min_snap_trajectory(best_path)
        if len(best_path)>0:
            lengths.append(length)

        # if (traj is None)  or (solution_found is False) :
        #     solution_found = False
        # else:
        #     lengths.append(length)

        end = time.time()
        times.append(end-start)
        #found_sol.append(solution_found)
        print('Total_time = ', end - start)
        print('Total_length = ', length)
        #print('Found solution = ', solution_found)
    
    total_found = 0
    for i in range(len(found_sol)):
        if found_sol[i]:
            total_found +=1
    print('percentage found = ', total_found/100.0)
    times = np.array(times)
    lengths = np.array(lengths)
    print('avg_time = ', np.average(times))
    print('time_std_dev', np.std(times))
    print('avg_length = ', np.average(lengths))
    print('length_std_dev', np.std(lengths))
    rrt.plot_path(best_path, traj)


    #rrt.publish_path(best_path)
    #rospy.spin()

