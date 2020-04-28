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
from helper_functions import polynomial_interp, nearby_nodes, get_total_cost, truncate, min_snap_trajectory
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
        self.G = {0:{'parent':None,'state':np.array([0, 0, 2, 0, 0, 0, 0, 0, 0]), 'cost': 0, 'path':None}}
        self.state_goal = np.array([12, 3, 2, 0, 0, 0, 0, 0, 0])
        self.r_search = 2
        self.step_size = 0.5
        self.max_iter = 1000
        self.dt_des = 1/10.0

        #define search range
        self.x_range = [0,13]
        self.y_range = [-2,4]

        #initiate narrow window
        T1= np.identity(4)
        T1[0:3,3] = np.array([4,-2,2])
        win1 = Window(1,1,1,2,T1)

        T2= np.identity(4)
        T2[0:3,3] = np.array([4,3,2])
        win2 = Window(1,1,1,-2,T2)

        T3= np.identity(4)
        T3[0:3,3] = np.array([8,1,2])
        win3 = Window(1,1,1,2,T3)

        T4= np.identity(4)
        T4[0:3,3] = np.array([8,3,2])
        win4 = Window(1,1,1,-2,T4)

        self.windows = [win1, win2, win3, win4]

        #define circular obstacles
        self.obs_rad = 0.5
        self.obs_locs = [np.array([2,2]), np.array([6,2]), np.array([10,2])]

        


    def check_collision(self, p_rand, p_nearest):

        #check collision with circular obstacles
        t = np.linspace(0,1,20)
        for i in range(20):
            p = p_nearest + t[i] * (p_rand - p_nearest)
            for obs_loc in self.obs_locs:
                if np.linalg.norm(p[0:2] - obs_loc) < self.obs_rad:
                    return True

        #check collision with walls
        for win in self.windows:
            if win.check_collision(p_rand, p_nearest):
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



    def check_in_window(self, p_rand):

        #check if random point lies within window range
        for win in self.windows:
            if win.check_in_window(p_rand):
                return True, win

        return False, None

        

    def rrt_plan(self):

        #Iterate to get converges state
        for it in range(1,self.max_iter):
            if it%100 == 0:
                print('iteration = ', it)

            #Random sample random point
            state_rand = np.zeros(9)
            state_rand[0] = np.random.uniform(self.x_range[0],self.x_range[1])
            state_rand[1] = np.random.uniform(self.y_range[0],self.y_range[1])
            state_rand[2] = 2 #set z to 2
            
            #check if in window
            in_win, win = self.check_in_window(state_rand[0:3])
            if in_win:
                state_rand, next_node = win.generate_parabolic_nodes(it, self.dt_des)
             
            #look for nearby nodes
            Near_nodes, Nearest_node, key_Nearest_node = nearby_nodes(state_rand,self.G,self.r_search)

            
            #truncate state_rand if not in window
            if not(in_win):
                state_rand = truncate(state_rand, Nearest_node['state'], self.step_size)

            #check collision   
            if self.check_collision(state_rand, Nearest_node['state']):
                continue

            #Wire new node
            self.G[it] = {'parent':key_Nearest_node,'state':state_rand, 'cost':np.linalg.norm(state_rand[0:3] - Nearest_node['state'][0:3]), 'path':None}  
            
            #Wire next node if in window
            if in_win:
                self.G[-it] = next_node

            #rewire close nodes to reduce cost
            for key in Near_nodes.keys():
                node = self.G[key]

                #do not re-wire nodes with parabolic profile
                if node['path'] is not None:
                    continue
                
                if self.check_collision(state_rand, node['state']):
                    continue
                    
                stage_cost = np.linalg.norm(node['state'][0:3] - state_rand[0:3])
                total_cost = stage_cost + get_total_cost(self.G, it)
                
                if total_cost < node['cost']:
                    self.G[key]['parent'] = it
                    self.G[key]['cost'] = stage_cost
            

        #find best node to connect to goal
        min_cost = float('inf')
        best_node = None
        Near_nodes,Nearest_node, key_Nearest_node = nearby_nodes(self.state_goal,self.G,self.r_search)
        for key in Near_nodes.keys():
            node = self.G[key]
            
            if self.check_collision(self.state_goal, node['state']):
                    continue
                    
            stage_cost = np.linalg.norm(node['state'][0:3] - self.state_goal[0:3])
            total_cost = stage_cost + get_total_cost(self.G, key)
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_node = key
                    
        #wire goal state
        self.G['goal'] = {'parent':best_node,'state':self.state_goal, 'cost':min_cost, 'path':None}  

        #generate best path
        best_path = [self.G['goal']]
        parent = best_node
        while parent != None:
            best_path.append(self.G[parent])
            parent = self.G[parent]['parent']

        return best_path
    
    def plot_path(self, best_path, traj):

       
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
        fig, ax = plt.subplots()
        for obs in self.obs_locs:
            circle = plt.Circle((obs[0], obs[1]), self.obs_rad, color='r')
            ax.add_artist(circle)
        xl = []
        yl = []
        for i in range(len(best_path)):
            x = best_path[i]['state'][0]
            y = best_path[i]['state'][1]
            xl.append(x)
            yl.append(y)
        
        ax.plot(xl,yl)
        plt.xlim(self.x_range)
        plt.ylim(self.y_range)
        
        
        # fig, ax = plt.subplots()
        # for obs in self.obs_locs:
        #     circle = plt.Circle((obs[0], obs[1]), self.obs_rad, color='r')
        #     ax.add_artist(circle)
        ax.plot(traj[0,:], traj[1,:], 'b.')

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
            if self.check_collision(node1['state'], node2['state']):
                curr_idx += 1
                mid_idx = curr_idx + 1
                next_idx = curr_idx + 2
                continue    

            best_path.pop(mid_idx)
        return best_path

    def min_snap_trajectory(self,best_path):

        traj = None
        i = 0

        while i < (len(best_path)-1):

            #if node[i] has no path
            if best_path[i]['path'] is None:
                state_final = best_path[i]['state'] 
                int_points = [] 
                for j in range(i+1, len(best_path)):
                    if best_path[j]['path'] is None:
                        if j+1 == len(best_path):
                            state_init = best_path[j]['state'] 
                            break
                        int_points.append(best_path[j]['state'][0:3])
                        continue
                    else:
                        state_init = best_path[j]['state'] 
                        break
                n_int = len(int_points)
                s, X = min_snap_trajectory(state_init, state_final, int_points, self.dt_des)

                #Check min snap trajectory collision
                while self.check_path_collision(X):

                    #add intermediate points
                    p0 = state_init[0:3]
                    idx = 0
                    N = len(int_points)
                    for j in range(N):
                        p1 = int_points[idx]
                        p_mid = p0 + 0.5 *(p1 - p0)
                        int_points.insert(idx,p_mid)
                        p0 = p1
                        idx +=2

                    p1 = state_final[0:3]

                    p_mid = p0 + 0.5 *(p1 - p0)
                    int_points.insert(len(int_points),p_mid)
                        
                    print('iterating')
                    print(int_points)
                    #recalculate path using intermediate points
                    s, X = min_snap_trajectory(state_init, state_final, int_points, self.dt_des)
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

        return traj


    def publish_path(self, traj):
        rate = rospy.Rate(10)

        for i in range(traj.shape[1]):
            rrt_point = traj[:,i].flatten()
            self.rrt_msg.flat_state = list(rrt_point)
            self.pub_rrt.publish(self.rrt_msg)
            rate.sleep()


        print('RRT_star_done')
        comp_rrt = 1
        self.pub_send_comp.publish(comp_rrt)



if __name__ == '__main__':
    #rospy.init_node('rrt_star_planner')
    rrt = RRT_star_planner()
    start = time.time()
    best_path = rrt.rrt_plan()
    best_path = rrt.lazy_states_contraction(best_path)
    rrt.plot_path(best_path, traj=np.array([[0],[0]]))
    traj = rrt.min_snap_trajectory(best_path)
    end = time.time()
    print('Total_time = ', end - start)
    #rrt.publish_path(traj)
    rrt.plot_path(best_path, traj)
    #rospy.spin()





