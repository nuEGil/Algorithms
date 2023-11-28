'''
Implementing this thing
https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms

Its a technique that finds paths through graphs using artificial ant
agents. Each ant is trying to find the shortes path on a weighted graph. 
So applying this means you have to convert problems to weighted graph problems.

In this case, lets use a known challenge function, but lets make it a continuous function 
for funsies. 

2 notes. This particular implementation is a gradient based optimization. 
So adding mommentum and batches can help here. 


TODO:
    add pheremone updating method. 
    clean up the plotting function.
    add other cost functions like rosenbrock.

'''
import numpy as np
import matplotlib.pyplot as plt

def eggholder(x1, x2):
    # this is a challenge function for optimization algorithms - known global min is 
    # f(x*) = -959.6407 @ x* = (512, 404.2319)
    # set up an x,y space
    # this function has a ton of local minima. makes it difficult for gradient based techniques to solve this
    
    fx = (-(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1/2 + 47)))) - (x1 * np.sin(np.sqrt(np.abs(x1-(x2 +47))))) 
    return fx

def plot_function():
    
    # generate meshgrid so we can evaluate many point pairs
    axis_ = np.linspace(-600, 600, 1000)
    x2, x1  = np.meshgrid(axis_, axis_)
    z = eggholder(x1, x2)
    # make an initial pheremone map
    pheremone_map = (50*np.random.rand(100,) -25) + np.linspace(0,600, 100)


    print('sampling minimum, ', np.min(z))

    # run the simulation for multiple ants
    ant_paths = []
    for jj in range(10):
        # make an Ant Agent to put on this thing
        starting_point = (100*np.random.rand(2,)-50)
        A = AntAgent(eggholder, init_vec=starting_point)
        for ii in range(10000):
            A.EdgeSelection() # 
        A_paths = np.array(A.path_)

        # append the agent path
        ant_paths.append(A_paths)
        
        print('A_paths.shape ', A_paths.shape)
        end_val = eggholder(A_paths[-1][0], A_paths[-1][1])
        print('end location {},{} and val {}'.format(A_paths[-1][0], A_paths[-1][1] , end_val))
    
    plt.figure(figsize = (10,10))
    plt.imshow(z, origin = 'lower', cmap = 'gray', extent=(-600,600,-600,600))
    
    for ia, a in enumerate(ant_paths):
        plt.plot(a[:,0], a[:,1], label = 'ant-{}'.format(ia))

    plt.legend()
    plt.show()

class AntAgent():
    def __init__(self, cost_func, init_vec=np.array([0.1, 0]), bounds = [-600,600,-600,600]):
        # reset this as a dictionary
        self.x = init_vec # initial vector is an input argument
        self.starting_point = init_vec
        
        self.path_ = [init_vec]
        
        # since I want to optimize a continuous function, I need to swap out 
        # edges with just a number of directions.
        self.N_directions = int(5*np.random.rand())+1 # number of directions this ant is capable of considering. 

        # parameters to control influence of edge selection 
        self.alpha = 0.5 # set alpha >= 0
        self.beta = 0.5 # set beta >= 1
        # how far can this ant go in 1 step?
        self.step = 25*np.random.rand() # depends on the ant's apptitude. 

        # functions are objects in python, so... 
        self.cost_func = cost_func
        #set bounds
        self.bounds = bounds

    def EdgeSelection(self, pheremone_map=np.ones((10, 10))):
        # first generate a set of directions
        angles = 2*np.pi*np.random.rand(self.N_directions) # 0 to 2pi radians 
        
        # store the new states
        new_states = np.zeros((self.N_directions, 2))
        attractiveness = np.zeros((self.N_directions,))
        
        for thi, th in enumerate(angles):
            # use the rotation matrix to rotate the vector should be Nx2x2
            R_ = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            v = np.dot(R_, self.x) # apply the rotation 
            # print(v)
            # change the length of the vector to the step size
            v = self.x + (self.step * v / np.sqrt(np.sum(v ** 2)) )
            new_states[thi, :] = 0 + v  # store new states

            # compute attractiveness -- hehe it's a little slope. 
            mag_vx = np.sqrt(np.sum((v - self.x) ** 2))
            # this is giving the highest slope..... multiply by -1 to get lowest 
            attractiveness[thi] = (self.cost_func(v[0], v[1]) - self.cost_func(self.x[0], self.x[1])) / (mag_vx)

        ns = new_states[np.argmin(attractiveness)]
        
        boundary = [ns[0]<self.bounds[0], 
                    ns[0]>self.bounds[1], 
                    ns[1]<self.bounds[2], 
                    ns[1]>self.bounds[3],] 
        boundary = any(boundary)
        if boundary:
            # go back to init?
            self.x = 0 + self.starting_point
        else:
            self.x = 0 +ns
        # get the minimum 
        self.path_.append(ns)
        # print('attractivenes', attractiveness)
        # # Probability of moving to a given state is based on attractivenss, and the trail level
        # attractiveness_xy = self.cost_func(x1, x2)
        # return new_states
        
    def PheremoneUpdate():
        print()
        # need to place a ball of pheremone - box to make this easy
        #... so rather than updating a weight on a graph, 
        # we need to set a range of xy, to place pheremone


if __name__ == '__main__':
    plot_function() # show the eggholder function 
