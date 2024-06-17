import numpy as np

class ValueIteration:
    def __init__(self, grid_world, size, terminal_state, policy, terminal_reward, non_terminal_reward, gamma) -> None:
        self.size_ = size
        #self.grid_world_ = 5*np.ones(self.size_)
        self.grid_world_ = grid_world
        self.terminal_states_ = terminal_state
        self.policy_ = policy                            #Analogy from clock starting from 3'O clock
        self.max_iter_ = 1000
        self.threshold_ = 0.00000001
        self.terminal_reward_ = terminal_reward
        self.non_terminal_reward_ = non_terminal_reward
        self.reward_ = (non_terminal_reward*np.ones(self.size_))
        self.gamma = gamma
        for location in terminal_state:
            self.reward_[location] = terminal_reward
            self.grid_world_[location] = 0

    def pad_matrix(self,matrix):
        grid_world = matrix
        grid_world = np.concatenate((grid_world[:,0].reshape(-1,1), grid_world, grid_world[:,-1].reshape(-1,1)), axis=1)
        grid_world = np.concatenate((grid_world[0].reshape(1,-1), grid_world, grid_world[-1].reshape(1,-1)), axis = 0)
        return grid_world
    
    def value_evaluation(self):
        h, w = self.size_[0], self.size_[1]
        dummy_reward = self.pad_matrix(self.reward_)
        
        for itr in range(self.max_iter_):
            n = itr
            dummy_matrix = self.pad_matrix(self.grid_world_)
            grid_world_copy = self.grid_world_.copy()
            
            for i in range(h):
                for j in range(w):
                    if (i, j) not in  self.terminal_states_:
                        v1 = np.array([dummy_matrix[(i+1,j+2)], dummy_matrix[(i+2,j+1)], dummy_matrix[(i+1,j)], dummy_matrix[(i,j+1)]])
                        r = np.array([dummy_reward[(i+1,j+2)], dummy_reward[(i+2,j+1)], dummy_reward[(i+1,j)], dummy_reward[(i,j+1)]])
                        gamma_v1 = self.gamma*v1
                        r_plus_gv = r + gamma_v1
                        grid_world_copy[i][j] = self.policy_.T@r_plus_gv
            
            delta = np.sum(np.abs(grid_world_copy - self.grid_world_))/(h*w)
            print(self.grid_world_)
            self.grid_world_ = grid_world_copy
            if delta<self.threshold_:
                return n, grid_world_copy
                break


size = (4,4)
grid_world = -16.5*np.ones(size)
terminal_states = [(3,3), (0,0)]
policy = np.array([0.25, .25, .25, .25])
terminal_reward = -1
non_terminal_reward = -1
gamma = 1

GetVal = ValueIteration(grid_world, size,terminal_states,policy,terminal_reward,non_terminal_reward,gamma)

iter, val = GetVal.value_evaluation()
print(val)
print(iter)
