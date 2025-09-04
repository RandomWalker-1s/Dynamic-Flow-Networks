import numpy as np

def safe_div(x, y, fill = np.inf):
    return np.divide(x, y, out=np.full_like(x, fill, dtype=float), where=(y != 0))


def generate_boundary_combos(*arrays):
    array_list = [np.asarray(a) for a in arrays]
    N = len(array_list)

    endpoint = [[array[0], array[-1]] for array in array_list]
    
    grid = np.meshgrid(*arrays, indexing="ij")
    unravel = np.array([k.ravel() for k in grid])
    
    is_endpoint = []
    for i in range(N):
        is_endpoint.append((unravel[i, :] == endpoint[i][0]) | (unravel[i, :] == endpoint[i][1]))
        
    return unravel[:, np.any(is_endpoint, axis=0)]



class NetUnit:
    def __init__(self, net=None, is_saved=True):

        self.hook_up_to_net(net)

        self.param = {'is_saved': is_saved}

        self.state = {}
        self.state_output = {}

        self.co_state = {}
        self.co_state_output = {}

        self.initial_condition = {}


    def hook_up_to_net(self, net):
        self.net = net


    def set_initial_condition(self, initial_condition):
        if initial_condition is None:
            return
        
        # Set initial condition.
        for k, v in initial_condition.items():
            self.initial_condition[k] = np.atleast_1d(v)
    
    
    def initialize(self):
        self.initialize_state()
        self.initialize_co_state()
        self.initialize_output()


    def initialize_state(self):
        pass


    def initialize_co_state(self):
        pass


    def initialize_output(self):
        if self.param['is_saved']:
            for k, v in self.state.items():
                self.state_output[k] = np.full([self.net.param['num_step']+1, self.net.param['state_len']], np.nan)
                self.state_output[k][0, :] = v
            
            for k in self.co_state:
                self.co_state_output[k] = np.full([self.net.param['num_step'], self.net.param['state_len']], np.nan)
            
    
    def save_output(self):
        if self.param['is_saved']:
            for k, v in self.state_output.items():
                v[self.net.step+1, :] = self.state[k]

            for k, v in self.co_state_output.items():
                v[self.net.step, :] = self.co_state[k]