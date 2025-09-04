import numpy as np
from . import utils

# ============================== Node =====================================

class LocalController(utils.NetUnit):
    def __init__(self, min_control_input=0, max_control_input=np.inf, node=None, is_saved=True):

        net = None if node is None else node.net

        super().__init__(net, is_saved)

        self.param['min_control_input'] = min_control_input
        self.param['max_control_input'] = max_control_input

        self.hook_up_to_node(node)
        

    def hook_up_to_node(self, node):
        self.node = node
    

    def initialize_co_state(self):
        self.co_state['control_input'] = np.full(self.net.param['state_len'], np.nan)


    def compute_control_input(self):
        return


    def _compute_control_input(self):
        return


    def iterate(self):
        self.co_state['control_input'] = self._compute_control_input()


    def get_control_input(self):
        return self.co_state['control_input']



class ALINEA(LocalController):
    def __init__(self, gain, setpoint, initial_condition, min_control_input=0, max_control_input=np.inf, node=None, cell_list=None, is_saved=True):

        super().__init__(min_control_input, max_control_input, node, is_saved)

        self.param['gain'] = gain
        self.param['setpoint'] = setpoint

        self.cell_list = [] if cell_list is None else cell_list

        self.set_initial_condition(initial_condition)

    def initialize_state(self):
        if len(self.initial_condition['control_input']) == 1:
            self.state['control_input'] = self.initial_condition['control_input'] * np.ones(self.net.param['state_len'])
        elif len(self.initial_condition['control_input']) == self.net.param['state_len']:
            self.state['control_input'] = self.initial_condition['control_input']
        else:
            raise ValueError('Wrong length of initial condition.')


    def compute_control_input(self, density, last_step_control_input):
        control_input = last_step_control_input + self.param['gain'] * (self.param['setpoint'] - density)
        return np.clip(control_input, self.param['min_control_input'], self.param['max_control_input'])


    def _compute_control_input(self):
        return self.compute_control_input(self.cell_list[0].state['density'], self.state['control_input'])
         
    
    def iterate(self):
        self.state['control_input'] = self._compute_control_input()


    def get_control_input(self):
        return self.state['control_input']


class AffineController(LocalController):
    # Control law: H - K*x. 
    def __init__(self, gain, min_control_input=0, max_control_input=np.inf, node=None, cell_list=None, is_saved=True):

        super().__init__(min_control_input, max_control_input, node, is_saved)

        self.param['gain'] = gain

        self.cell_list = [] if cell_list is None else cell_list


    def initialize_co_state(self):
        self.co_state['control_input'] = np.full(self.net.param['state_len'], np.nan)
        

    def compute_control_input(self, density):
        control_input = self.param['max_control_input'] - self.param['gain'] * density
        return np.maximum(control_input, self.param['min_control_input'])
    

    def _compute_control_input(self):
        return self.compute_control_input(self.cell_list[0].state['density'])




if __name__ == '__main__':
    pass