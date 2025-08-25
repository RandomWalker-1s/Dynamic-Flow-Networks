import numpy as np

#===============================================================
class Cell:
    __slots__ = (
        'ID', 'param', 'net', 'node', 
        'state_name_list', 'state', 'state_output', 
        'co_state_name_list', 'co_state', 'co_state_output',
        'initial_condition', 'flow_dict'
    )

    def __init__(self, ID, max_density=np.inf, max_speed=np.inf, state_len=1, cell_len=1, model_order=1):
        
        # Set cell ID. 
        self.ID = ID

        # Set cell parameters. 
        self.param = {
            'min_density': 0,
            'max_density': max_density,
            'min_speed': 0,
            'max_speed': max_speed,
            'state_len': state_len,
            'cell_len': cell_len,
            'model_order': model_order
        }

        # Set network. 
        self.net = None

        # Set head and tail nodes. 
        self.node = {'head': None, 'tail': None}

        # State. 
        self.state_name_list = []
        self.state = {}
        self.state_output = {}

        # Co-state. 
        self.co_state_name_list = []
        self.co_state = {}
        self.co_state_output = {}

        # Initial condition. 
        self.initial_condition = {}

        # Initialize flow dictionary.
        self.flow_dict = {}

    def set_initial_condition(self, initial_condition):
        if initial_condition is None:
            return 
        
        # Check initial state. 
        if set(initial_condition.keys()) != set(self.state_name_list):
            raise ValueError('Unmatched state names.')
        
        # Check length for each initial state. 
        for k, v in initial_condition.items():
            if len(np.atleast_1d(v)) != self.param['state_len']:
                raise ValueError(f'Wrong length for {k}.')
            
        # Set initial condition.
        for k, v in initial_condition.items():
            self.initial_condition[k] = np.atleast_1d(v)

    def set_flow(self, name, obj):
        if obj is not None:
            self.flow_dict[name] = obj
            self.flow_dict[name].set_cell(self)
        else:
            self.flow_dict[name] = None

    def initialize(self):
        # Initialize state. 
        for k in self.state_name_list:
            self.state[k] = self.initial_condition[k]

        # Initialize co-state. 
        for k in self.co_state_name_list:
            self.co_state[k] = np.full(self.param['state_len'], np.nan)

        # Initialize state output. 
        for k in self.state_name_list:
            self.state_output[k] = np.full([self.net.num_step+1, self.param['state_len']], np.nan)
            self.state_output[k][0, :] = self.state[k]

        # Initialize co-state output. 
        for k in self.co_state_name_list:
            self.co_state_output[k] = np.full([self.net.num_step, self.param['state_len']], np.nan)

        # Initialize flow function.  
        for flow in self.flow_dict.values():
            flow.initialize()

    def save_output(self):
        for k in self.state_name_list:
            self.state_output[k][self.net.step+1, :] = self.state[k]

        for k in self.co_state_name_list:
            self.co_state_output[k][self.net.step, :] = self.co_state[k]

    def update_speed(self):
        self.co_state['speed'] = self.calculate_speed(self.state['density'], self.co_state['outflow'])

    def update_density(self):
        self.state['density'] = self.calculate_density(self.state['density'], self.co_state['inflow'], self.co_state['outflow'])

    def calculate_density(self, density, inflow, outflow):
        density = density + (inflow - outflow) * self.net.time_step_size / self.param['cell_len']
        return np.clip(density, self.param['min_density'], self.param['max_density'])
    
    def calculate_speed(self, density, outflow):
        speed = np.divide(outflow, density, out=np.full_like(outflow, self.param['max_speed'], dtype=float), where=(density!=0))
        return np.clip(speed, self.param['min_speed'], self.param['max_speed'])


#===============================================================

class Link(Cell):

    def __init__(self, ID, max_density=np.inf, max_speed=np.inf, state_len=1, cell_len=1, model_order=1,  
                 initial_condition=None, receiving=None, sending=None):
                
        super().__init__(ID, max_density, max_speed, state_len, cell_len, model_order)

        self.set_state_name_list()

        self.set_co_state_name_list()

        self.set_initial_condition(initial_condition)

        self.set_flow('receiving', receiving)

        self.set_flow('sending', sending)


    def set_state_name_list(self):
        self.state_name_list = ['density']

    def set_co_state_name_list(self):
        self.co_state_name_list = ['speed', 'receiving', 'sending', 'inflow', 'outflow']

    def update_receiving(self):
        self.flow_dict['receiving'].update_state()
        self.co_state['receiving'] = self.flow_dict['receiving'].get_flow_value()

    def update_sending(self):
        self.flow_dict['sending'].update_state()
        self.co_state['sending'] = self.flow_dict['sending'].get_flow_value()


class Source(Cell):
    
    def __init__(self, ID, max_density=np.inf, max_speed=np.inf,  state_len=1, cell_len=1, model_order=1,
                 initial_condition=None, boundary_inflow=None, sending=None):
        
        super().__init__(ID, max_density, max_speed, state_len, cell_len, model_order)

        self.set_state_name_list()

        self.set_co_state_name_list()

        self.set_initial_condition(initial_condition)

        self.set_flow('boundary_inflow', boundary_inflow)

        self.set_flow('sending', sending)
    

    def set_state_name_list(self):
        self.state_name_list = ['density']

    def set_co_state_name_list(self):
        self.co_state_name_list = ['speed', 'sending', 'inflow', 'outflow']

    def update_sending(self):
        self.flow_dict['sending'].update_state()
        self.co_state['sending'] = self.flow_dict['sending'].get_flow_value()

    def update_boundary_inflow(self):
        self.flow_dict['boundary_inflow'].update_state()
        self.co_state['inflow'] = self.flow_dict['boundary_inflow'].get_flow_value()



class Sink(Cell):

    def __init__(self, ID, max_density=np.inf, max_speed=np.inf, state_len=1, cell_len=1, model_order=1,
                 initial_condition=None, receiving=None, boundary_outflow=None):

        super().__init__(ID, max_density, max_speed, state_len, cell_len, model_order)

        self.set_state_name_list()

        self.set_co_state_name_list()

        self.set_initial_condition(initial_condition)

        self.set_flow('receiving', receiving)
        
        self.set_flow('boundary_outflow', boundary_outflow)


    def set_state_name_list(self):
        self.state_name_list = ['density']

    def set_co_state_name_list(self):
        self.co_state_name_list = ['speed', 'receiving', 'inflow', 'outflow']

    def update_receiving(self):
        self.flow_dict['receiving'].update_state()
        self.co_state['receiving'] = self.flow_dict['receiving'].get_flow_value()

    def update_boundary_outflow(self):
        self.flow_dict['boundary_outflow'].update_state()
        self.co_state['outflow'] = self.flow_dict['boundary_outflow'].get_flow_value()



if __name__ == '__main__':
    pass