import numpy as np
from . import utils

#===============================================================
class Cell(utils.NetUnit):

    def __init__(self, ID, max_density=np.inf, max_speed=np.inf, cell_len=1, model_order=1, net=None, is_state_saved=True, is_co_state_saved=True):
        
        super().__init__(net, is_state_saved, is_co_state_saved)

        # Set cell ID. 
        self.ID = ID

        # Set cell parameters. 
        self.param['min_density'] = 0
        self.param['max_density'] = max_density
        self.param['min_speed'] = 0
        self.param['max_speed'] = max_speed
        self.param['cell_len'] = cell_len
        self.param['model_order'] = model_order

        # Set head and tail nodes. 
        self.node = {'head': None, 'tail': None}

        # Initialize flow dictionary.
        self.flow_dict = {}


    def add_flow(self, name, flow_obj):
        self.flow_dict[name] = flow_obj
        if flow_obj is not None:
            self.flow_dict[name].hook_up_to_cell(self)
            self.flow_dict[name].hook_up_to_net(self.net)


    def initialize(self):
        super().initialize()
        self.initialize_flow()
        print(f'{self.ID} initialized.') 


    def initialize_state(self):
        if len(self.initial_condition['density']) == self.net.param['state_len']:
            self.state['density'] = self.initial_condition['density']
        else:
            raise ValueError('Wrong length of initial condition.')


    def initialize_co_state(self):
        for name in ['speed', 'inflow', 'outflow']:
            self.co_state[name] = np.full(self.net.param['state_len'], np.nan)


    def initialize_flow(self):
        for flow in self.flow_dict.values():
            flow.initialize()


    def update_speed(self):
        self.co_state['speed'] = self.compute_speed(self.state['density'], self.co_state['outflow'])


    def update_density(self):
        self.state['density'] = self.compute_density(self.state['density'], self.co_state['inflow'], self.co_state['outflow'])


    def compute_speed(self, density, outflow):
        speed = np.divide(outflow, density, out=np.full_like(outflow, self.param['max_speed'], dtype=float), where=(density!=0))
        return np.clip(speed, self.param['min_speed'], self.param['max_speed'])


    def compute_density(self, density, inflow, outflow):
        density = density + (inflow - outflow) * self.net.param['time_step_size'] / self.param['cell_len']
        return np.clip(density, self.param['min_density'], self.param['max_density'])


    def save_output(self):
        super().save_output()
        
        for flow in self.flow_dict.values():
            flow.save_output()

#===============================================================

class Link(Cell):

    def __init__(self, ID, max_density=np.inf, max_speed=np.inf, cell_len=1, model_order=1,  
                 initial_condition=None, receiving=None, sending=None, net=None, is_state_saved=True, is_co_state_saved=True):
                
        super().__init__(ID, max_density, max_speed, cell_len, model_order, net, is_state_saved, is_co_state_saved)

        self.set_initial_condition(initial_condition)

        self.add_flow('receiving', receiving)

        self.add_flow('sending', sending)


    def update_receiving(self):
        self.flow_dict['receiving'].iterate()


    def update_sending(self):
        self.flow_dict['sending'].iterate()



class Source(Cell):
    
    def __init__(self, ID, max_density=np.inf, max_speed=np.inf, cell_len=1, model_order=1,
                 initial_condition=None, boundary_inflow=None, sending=None, net=None, is_state_saved=True, is_co_state_saved=True):
        
        super().__init__(ID, max_density, max_speed, cell_len, model_order, net, is_state_saved, is_co_state_saved)

        self.set_initial_condition(initial_condition)

        self.add_flow('boundary_inflow', boundary_inflow)

        self.add_flow('sending', sending)
    

    def update_boundary_inflow(self):
        self.flow_dict['boundary_inflow'].iterate()
        self.co_state['inflow'] = self.flow_dict['boundary_inflow'].get_flow()

    def update_sending(self):
        self.flow_dict['sending'].iterate()



class Sink(Cell):

    def __init__(self, ID, max_density=np.inf, max_speed=np.inf, cell_len=1, model_order=1,
                 initial_condition=None, receiving=None, boundary_outflow=None, net=None, is_state_saved=True, is_co_state_saved=True):
        
        super().__init__(ID, max_density, max_speed, cell_len, model_order, net, is_state_saved, is_co_state_saved)

        self.set_initial_condition(initial_condition)

        self.add_flow('receiving', receiving)
        
        self.add_flow('boundary_outflow', boundary_outflow)


    def update_receiving(self):
        self.flow_dict['receiving'].iterate()


    def update_boundary_outflow(self):
        self.flow_dict['boundary_outflow'].iterate()
        self.co_state['outflow'] = self.flow_dict['boundary_outflow'].get_flow()


if __name__ == '__main__':
    pass