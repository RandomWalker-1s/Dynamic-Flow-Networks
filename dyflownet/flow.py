import numpy as np

class Flow:
    def __init__(self, cell=None):

        # Parameter dictionary. 
        self.param = {}

        # State dictionary. 
        self.state = {} 

        # Bound to cell.
        self.set_cell(cell)

    def set_cell(self, cell):
        if cell is not None:
            self.cell = cell
            self.param['state_len'] = cell.param['state_len']
        else:
            self.cell = None
            self.param['state_len'] = np.nan

    def initialize(self):
        # flow: (state_len, ).
        self.state['flow'] = np.full(self.param['state_len'], np.nan) 

    def get_flow(self):
        return self.state['flow']

    def calculate_flow(self):
        pass

    def update_state(self):
        pass


#------------------------Boundary inflow & outflow functions.------------------------------

class BoundaryInflow(Flow):
    def __init__(self, boundary_inflow, is_bc_constant=True, cell=None):

        super().__init__(cell=cell)

        # boundary_inflow: if constant, (1, ) or (state_len, ), if not constant, (1, num_step) or (state_len, num_step)
        self.param['boundary_inflow'] = np.atleast_1d(boundary_inflow) if is_bc_constant else np.atleast_2d(boundary_inflow)

        # is_constant: bool.
        self.param['is_bc_constant'] = is_bc_constant

    def calculate_flow(self, step=None):
        # _boundary_inflow: : (1, ) or (state_len, ).
        if self.param['is_bc_constant']:
            _boundary_inflow = self.param['boundary_inflow']
        else:
            _boundary_inflow = self.param['boundary_inflow'][:, step]

        return _boundary_inflow

    def update_state(self):
        # flow: (state_len, ).
        self.state['flow'] = self.calculate_flow(self.cell.net.step) * np.ones(self.param['state_len'])


class BoundaryOutflow(Flow):
    def __init__(self, boundary_speed, boundary_capacity, is_bc_constant=True, cell=None):

        super().__init__(cell=cell)

        # Check if the lengths are consistent.  
        if (not is_bc_constant) and (len(boundary_speed) != len(boundary_capacity)):
            raise ValueError('Time-varying boundary speed and boundary capacity must have the same length.')

        # boundary_speed: if constant, (1, ) or (state_len, ), if not constant, (1, num_step) or (state_len, num_step)
        self.param['boundary_speed'] = np.atleast_1d(boundary_speed) if is_bc_constant else np.atleast_2d(boundary_speed)
          
        # boundary_capacity: if constant, (1, ) or (state_len, ), if not constant, (1, num_step) or (state_len, num_step)
        self.param['boundary_capacity'] = np.atleast_1d(boundary_capacity) if is_bc_constant else np.atleast_2d(boundary_capacity)

        # is_constant: bool.
        self.param['is_bc_constant'] = is_bc_constant

    def calculate_flow(self, density, step=None):
        # _boundary_speed, _boundary_capacity: (1, ) or (state_len, ).
        if self.param['is_bc_constant']:
            _boundary_speed, _boundary_capacity = self.param['boundary_speed'], self.param['boundary_capacity']
        else:
            _boundary_speed, _boundary_capacity = self.param['boundary_speed'][:, step], self.param['boundary_capacity'][:, step]

        return np.minimum(_boundary_speed * density, _boundary_capacity)

    def update_state(self):
        # flow: (state_len, ).
        self.state['flow'] = self.calculate_flow(self.cell.state['density'], self.cell.net.step)


#----------------------------Sending flow functions---------------------------------

class BufferSendingFlow(Flow):
    def __init__(self, demand, is_demand_constant=True, capacity=np.inf, ignore_queue=False, cell=None):

        super().__init__(cell=cell)

        # demand: if constant, (1, ) or (state_len, ), if not constant, (1, num_step) or (state_len, num_step)
        self.param['demand'] = np.atleast_1d(demand) if is_demand_constant else np.atleast_2d(demand)

        # capacity: (1, ) or (state_len, ).
        self.param['capacity'] = np.atleast_1d(capacity)

        # is_constant: bool. 
        self.param['is_demand_constant'] = is_demand_constant

        # ignore_queue: bool. 
        self.param['ignore_queue'] = ignore_queue

    def calculate_flow(self, step=None, queue_len=None):
        # _demand: (1, ) or (state_len, ).
        if self.param['is_demand_constant']:
            _demand = self.param['demand']
        else:
            _demand = self.param['demand'][:, step]
        
        if self.param['ignore_queue']:
            return _demand
        else:
            # queue_len: (state_len, ).
            return np.minimum(_demand + queue_len / self.cell.net.time_step_size, self.param['capacity'])

    def update_state(self):
        # flow: (state_len, ).
        queue_len = self.cell.state['density'] * self.cell.param['cell_len']
        self.state['flow'] = self.calculate_flow(self.cell.net.step, queue_len) * np.ones(self.param['state_len'])



class PiecewiseLinearSendingFlow(Flow):
    def __init__(self, free_flow_speed, capacity, cell=None):

        super().__init__(cell=cell)

        # free_flow_speed: (1, ) or (state_len, ).
        self.param['free_flow_speed'] = np.atleast_1d(free_flow_speed)

        # capacity: (1, ) or (state_len, ).
        self.param['capacity'] = np.atleast_1d(capacity)

    def calculate_flow(self, density):
        return np.minimum(self.param['free_flow_speed'] * density, self.param['capacity'])

    def update_state(self):
        # flow: (state_len, ).
        self.state['flow'] = self.calculate_flow(self.cell.state['density'])
        


class CapacityDropPiecewiseLinearSendingFlow(Flow):
    def __init__(self, free_flow_speed, capacity, capacity_drop_density_threshold=None, capacity_dropped=None, cell=None):

        super().__init__(cell=cell)

        # free_flow_speed: (1, ) or (state_len, ).
        self.param['free_flow_speed'] = np.atleast_1d(free_flow_speed)

        # capacity: (1, ) or (state_len, ).
        self.param['capacity'] = np.atleast_1d(capacity)

        # density_threshold: (1, ) or (state_len, ).
        self.param['capacity_drop_density_threshold'] = np.atleast_1d(capacity_drop_density_threshold)

        # capacity_drop_rate: (1, ) or (state_len, ).
        self.param['capacity_dropped'] = np.atleast_1d(capacity_dropped)


    def calculate_flow(self, density):
        real_capacity = np.where(density >  self.param['capacity_drop_density_threshold'], self.param['capacity_dropped'], self.param['capacity'])
        return np.minimum(self.param['free_flow_speed'] * density, real_capacity)


    def update_state(self):        
        # flow: (state_len, ).
        self.state['flow'] = self.calculate_flow(self.cell.state['density']) 
        


class MarkovianPiecewiseLinearSendingFlow(Flow):
    def __init__(self, mode_list, free_flow_speed, capacity, prob_matrix, initial_mode, is_multi_regime=False, regime_bound_list=None, cell=None):

        super().__init__(cell=cell)

        # mode_list.
        self.param['mode_list'] = mode_list

        # boundary_speed: (num_mode, ).
        self.param['free_flow_speed'] = np.atleast_1d(free_flow_speed)

        # boundary_capacity: (num_mode, ).
        self.param['capacity'] = np.atleast_1d(capacity)

        # is_multi_regime: bool. 
        self.param['is_multi_regime'] = is_multi_regime

        # regime_bound_list.
        self.param['regime_bound_list'] = regime_bound_list

        # prob_matrix: (num_mode, num_mode) or (num_regime, num_mode, num_mode)
        self.param['prob_matrix'] = np.atleast_2d(prob_matrix) if not is_multi_regime else np.atleast_3d(prob_matrix)

        # initial_mode: (state_len, ).
        self.param['initial_mode'] = np.atleast_1d(initial_mode) * np.ones(state_len) 


    def initialize(self):
        # flow: (state_len, ).
        self.state['flow'] = np.full(self.param['state_len'], np.nan)
        
        # real_time_mode: (state_len, ).
        self.state['real_time_mode'] = self.param['initial_mode']


    def calculate_flow(self, density, mode):
        return np.minimum(self.param['free_flow_speed'][mode] * density, self.param['capacity'][mode])


    def update_state(self):
        # flow: (state_len, ).
        self.state['flow'] = self.calculate_flow(self.cell.state['density'], self.state['real_time_mode']) 
        
        # real_time_mode: (state_len, ).
        self.state['real_time_mode'] = self.sample_next_mode()
    

    def sample_next_mode(self):
        if not self.param['is_multi_regime']:
            next_mode = [np.random.choice(self.param['modes'], p=self.param['prob_matrix'][m, :]) for m in self.state['real_time_mode']]

        else:
            real_time_regime = self.find_regime(self.cell.state['density'])
            next_mode = [np.random.choice(self.param['modes'], p=self.param['prob_matrix'][r, m, :]) for r, m in zip(real_time_regime, self.state['real_time_mode'])]

        # next_mode: (state_len, ).
        return np.atleast_1d(next_mode)


    def find_regime(self, density):
        # regime_idx: (state_len, ).
        regime_idx = np.searchsorted(self.param['regime_bound_list'], density, side='right')
        return regime_idx 


#----------------------------Receiving flow functions---------------------------------
 
class UnboundedReceivingFlow(Flow):
    def __init__(self, cell=None):
        super().__init__(cell=cell)

    def calculate_flow(self):
        return np.inf

    def update_state(self):
        # flow: (state_len, ).
        self.state['flow'] = self.calculate_flow() * np.ones(self.param['state_len'])


class PiecewiseLinearReceivingFlow(Flow):
    def __init__(self, congestion_wave_speed, max_density, capacity=np.inf, cell=None):
        
        super().__init__(cell=cell)

        # congestion_wave_speed: (1, ) or (state_len, ).
        self.param['congestion_wave_speed'] = np.atleast_1d(congestion_wave_speed)

        # max_density: (1, ) or (state_len, ).
        self.param['max_density'] = np.atleast_1d(max_density)

        # capacity: (1, ) or (state_len, ).
        self.param['capacity'] = np.atleast_1d(capacity)

    def calculate_flow(self, density):
        return np.minimum(self.param['congestion_wave_speed'] * (self.param['max_density'] - density), self.param['capacity'])

    def update_state(self):
        # flow: (state_len, ).
        self.state['flow'] = self.calculate_flow(self.cell.state['density'])
        
        

class LookAheadPiecewiseLinearReceivingFlow(Flow):
    def __init__(self, congestion_wave_speed, max_density, capacity, 
                 cell=None, cell_upstream=None, look_ahead_density_threshold=None, 
                 look_ahead_congestion_wave_speed=None, look_ahead_max_density=None, look_ahead_capacity=None):
        
        super().__init__(cell=cell)

        self.cell_upstream = cell_upstream

        # congestion_wave_speed: (1, ) or (state_len, ).
        self.param['congestion_wave_speed'] = congestion_wave_speed

        # max_density: (1, ) or (state_len, ).
        self.param['max_density'] = max_density

        # capacity: (1, ) or (state_len, ).
        self.param['capacity'] = capacity

        # look_ahead_density_threshold: (1, ) or (state_len, ).
        self.param['look_ahead_density_threshold'] = look_ahead_density_threshold

        # look_ahead_congestion_wave_speed: (1, ) or (state_len, ).
        self.param['look_ahead_congestion_wave_speed'] = look_ahead_congestion_wave_speed

        # look_ahead_max_density: (1, ) or (state_len, ).
        self.param['look_ahead_max_density'] = look_ahead_max_density

        # look_ahead_capacity: (1, ) or (state_len, ).
        self.param['look_ahead_capacity'] = look_ahead_capacity


    def calculate_flow(self, density, density_upstream):

        is_look_ahead_triggered = density_upstream <= self.param['look_ahead_density_threshold']

        real_congestion_wave_speed = np.where(
            is_look_ahead_triggered, 
            self.param['look_ahead_congestion_wave_speed'], 
            self.param['congestion_wave_speed'], 
        )

        real_max_density = np.where(
            is_look_ahead_triggered,
            self.param['look_ahead_max_density'], 
            self.param['max_density'],    
        )

        real_capacity = np.where(
            is_look_ahead_triggered, 
            self.param['look_ahead_capacity'], 
            self.param['capacity']
        )

        return np.minimum(real_congestion_wave_speed * (real_max_density - density), real_capacity)


    def update_state(self):
        # flow: (state_len, ).
        self.state['flow'] = self.calculate_flow(self.cell.state['density'], self.cell_upstream.state['density'])
        
        
        

if __name__ == '__main__':
    pass