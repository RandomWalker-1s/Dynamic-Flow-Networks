import numpy as np
from . import utils


class Flow(utils.NetUnit):
    def __init__(self, cell=None, is_saved=True):

        net = None if cell is None else cell.net

        super().__init__(net, is_saved)

        self.hook_up_to_cell(cell)

    
    def hook_up_to_cell(self, cell):
        self.cell = cell


    def initialize_co_state(self):
        # flow: (state_len, ). 
        self.co_state['flow'] = np.full(self.net.param['state_len'], np.nan)

    
    def compute_flow(self):
        return np.full(self.net.param['state_len'], np.nan)
    

    def _compute_flow(self):
        return self.compute_flow()


    def iterate(self):
        self.co_state['flow'] = self._compute_flow()


    def get_flow(self):
        return self.co_state['flow']

    

#------------------------Boundary inflow & outflow functions.------------------------------

class BoundaryInflow(Flow):
    def __init__(self, boundary_inflow, is_bc_constant=True, cell=None, is_saved=True):

        super().__init__(cell, is_saved)

        # boundary_inflow: if constant, (1, ) or (state_len, ), if not constant, (1, num_step) or (state_len, num_step)
        self.param['boundary_inflow'] = np.atleast_1d(boundary_inflow) if is_bc_constant else np.atleast_2d(boundary_inflow)

        # is_constant: bool.
        self.param['is_bc_constant'] = is_bc_constant
    

    def compute_flow(self, state_len, step=None):
        # boundary_inflow: : (1, ) or (state_len, ).
        if self.param['is_bc_constant']:
            boundary_inflow = self.param['boundary_inflow']
        else:
            boundary_inflow = self.param['boundary_inflow'][:, step]

        return boundary_inflow * np.ones(state_len)
    

    def _compute_flow(self):
        return self.compute_flow(self.net.param['state_len'], self.net.step)


class BoundaryOutflow(Flow):
    def __init__(self, boundary_speed, boundary_capacity, is_bc_constant=True, cell=None, is_saved=True):

        super().__init__(cell, is_saved)

        # Check if the lengths are consistent.  
        if (not is_bc_constant) and (len(boundary_speed) != len(boundary_capacity)):
            raise ValueError('Time-varying boundary speed and boundary capacity must have the same length.')

        # boundary_speed: if constant, (1, ) or (state_len, ), if not constant, (1, num_step) or (state_len, num_step)
        self.param['boundary_speed'] = np.atleast_1d(boundary_speed) if is_bc_constant else np.atleast_2d(boundary_speed)
          
        # boundary_capacity: if constant, (1, ) or (state_len, ), if not constant, (1, num_step) or (state_len, num_step)
        self.param['boundary_capacity'] = np.atleast_1d(boundary_capacity) if is_bc_constant else np.atleast_2d(boundary_capacity)

        # is_constant: bool.
        self.param['is_bc_constant'] = is_bc_constant


    def compute_flow(self, density, step=None):
        # boundary_speed, boundary_capacity: (1, ) or (state_len, ).
        if self.param['is_bc_constant']:
            boundary_speed, boundary_capacity = self.param['boundary_speed'], self.param['boundary_capacity']
        else:
            boundary_speed, boundary_capacity = self.param['boundary_speed'][:, step], self.param['boundary_capacity'][:, step]

        return np.minimum(boundary_speed * density, boundary_capacity)


    def _compute_flow(self):
        return self.compute_flow(self.cell.state['density'], self.net.step)


#----------------------------Sending flow functions---------------------------------

class BufferSendingFlow(Flow):
    def __init__(self, demand, is_demand_constant=True, capacity=np.inf, ignore_queue=False, cell=None, is_saved=True):

        super().__init__(cell, is_saved)

        # demand: if constant, (1, ) or (state_len, ), if not constant, (1, num_step) or (state_len, num_step)
        self.param['demand'] = np.atleast_1d(demand) if is_demand_constant else np.atleast_2d(demand)

        # capacity: (1, ) or (state_len, ).
        self.param['capacity'] = np.atleast_1d(capacity)

        # is_constant: bool. 
        self.param['is_demand_constant'] = is_demand_constant

        # ignore_queue: bool. 
        self.param['ignore_queue'] = ignore_queue


    def compute_flow(self, state_len=None, step=None, queue_len=None):
        # _demand: (1, ) or (state_len, ).
        if self.param['is_demand_constant']:
            _demand = self.param['demand']
        else:
            _demand = self.param['demand'][:, step]
        
        if self.param['ignore_queue']:
            return _demand * np.ones(state_len)
        else:
            # queue_len: (state_len, ).
            return np.minimum(_demand + queue_len / self.net.param['time_step_size'], self.param['capacity'])


    def _compute_flow(self):
        # flow: (state_len, ).
        queue_len = self.cell.state['density'] * self.cell.param['cell_len']
        return self.compute_flow(self.net.param['state_len'], self.cell.net.step, queue_len)



class PiecewiseLinearSendingFlow(Flow):
    def __init__(self, free_flow_speed, capacity, cell=None, is_saved=True):

        super().__init__(cell, is_saved)

        # free_flow_speed: (1, ) or (state_len, ).
        self.param['free_flow_speed'] = np.atleast_1d(free_flow_speed)

        # capacity: (1, ) or (state_len, ).
        self.param['capacity'] = np.atleast_1d(capacity)


    def compute_flow(self, density):
        return np.minimum(self.param['free_flow_speed'] * density, self.param['capacity'])


    def _compute_flow(self):
        # flow: (state_len, ).
        return self.compute_flow(self.cell.state['density'])
        


class CapacityDropPiecewiseLinearSendingFlow(Flow):
    def __init__(self, free_flow_speed, capacity, capacity_drop_density_threshold=None, capacity_dropped=None, cell=None, is_saved=True):

        super().__init__(cell, is_saved)

        # free_flow_speed: (1, ) or (state_len, ).
        self.param['free_flow_speed'] = np.atleast_1d(free_flow_speed)

        # capacity: (1, ) or (state_len, ).
        self.param['capacity'] = np.atleast_1d(capacity)

        # density_threshold: (1, ) or (state_len, ).
        self.param['capacity_drop_density_threshold'] = np.atleast_1d(capacity_drop_density_threshold)

        # capacity_drop_rate: (1, ) or (state_len, ).
        self.param['capacity_dropped'] = np.atleast_1d(capacity_dropped)


    def initialize_co_state(self):
        self.co_state['flow'] = np.full(self.net.param['state_len'], np.nan)
        self.co_state['real_capacity'] = np.full(self.net.param['state_len'], np.nan)


    def compute_flow(self, density):
        real_capacity = np.where(density >  self.param['capacity_drop_density_threshold'], self.param['capacity_dropped'], self.param['capacity'])
        return np.minimum(self.param['free_flow_speed'] * density, real_capacity), real_capacity


    def _compute_flow(self):        
        # flow: (state_len, ).
        return self.compute_flow(self.cell.state['density']) 
        

    def iterate(self):
        flow, real_capacity = self._compute_flow()
        self.co_state['flow'] = flow
        self.co_state['real_capacity'] = real_capacity



class MarkovianPiecewiseLinearSendingFlow(Flow):
    def __init__(self, mode_list, free_flow_speed, capacity, prob_matrix, initial_mode, has_multi_regime=False, regime_bound_list=None, cell=None, is_saved=True):

        super().__init__(cell, is_saved)

        # mode_list.
        self.param['mode_list'] = mode_list

        # boundary_speed: (num_mode, ).
        self.param['free_flow_speed'] = np.atleast_1d(free_flow_speed)

        # boundary_capacity: (num_mode, ).
        self.param['capacity'] = np.atleast_1d(capacity)

        # has_multi_regime: bool. 
        self.param['has_multi_regime'] = has_multi_regime

        # regime_bound_list.
        self.param['regime_bound_list'] = regime_bound_list

        # prob_matrix: (num_mode, num_mode) or (num_regime, num_mode, num_mode)
        self.param['prob_matrix'] = np.atleast_2d(prob_matrix) if not has_multi_regime else np.atleast_3d(prob_matrix)

        # initial_mode: (state_len, ).
        self.initial_condition['mode'] = np.atleast_1d(initial_mode) * np.ones(self.net.param['state_len'])


    def initialize_state(self):
        self.state['real_time_mode'] = self.initial_condition['mode']


    def initialize_co_state(self):
        self.co_state['flow'] = np.full(self.net.param['state_len'], np.nan)

        if self.param['has_multi_regime']:
            self.co_state['real_time_regime'] = np.full(self.net.param['state_len'], np.nan)


    def compute_flow(self, density, mode):
        return np.minimum(self.param['free_flow_speed'][mode] * density, self.param['capacity'][mode])


    def _compute_flow(self):
        # flow: (state_len, ).
        return self.compute_flow(self.cell.state['density'], self.state['real_time_mode']) 


    def sample_next_mode(self):
        if not self.param['has_multi_regime']:
            next_mode = [np.random.choice(self.param['modes'], p=self.param['prob_matrix'][m, :]) for m in self.state['real_time_mode']]

        else:
            next_mode = [np.random.choice(self.param['modes'], p=self.param['prob_matrix'][r, m, :]) for r, m in zip(self.co_state['real_time_regime'], self.state['real_time_mode'])]

        # next_mode: (state_len, ).
        return np.atleast_1d(next_mode)


    def find_regime(self, density):
        # regime_idx: (state_len, ).
        regime_idx = np.searchsorted(self.param['regime_bound_list'], density, side='right')
        return regime_idx 


    def iterate(self):
        self.co_state['flow'] = self._compute_flow()

        if self.param['has_multi_regime']:
            self.co_state['real_time_regime'] = self.find_regime(self.cell.state['density'])
        
        self.state['real_time_mode'] = self.sample_next_mode()


#----------------------------Receiving flow functions---------------------------------
 
class UnboundedReceivingFlow(Flow):
    def __init__(self, cell=None, is_saved=True):
        super().__init__(cell, is_saved)

    def compute_flow(self, state_len):
        return np.inf * np.ones(state_len)

    def _compute_flow(self):
        # flow: (state_len, ).
        return self.compute_flow(self.net.param['state_len'])


class PiecewiseLinearReceivingFlow(Flow):
    def __init__(self, congestion_wave_speed, max_density, capacity=np.inf, cell=None, is_saved=True):
        
        super().__init__(cell, is_saved)

        # congestion_wave_speed: (1, ) or (state_len, ).
        self.param['congestion_wave_speed'] = np.atleast_1d(congestion_wave_speed)

        # max_density: (1, ) or (state_len, ).
        self.param['max_density'] = np.atleast_1d(max_density)

        # capacity: (1, ) or (state_len, ).
        self.param['capacity'] = np.atleast_1d(capacity)


    def compute_flow(self, density):
        return np.minimum(self.param['congestion_wave_speed'] * (self.param['max_density'] - density), self.param['capacity'])


    def _compute_flow(self):
        # flow: (state_len, ).
        return self.compute_flow(self.cell.state['density'])
        

class LookAheadPiecewiseLinearReceivingFlow(Flow):
    def __init__(self, congestion_wave_speed, max_density, capacity, 
                 cell=None, cell_upstream=None, look_ahead_density_threshold=None, 
                 look_ahead_congestion_wave_speed=None, look_ahead_max_density=None, look_ahead_capacity=None, is_saved=True):
        
        super().__init__(cell, is_saved)

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


    def compute_flow(self, density, density_upstream):

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


    def _compute_flow(self):
        # flow: (state_len, ).
        return self.compute_flow(self.cell.state['density'], self.cell_upstream.state['density'])
        

if __name__ == '__main__':
    pass