import numpy as np
from . import utils

# ============================== Node =====================================

class Node(utils.NetUnit):
    def __init__(self, ID, incoming_cell_list, outgoing_cell_list, controller=None, net=None, is_saved=True):

        super().__init__(net, is_saved)

        # Set node ID. 
        self.ID = ID
    
        self.incoming_cell_list = incoming_cell_list
        for c in self.incoming_cell_list:
            c.node['tail'] = self

        self.outgoing_cell_list = outgoing_cell_list
        for c in self.outgoing_cell_list:
            c.node['head'] = self

        self.param['num_incoming_cell'] = len(incoming_cell_list)
        self.param['num_outgoing_cell'] = len(outgoing_cell_list)

        self.set_controller(controller)


    def set_controller(self, controller):
        self.controller = controller
        if controller is not None:
            self.controller.hook_up_to_node(self)
            self.controller.hook_up_to_net(self.net)


    def initialize(self):
        super().initialize()

        self.initialize_controller()
    
        print(f'{self.ID} initialized.') 


    def initialize_co_state(self):
        self.co_state['inter_cell_flow'] = np.zeros([self.net.param['state_len'], self.param['num_incoming_cell'], self.param['num_outgoing_cell']])


    def initialize_controller(self):
        if self.controller is not None:
            self.controller.initialize()


    def _sending_list(self):
        return [cell.flow_dict['sending'].get_flow() for cell in self.incoming_cell_list]


    def _receiving_list(self):
        return [cell.flow_dict['receiving'].get_flow() for cell in self.outgoing_cell_list]
    

    def update_control_input(self):
        if self.controller is not None:
            self.controller.iterate()


    def compute_inter_cell_flow(self):
        # Need customization. 
        return np.zeros([self.net.param['state_len'], self.param['num_incoming_cell'], self.param['num_outgoing_cell']])
    

    def _compute_inter_cell_flow(self):
        # Need customization. 
        return self.compute_inter_cell_flow()


    def update_inter_cell_flow(self):
        self.co_state['inter_cell_flow'] = self._compute_inter_cell_flow()


    def update_cell_outflow(self):
        incoming_cell_outflow = np.sum(self.co_state['inter_cell_flow'], axis=2).T

        for cell, flow in zip(self.incoming_cell_list, incoming_cell_outflow):
            cell.co_state['outflow'] = flow


    def update_cell_inflow(self):
        outgoing_cell_inflow = np.sum(self.co_state['inter_cell_flow'], axis=1).T

        for cell, flow in zip(self.outgoing_cell_list, outgoing_cell_inflow):
            cell.co_state['inflow'] = flow

    
    def save_output(self):
        super().save_output()
        
        if self.controller is not None:
            self.controller.save_output()
            

# ============================== 1 -> 1 =====================================

class BasicJunction(Node):
    def __init__(self, ID, incoming_cell_list, outgoing_cell_list, controller=None, net=None, is_saved=True):

        super().__init__(ID, incoming_cell_list, outgoing_cell_list, controller, net, is_saved)


    def compute_inter_cell_flow(self, state_len, sending_list, receiving_list):
        inter_cell_flow = np.zeros([state_len, len(sending_list), len(receiving_list)])
        inter_cell_flow[:, 0, 0] = np.minimum(sending_list[0], receiving_list[0])
        return inter_cell_flow
    

    def _compute_inter_cell_flow(self):
        return self.compute_inter_cell_flow(self.net.param['state_len'], self._sending_list(), self._receiving_list())



# ============================== 2 -> 1 (merging) =====================================
class TwoToOneMergeJunction(Node):
    def __init__(self, ID, incoming_cell_list, outgoing_cell_list, merging_priority, controller=None, net=None, is_saved=True):
        
        super().__init__(ID, incoming_cell_list, outgoing_cell_list, controller, net, is_saved)

        # Shape of merging priority: (1, 2) or (state_len, 2).
        self.param['merging_priority'] = np.atleast_2d(merging_priority)


    def _merging_priority(self):
        return self.param['merging_priority'][:, 0], self.param['merging_priority'][:, 1] 


    def compute_inter_cell_flow(self, state_len, sending_list, receiving_list):
        sending_i_0, sending_i_1 = sending_list
        receiving_j_0 = receiving_list[0]

        p_i_0, p_i_1 = self._merging_priority() 

        is_space_enough = (sending_i_0 + sending_i_1) <= receiving_j_0
        flow_i_0_j_0 = np.where(is_space_enough, sending_i_0, np.median([sending_i_0, receiving_j_0 - sending_i_1, p_i_0 * receiving_j_0], axis=0))
        flow_i_1_j_0 = np.where(is_space_enough, sending_i_1, np.median([sending_i_1, receiving_j_0 - sending_i_0, p_i_1 * receiving_j_0], axis=0))

        inter_cell_flow = np.zeros([state_len, len(sending_list), len(receiving_list)])
        inter_cell_flow[:, 0, 0] = flow_i_0_j_0
        inter_cell_flow[:, 1, 0] = flow_i_1_j_0

        return inter_cell_flow


    def _compute_inter_cell_flow(self):
        return self.compute_inter_cell_flow(self.net.param['state_len'], self._sending_list(), self._receiving_list())



# ============================== 1 -> 2 (diverging) =====================================

class OneToTwoDivergeJunction(Node):
    def __init__(self, ID, incoming_cell_list, outgoing_cell_list, split_ratio, 
                 is_split_ratio_constant=True, is_FIFO=True, controller=None, net=None, is_saved=True):
        
        super().__init__(ID, incoming_cell_list, outgoing_cell_list, controller, net, is_saved)

        # Shape of split ratio: if constant, (1, 2) or (state_len, 2); if not constant, (1, 2, num_step) or (state_len, 2, num_step). 
        self.param['split_ratio'] = np.atleast_2d(split_ratio) if is_split_ratio_constant else  np.atleast_3d(split_ratio)
        self.param['is_split_ratio_constant'] = is_split_ratio_constant
        self.param['is_FIFO'] = is_FIFO


    def _split_ratio(self, step=None):
        if self.param['is_split_ratio_constant']:
            split_j_0, split_j_1 = self.param['split_ratio'][:, 0], self.param['split_ratio'][:, 1]
        else:
            split_j_0, split_j_1 = self.param['split_ratio'][:, 0, step], self.param['split_ratio'][:, 1, step]
        return split_j_0, split_j_1


    def compute_inter_cell_flow(self, state_len, sending_list, receiving_list, step=None):
        sending_i_0  = sending_list[0]
        receiving_j_0, receiving_j_1 = receiving_list

        split_j_0, split_j_1 = self._split_ratio(step)

        if not self.param['is_FIFO']:
            flow_i_0_j_0 = np.minimum(split_j_0 * sending_i_0, receiving_j_0)
            flow_i_0_j_1 = np.minimum(split_j_1 * sending_i_0, receiving_j_1)
        else:
            total_flow = np.minimum.reduce((sending_i_0, utils.safe_div(receiving_j_0, split_j_0), utils.safe_div(receiving_j_1, split_j_1)))
            flow_i_0_j_0 = split_j_0 * total_flow
            flow_i_0_j_1 = split_j_1 * total_flow

        inter_cell_flow = np.zeros([state_len, len(sending_list), len(receiving_list)])
        inter_cell_flow[:, 0, 0] = flow_i_0_j_0
        inter_cell_flow[:, 0, 1] = flow_i_0_j_1

        return inter_cell_flow


    def _compute_inter_cell_flow(self):
        return self.compute_inter_cell_flow(self.net.param['state_len'], self._sending_list(), self._receiving_list(), self.net.step)


# ============================== 2 -> 2 (first diverging then merging) =====================================

class FreewayRampJunction(Node):
    # Incoming links: one freeway mainline, one on-ramp.
    # Outgoing links: one off-ramp, one freeway mainline. 
    def __init__(self, ID, incoming_cell_list, outgoing_cell_list, 
                 onramp_priority, split_ratio, is_split_ratio_constant=True,
                 controller=None,  net=None, is_saved=True):

        super().__init__(ID, incoming_cell_list, outgoing_cell_list, controller, net, is_saved)

        # Shape of merging priority: (1, 1) or (state_len, 1).
        self.param['onramp_priority'] = np.atleast_2d(onramp_priority)

        # Shape of split ratio: if constant, (1, 2) or (state_len, 2); if not constant, (1, 2, num_step) or (state_len, 2, num_step).
        self.param['split_ratio'] = np.atleast_2d(split_ratio) if is_split_ratio_constant else  np.atleast_3d(split_ratio)

        self.param['is_split_ratio_constant'] = is_split_ratio_constant

    def _onramp_priority(self):
        return self.param['onramp_priority'][:, 0]

    def _split_ratio(self, step=None):
        if self.param['is_split_ratio_constant']:
            split_to_mainline, split_to_offramp = self.param['split_ratio'][:, 0], self.param['split_ratio'][:, 1]
        else:
            split_to_mainline, split_to_offramp = self.param['split_ratio'][:, 0, step], self.param['split_ratio'][:, 1, step]
        return split_to_mainline, split_to_offramp


    def compute_inter_cell_flow(self, state_len, sending_list, receiving_list, step=None):
        sending_mainline, sending_onramp = sending_list
        receiving_mainline, receiving_offramp = receiving_list

        p_onramp = self._onramp_priority()
        split_to_mainline, split_to_offramp = self._split_ratio(step)

        # Compute flow from onramp to mainline.
        if self.controller is None:
            flow_onramp_to_mainline = np.minimum(sending_onramp, receiving_mainline)
        else:
            flow_onramp_to_mainline = np.minimum.reduce((sending_onramp, receiving_mainline, self.controller.get_control_input()))

        # Compute flow from mainline to mainline.
        sending_mainline_to_mainline = split_to_mainline * np.minimum(sending_mainline, utils.safe_div(receiving_offramp, split_to_offramp))
        flow_mainline_to_mainline = np.minimum(sending_mainline_to_mainline, receiving_mainline - p_onramp * flow_onramp_to_mainline)

        # Compute flow from mainline to off-ramp. 
        flow_mainline_to_offramp  = np.where(
            split_to_mainline != 0, 
            flow_mainline_to_mainline * utils.safe_div(split_to_offramp, split_to_mainline), 
            np.minimum(sending_mainline, receiving_offramp), 
        )

        inter_cell_flow = np.zeros([state_len, len(sending_list), len(receiving_list)])
        inter_cell_flow[:, 0, 0] = flow_mainline_to_mainline
        inter_cell_flow[:, 0, 1] = flow_mainline_to_offramp
        inter_cell_flow[:, 1, 0] = flow_onramp_to_mainline

        return inter_cell_flow


    def _compute_inter_cell_flow(self):
        return self.compute_inter_cell_flow(self.net.param['state_len'], self._sending_list(), self._receiving_list(), self.net.step)




if __name__ == '__main__':
    pass