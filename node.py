import numpy as np

def safe_div(x, y, fill = np.inf):
    return np.divide(x, y, out=np.full_like(x, fill, dtype=float), where=(y != 0))

def merge_two_flows_with_priority(s_0, s_1, r, p_0, p_1):
    is_space_enough = (s_0 + s_1) <= r
    flow_0 = np.where(is_space_enough, s_0, np.median([s_0, r - s_1, p_0 * r], axis=0))
    flow_1 = np.where(is_space_enough, s_1, np.median([s_1, r - s_0, p_1 * r], axis=0))
    return flow_0, flow_1

# ============================== Node =====================================

class Node:
    def __init__(self, ID, incoming_cell_list, outgoing_cell_list):
        self.ID = ID
        self.param = {}
    
        self.net = None

        self.incoming_cell_list = incoming_cell_list
        for c in self.incoming_cell_list:
            c.node['tail'] = self

        self.outgoing_cell_list = outgoing_cell_list
        for c in self.outgoing_cell_list:
            c.node['head'] = self

    def initialize(self):
        pass

    def _sending_list(self):
        return [cell.co_state['sending'] for cell in self.incoming_cell_list]

    def _receiving_list(self):
        return [cell.co_state['receiving'] for cell in self.outgoing_cell_list]
    
    def update_inter_cell_flow(self) -> None:
        outflow_list, inflow_list = self.assign_inter_cell_flow()

        for cell, outflow in zip(self.incoming_cell_list, outflow_list):
            cell.co_state['outflow'] = outflow
            
        for cell, inflow in zip(self.outgoing_cell_list, inflow_list):
            cell.co_state['inflow'] = inflow
        
    def assign_inter_cell_flow(self):
        outflow_list = [None for _ in self.incoming_cell_list]
        inflow_list = [None for _ in self.outgoing_cell_list]
        return outflow_list, inflow_list


# ============================== 1 -> 1 =====================================

class BasicJunction(Node):
    def __init__(self, ID, incoming_cell_list, outgoing_cell_list):
        super().__init__(ID, incoming_cell_list, outgoing_cell_list)

    def assign_inter_cell_flow(self):
        flow = np.minimum(self._sending_list()[0],  self._receiving_list()[0])
        return [flow], [flow]


# ============================== 2 -> 1 (merging) =====================================
class TwoToOneMergeJunction(Node):
    def __init__(self, ID, incoming_cell_list, outgoing_cell_list, merging_priority):
        
        super().__init__(ID, incoming_cell_list, outgoing_cell_list)

        # Shape of merging priority: (1, 2) or (state_len, 2).
        self.param['merging_priority'] = np.atleast_2d(merging_priority)

    def _merging_priority(self):
        return self.param['merging_priority'][:, 0], self.param['merging_priority'][:, 1] 

    def assign_inter_cell_flow(self):
        sending_i_0, sending_i_1 = self._sending_list()
        receiving_j_0 = self._receiving_list()[0]
        p_i_0, p_i_1 = self._merging_priority() 

        flow_i_0_j_0, flow_i_1_j_0 = merge_two_flows_with_priority(sending_i_0, sending_i_1, receiving_j_0, p_i_0, p_i_1)

        outflow_list = [flow_i_0_j_0, flow_i_1_j_0]
        inflow_list = [flow_i_0_j_0 + flow_i_1_j_0]

        return outflow_list, inflow_list


# ============================== 1 -> 2 (diverging) =====================================

class OneToTwoDivergeJunction(Node):
    def __init__(self, ID, incoming_cell_list, outgoing_cell_list, 
                 split_ratio, is_split_ratio_constant=True, is_FIFO=True):
        
        super().__init__(ID, incoming_cell_list, outgoing_cell_list)

        # Shape of split ratio: if constant, (1, 2) or (state_len, 2); if not constant, (1, 2, num_step) or (state_len, 2, num_step). 
        self.param['split_ratio'] = np.atleast_2d(split_ratio) if is_split_ratio_constant else  np.atleast_3d(split_ratio)
        self.param['is_split_ratio_constant'] = is_split_ratio_constant
        self.param['is_FIFO'] = is_FIFO

    def _split_ratio(self):
        if self.param['is_split_ratio_constant']:
            split_j_0, split_j_1 = self.param['split_ratio'][:, 0], self.param['split_ratio'][:, 1]
        else:
            split_j_0, split_j_1 = self.param['split_ratio'][:, 0, self.net.step], self.param['split_ratio'][:, 1, self.net.step]
        return split_j_0, split_j_1

    def assign_inter_cell_flow(self):
        sending_i_0  = self._sending_list()[0]
        receiving_j_0, receiving_j_1 = self._receiving_list()
        split_j_0, split_j_1 = self._split_ratio()

        if not self.param['is_FIFO']:
            flow_i_0_j_0 = np.minimum(split_j_0 * sending_i_0, receiving_j_0)
            flow_i_0_j_1 = np.minimum(split_j_1 * sending_i_0, receiving_j_1)
        else:
            total_flow = np.minimum.reduce((sending_i_0, safe_div(receiving_j_0, split_j_0), safe_div(receiving_j_1, split_j_1)))
            flow_i_0_j_0 = split_j_0 * total_flow
            flow_i_0_j_1 = split_j_1 * total_flow

        outflow_list = [flow_i_0_j_0 + flow_i_0_j_1]
        inflow_list = [flow_i_0_j_0, flow_i_0_j_1]

        return outflow_list, inflow_list


# ============================== 2 -> 2 (first diverging then merging) =====================================

class FreewayRampJunction(Node):
    # Incoming links: one freeway mainline, one on-ramp.
    # Outgoing links: one off-ramp, one freeway mainline. 
    def __init__(self, ID, incoming_cell_list, outgoing_cell_list, 
                 merging_priority, split_ratio, is_split_ratio_constant=True):

        super().__init__(ID, incoming_cell_list, outgoing_cell_list)

        # Shape of merging priority: (1, 2) or (state_len, 2).
        self.param['merging_priority'] = np.atleast_2d(merging_priority)

        # Shape of split ratio: if constant, (1, 2) or (state_len, 2); if not constant, (1, 2, num_step) or (state_len, 2, num_step).
        self.param['split_ratio'] = np.atleast_2d(split_ratio) if is_split_ratio_constant else  np.atleast_3d(split_ratio)

        self.param['is_split_ratio_constant'] = is_split_ratio_constant

    def _merging_priority(self):
        return self.param['merging_priority'][:, 0], self.param['merging_priority'][:, 1] 

    def _split_ratio(self):
        if self.param['is_split_ratio_constant']:
            split_to_mainline, split_to_offramp = self.param['split_ratio'][:, 0], self.param['split_ratio'][:, 1]
        else:
            split_to_mainline, split_to_offramp = self.param['split_ratio'][:, 0, self.net.step], self.param['split_ratio'][:, 1, self.net.step]
        return split_to_mainline, split_to_offramp

    def assign_inter_cell_flow(self):
        sending_mainline, sending_onramp = self._sending_list()
        receiving_mainline, receiving_offramp = self._receiving_list()
        p_mainline, p_onramp = self._merging_priority()
        split_to_mainline, split_to_offramp = self._split_ratio()

        # Compute flows i) from mainline to mainline and ii) from mainline to on-ramp. 
        flow_mainline_to_mainline, flow_onramp_to_mainline = merge_two_flows_with_priority(
            split_to_mainline * np.minimum(sending_mainline, safe_div(receiving_offramp, split_to_offramp)), # sending flow from mainline to mainline, restricted by off-ramp due to FIFO. 
            sending_onramp, # sending flow from on-ramp to mainline. 
            receiving_mainline, # receiving flow of downstream mainline. 
            p_mainline, # merging priority for upstream mainline. 
            p_onramp,# merging priority for onramp.  
        )

        # Compute flow from mainline to off-ramp. 
        flow_mainline_to_offramp  = np.where(
            split_to_mainline != 0, 
            flow_mainline_to_mainline * safe_div(split_to_offramp, split_to_mainline), 
            np.minimum(sending_mainline, receiving_offramp), 
        )

        outflow_list = [flow_mainline_to_mainline + flow_mainline_to_offramp, flow_onramp_to_mainline]
        inflow_list = [flow_mainline_to_mainline + flow_onramp_to_mainline, flow_mainline_to_offramp]

        return outflow_list, inflow_list
    

if __name__ == '__main__':
    pass