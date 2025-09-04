import numpy as np
import time

class Network:
    def __init__(self, ID, num_step, state_len, time_step_size, source_list=None, link_list=None, sink_list=None, node_list=None) -> None:
        self.ID = ID

        self.step = 0

        self.param = {
            'num_step': num_step, 
            'state_len': state_len, 
            'time_step_size': time_step_size,
        }

        self.source_list = source_list if source_list is not None else []
        self.link_list = link_list if link_list is not None else []
        self.sink_list = sink_list if sink_list is not None else []
        self.node_list = node_list if node_list is not None else []


    def add_cell(self, cell_type, cell):
        if cell_type == 'source':
            self.source_list.append(cell)
        elif cell_type == 'link':
            self.link_list.append(cell)
        elif cell_type == 'sink':
            self.sink_list.append(cell)

        cell.hook_up_to_net(self)
        for flow in cell.flow_dict.values():
            flow.hook_up_to_net(self)


    def add_node(self, node):
        self.node_list.append(node)

        node.hook_up_to_net(self)
        if node.controller is not None:
            node.controller.hook_up_to_net(self)
    
    
    def initialize_cell(self):
        for c_list in (self.source_list, self.link_list, self.sink_list):
            for c in c_list:
                c.initialize()


    def initialize_node(self):
        for n in self.node_list:
            n.initialize()


    def update_receiving(self):
        for c_list in (self.link_list, self.sink_list):
            for c in c_list:
                c.update_receiving()


    def update_sending(self):
        for c_list in (self.source_list, self.link_list):
            for c in c_list:
                c.update_sending()


    def update_boundary_inflow(self):
        for s in self.source_list:
            s.update_boundary_inflow()


    def update_boundary_outflow(self):
        for s in self.sink_list:
            s.update_boundary_outflow()

    
    def update_control_input(self):
        for n in self.node_list:
            n.update_control_input()        


    def update_inter_cell_flow(self):
        for n in self.node_list:
            n.update_inter_cell_flow()


    def update_cell_outflow(self):
        for n in self.node_list:
            n.update_cell_outflow()


    def update_cell_inflow(self):
        for n in self.node_list:
            n.update_cell_inflow()


    def update_cell_speed(self):
        for c_list in (self.source_list, self.link_list, self.sink_list):
            for c in c_list:
                c.update_speed()


    def update_cell_density(self):
        for c_list in (self.source_list, self.link_list, self.sink_list):
            for c in c_list:
                c.update_density()

        
    def save_cell(self):
        for c_list in (self.source_list, self.link_list, self.sink_list):
            for c in c_list:
                c.save_output()


    def save_node(self):
        for n in self.node_list:
            n.save_output()
    

    def run_one_step(self):
        # Step 1: update boundary inflow and outflows.
        self.update_boundary_inflow()
        self.update_boundary_outflow()

        # Step 2: update sending and receiving flows of links.
        self.update_receiving()
        self.update_sending()

        # Step 3: update control inputs. 
        self.update_control_input()

        # Step 4: update inter-cell flows.
        self.update_inter_cell_flow()

        # Step 5: update cell inflows and outflows. 
        self.update_cell_outflow()
        self.update_cell_inflow()

        # Step 6: update cell speed and density. 
        self.update_cell_speed()
        self.update_cell_density()

        # Step 7: save results. 
        self.save_node()
        self.save_cell()

        self.step += 1


    def initialize(self):
        self.initialize_cell()
        self.initialize_node()
        self.step = 0


    def run(self):
        start_time = time.time()

        self.initialize()

        for _ in range(self.param['num_step']):
            self.run_one_step()
        
        end_time = time.time()

        print(f'time cost: {end_time-start_time:.1f} seconds.')


if __name__ == '__main__':
    pass