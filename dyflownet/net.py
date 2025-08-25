import numpy as np
import time

class Network:
    __slots__ = ('ID', 'num_step', 'time_step_size', 'step', 'source_list', 'link_list', 'sink_list', 'node_list')

    def __init__(self, ID, num_step, time_step_size, source_list=None, link_list=None, sink_list=None, node_list=None) -> None:
        self.ID = ID

        self.step = 0

        # Number of step is integer.
        self.num_step = num_step 

        # Time step size is scalar.
        self.time_step_size = time_step_size 

        self.source_list = source_list if source_list is not None else []
        self.link_list = link_list if link_list is not None else []
        self.sink_list = sink_list if sink_list is not None else []
        self.node_list = node_list if node_list is not None else []

    def add_component(self, item_type, item):
        if item_type == 'source':
            self.source_list.append(item)
            self.source_list[-1].net = self
        elif item_type == 'link':
            self.link_list.append(item)
            self.link_list[-1].net = self
        elif item_type == 'sink':
            self.sink_list.append(item)
            self.sink_list[-1].net = self
        elif item_type == 'node':
            self.node_list.append(item)
            self.node_list[-1].net = self
    

    def initialize_cell(self):
        for c_list in (self.source_list, self.link_list, self.sink_list):
            for c in c_list:
                c.initialize()


    def initialize_node(self):
        for n in self.node_list:
            n.initialize()


    def update_cell(self):
        for c_list in (self.source_list, self.link_list, self.sink_list):
            for c in c_list:
                c.update_speed()
                c.update_density()
        
    def save_cell_output(self):
        for c_list in (self.source_list, self.link_list, self.sink_list):
            for c in c_list:
                c.save_output()

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

    def update_inter_cell_flow(self):
        for n in self.node_list:
            n.update_inter_cell_flow() 

    def run_one_step(self):
        # Update boundary inflow and outflows.
        self.update_boundary_inflow()
        self.update_boundary_outflow()

        # Update sending and receiving flows of links.
        self.update_receiving()
        self.update_sending()

        # Update inter-cell flows. 
        self.update_inter_cell_flow()

        # Update cell states.
        self.update_cell()

        # Save results. 
        self.save_cell_output()

        self.step += 1


    def initialize(self):
        self.initialize_cell()
        self.initialize_node()
        self.step = 0


    def run(self):
        start_time = time.time()

        self.initialize()

        for _ in range(self.num_step):
            self.run_one_step()
        
        end_time = time.time()

        print(f'time cost: {end_time-start_time:.1f} seconds.')


if __name__ == '__main__':
    pass