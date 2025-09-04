import math
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../')
import dyflownet as dfn 


v, F, w = 60, 6000, 20
max_density, max_speed = 400, 60

initial_density = dfn.utils.generate_boundary_combos(np.linspace(0, 400, 41), np.linspace(0, 400, 41))

initial_density_link_0 = initial_density[0, :]
initial_density_link_1 = initial_density[1, :]

state_len = len(initial_density_link_1)


def build_corridor():
    # Create corridor. 
    corridor = dfn.net.Network(ID='corridor', state_len=state_len, num_step=3600, time_step_size=6/3600)

    source_0 = dfn.cell.Source(
        ID = 'source_0',
        initial_condition={'density': [0]*state_len},
        boundary_inflow = dfn.flow.BoundaryInflow([4800]*state_len),
        sending = dfn.flow.BufferSendingFlow([4800]*state_len, ignore_queue=True),
    )

    source_1 = dfn.cell.Source(
        ID = 'source_1',
        initial_condition={'density': [0]*state_len},
        boundary_inflow = dfn.flow.BoundaryInflow([1200]*state_len),
        sending = dfn.flow.BufferSendingFlow([1200]*state_len, ignore_queue=True),
    )

    link_0 = dfn.cell.Link(
        ID = 'link_0',
        max_density = max_density,
        max_speed = max_speed,
        initial_condition={'density': initial_density_link_0},
        receiving = dfn.flow.PiecewiseLinearReceivingFlow(w, max_density, F), 
        sending = dfn.flow.PiecewiseLinearSendingFlow(v, F),
    )

    link_1 = dfn.cell.Link(
        ID = 'link_1',
        max_density = max_density,
        max_speed = max_speed,
        initial_condition={'density': initial_density_link_1},
        receiving = dfn.flow.PiecewiseLinearReceivingFlow(w, max_density, F), 
        sending = dfn.flow.PiecewiseLinearSendingFlow(v, F),
    )

    sink_0 = dfn.cell.Sink(
        ID = 'sink_0',
        max_density = max_density,
        max_speed = max_speed,
        initial_condition={'density': [0]*state_len},
        receiving = dfn.flow.PiecewiseLinearReceivingFlow(w, max_density, F),
        boundary_outflow = dfn.flow.BoundaryOutflow(v, F),
    )

    # Create node. 
    node_0 = dfn.node.BasicJunction(ID = 'node_0', incoming_cell_list=[source_0], outgoing_cell_list=[link_0])
    node_1 = dfn.node.TwoToOneMergeJunction(ID = 'node_1', incoming_cell_list=[link_0, source_1], outgoing_cell_list=[link_1], merging_priority=[1, 1])
    node_2 = dfn.node.BasicJunction(ID = 'node_2', incoming_cell_list=[link_1], outgoing_cell_list=[sink_0])


    # Add cells to corridor. 
    corridor.add_cell('source', source_0)
    corridor.add_cell('source', source_1)
    corridor.add_cell('link', link_0)
    corridor.add_cell('link', link_1)
    corridor.add_cell('sink', sink_0)

    # Add nodes to corridor. 
    corridor.add_node(node_0)
    corridor.add_node(node_1)
    corridor.add_node(node_2)

    return corridor


corridor = build_corridor()

corridor.run()


plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 8})

plt.scatter(corridor.link_list[1].state_output['density'][2, :], corridor.link_list[0].state_output['density'][2, :], s=5)
plt.scatter(corridor.link_list[1].state_output['density'][10, :], corridor.link_list[0].state_output['density'][10, :], s=5)
plt.scatter(corridor.link_list[1].state_output['density'][50, :], corridor.link_list[0].state_output['density'][50, :], s=5)
plt.scatter(corridor.link_list[1].state_output['density'][100, :], corridor.link_list[0].state_output['density'][100, :], s=5)
plt.scatter(corridor.link_list[1].state_output['density'][3600, :], corridor.link_list[0].state_output['density'][3600, :], s=10, color='k')

plt.legend(["12s", "60s", "300s", "600s", "21600s"], loc='upper right')

for i in range(state_len):
    plt.plot(corridor.link_list[1].state_output['density'][:, i], corridor.link_list[0].state_output['density'][:, i], 'b', linewidth="0.5")

plt.xlabel('Downstream density')
plt.ylabel('Upstream density')
plt.title('Mainline flow: 4800, on-ramp flow: 1200')

plt.grid()

plt.savefig('./ACTM_example_1.pdf')
plt.close()