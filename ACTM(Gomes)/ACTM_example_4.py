import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import itertools

sys.path.append('../')
import dyflownet as dfn 



v, F, w = 60, 6000, 20
max_density, max_speed = 400, 60

# Initialize three cells.
initial_density = list(itertools.product([0, 400], [0, 400], [0, 400]))
for i in [0, 400]:
    for j in np.linspace(40, 360, 9):
        initial_density.append([i, i, j])
        initial_density.append([400-i, i, j])
        
        initial_density.append([i, j, i])
        initial_density.append([i, j, 400-i])

        initial_density.append([j, i, i])
        initial_density.append([j, i, 400-i])
for i in [0, 400]:
    for j in np.linspace(40, 360, 9):
        for k in np.linspace(40, 360, 9):
            initial_density.append([i, j, k])
            initial_density.append([j, i, k])
            initial_density.append([j, k, i])

initial_density_link_0, initial_density_link_1, initial_density_link_2 = list(zip(*initial_density))

initial_density_link_0 = np.atleast_1d(initial_density_link_0)
initial_density_link_1 = np.atleast_1d(initial_density_link_1)
initial_density_link_2 = np.atleast_1d(initial_density_link_2)

state_len = len(initial_density_link_0)

def build_corridor():
    source_0 = dfn.cell.Source(
        ID = 'source_0',
        state_len = state_len,
        initial_condition={'density': [0]*state_len},
        boundary_inflow = dfn.flow.BoundaryInflow([4800]*state_len),
        sending = dfn.flow.BufferSendingFlow([4800]*state_len, ignore_queue=True),
    )

    source_1 = dfn.cell.Source(
        ID = 'source_1',
        state_len = state_len,
        initial_condition={'density': [0]*state_len},
        boundary_inflow = dfn.flow.BoundaryInflow([1200]*state_len),
        sending = dfn.flow.BufferSendingFlow([1200]*state_len, ignore_queue=True),
    )

    link_0 = dfn.cell.Link(
        ID = 'link_0',
        max_density = max_density,
        max_speed = max_speed,
        state_len = state_len,
        initial_condition={'density': initial_density_link_0},
        receiving = dfn.flow.PiecewiseLinearReceivingFlow(w, max_density, F), 
        sending = dfn.flow.PiecewiseLinearSendingFlow(v, F),
    )

    link_1 = dfn.cell.Link(
        ID = 'link_1',
        max_density = max_density,
        max_speed = max_speed,
        state_len = state_len,
        initial_condition={'density': initial_density_link_1},
        receiving = dfn.flow.PiecewiseLinearReceivingFlow(w, max_density, F), 
        sending = dfn.flow.PiecewiseLinearSendingFlow(v, F),
    )

    link_2 = dfn.cell.Link(
        ID = 'link_2',
        max_density = max_density,
        max_speed = max_speed,
        state_len = state_len,
        initial_condition={'density': initial_density_link_2},
        receiving = dfn.flow.PiecewiseLinearReceivingFlow(w, max_density, F), 
        sending = dfn.flow.PiecewiseLinearSendingFlow(v, F),
    )

    sink_0 = dfn.cell.Sink(
        ID = 'sink_0',
        max_density = max_density,
        max_speed = max_speed,
        state_len = state_len,
        initial_condition={'density': [0]*state_len},
        receiving = dfn.flow.PiecewiseLinearReceivingFlow(w, max_density, F),
        boundary_outflow = dfn.flow.BoundaryOutflow(v, F),
    )

    # Create node. 
    node_0 = dfn.node.BasicJunction(ID = 'node_0', incoming_cell_list=[source_0], outgoing_cell_list=[link_0])
    node_1 = dfn.node.BasicJunction(ID = 'node_1', incoming_cell_list=[link_0], outgoing_cell_list=[link_1])
    node_2 = dfn.node.TwoToOneMergeJunction(ID = 'node_1', incoming_cell_list=[link_1, source_1], outgoing_cell_list=[link_2], merging_priority=[1, 1])
    node_3 = dfn.node.BasicJunction(ID = 'node_2', incoming_cell_list=[link_2], outgoing_cell_list=[sink_0])

    # Create corridor. 
    corridor = dfn.net.Network(ID='corridor', num_step=3600, time_step_size=6/3600)

    # Add cells to corridor. 
    corridor.add_component('source', source_0)
    corridor.add_component('source', source_1)
    corridor.add_component('link', link_0)
    corridor.add_component('link', link_1)
    corridor.add_component('link', link_2)
    corridor.add_component('sink', sink_0)


    # Add nodes to corridor. 
    corridor.add_component('node', node_0)
    corridor.add_component('node', node_1)
    corridor.add_component('node', node_2)
    corridor.add_component('node', node_3)

    return corridor


corridor = build_corridor()

corridor.run()


ax = plt.figure(figsize=(6, 6)).add_subplot(projection='3d')
plt.rcParams.update({'font.size': 8})

ax.scatter(corridor.link_list[2].state_output['density'][3600, :], 
           corridor.link_list[1].state_output['density'][3600, :],
           corridor.link_list[0].state_output['density'][3600, :], color="red")

for i in range(state_len):
    ax.plot(corridor.link_list[2].state_output['density'][:, i],
            corridor.link_list[1].state_output['density'][:, i],
            corridor.link_list[0].state_output['density'][:, i], 'b', linewidth="0.5")

ax.set_xlabel('Downstream density')
ax.set_ylabel('Middle density')
ax.set_zlabel('Upstream density')

plt.tight_layout()
plt.savefig("./ACTM_example_4.pdf")