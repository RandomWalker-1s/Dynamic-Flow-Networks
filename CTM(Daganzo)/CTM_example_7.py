import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../')

import dyflownet as dfn 



v, F, w = 1, 50, 0.25
max_density, max_speed = 250, 1
num_link = 100

def build_corridor():
    # Create corridor. 
    corridor = dfn.net.Network(ID='corridor', state_len=1, num_step=40, time_step_size=1)

    # Create source.  
    source_0 = dfn.cell.Source(
        ID='source_0', 
        initial_condition={'density': 25}, 
        boundary_inflow=dfn.flow.BoundaryInflow(25), 
        sending=dfn.flow.PiecewiseLinearSendingFlow(v, F),
    )

    # Create link. 
    link_list = []
    for i in range(num_link):
        initial_density = 250 if i in [51, 52, 53] else 25
        cell_up = link_list[-1] if i > 0 else source_0

        link = dfn.cell.Link(
            ID=f'link_{i}', 
            max_density = max_density, 
            max_speed = max_speed, 
            initial_condition = {'density': initial_density},
            receiving = dfn.flow.LookAheadPiecewiseLinearReceivingFlow(
                congestion_wave_speed = w,
                max_density = max_density,
                capacity = F,
                cell_upstream = cell_up,
                look_ahead_density_threshold = 50, 
                look_ahead_congestion_wave_speed = v, 
                look_ahead_max_density = max_density, 
                look_ahead_capacity=F, 
            ),
            sending=dfn.flow.PiecewiseLinearSendingFlow(v, F),
        )
        link_list.append(link)

    # Create sink. 
    sink_0 = dfn.cell.Sink(
        ID='sink_0', 
        max_density = max_density, 
        max_speed = max_speed, 
        initial_condition = {'density': 25},
        receiving = dfn.flow.LookAheadPiecewiseLinearReceivingFlow(
            congestion_wave_speed = w,
            max_density = max_density,
            capacity = F,
            cell_upstream = link_list[-1],
            look_ahead_density_threshold = 50, 
            look_ahead_congestion_wave_speed = v, 
            look_ahead_max_density = max_density, 
            look_ahead_capacity = F, 
        ),
        boundary_outflow=dfn.flow.BoundaryOutflow(v, F),
    )

    # Create nodes. 
    node_list = []

    node = dfn.node.BasicJunction(
        ID='node_0', 
        incoming_cell_list=[source_0], 
        outgoing_cell_list=[link_list[0]],
    )
    node_list.append(node)

    for i in range(1, num_link):
        node = dfn.node.BasicJunction(
            ID=f'node_{i}', 
            incoming_cell_list=[link_list[i-1]], 
            outgoing_cell_list=[link_list[i]],
        )
        node_list.append(node)

    node = dfn.node.BasicJunction(
        ID='node_100', 
        incoming_cell_list=[link_list[-1]], 
        outgoing_cell_list=[sink_0],
    )
    node_list.append(node)

    # Add cells to corridor. 
    corridor.add_cell('source', source_0)
    for link in link_list:
        corridor.add_cell('link', link)
    corridor.add_cell('sink', sink_0)

    # Add nodes to corridor. 
    for node in node_list:
        corridor.add_node(node)

    return corridor


# Build corridor. 
corridor = build_corridor()

# Run simulation. 
corridor.run()

# Put density together. 
link_density = []
for l in corridor.link_list:
    link_density.append(l.state_output['density'])
link_density = np.squeeze(link_density).transpose()



plt.imshow(link_density, aspect='auto', vmin=0, vmax=100)

plt.gca().set_xticks(np.arange(0, 100, 4))
plt.gca().set_xticks(np.arange(-0.5, 100, 1), minor=True)

for tick in plt.gca().xaxis.get_major_ticks():
    tick.label1.set_visible(True)
    tick.label1.set_fontsize(8)

plt.gca().set_yticks(np.arange(0, 41, 2))
plt.gca().set_yticks(np.arange(-0.5, 41, 1), minor=True)

for tick in plt.gca().yaxis.get_major_ticks():
    tick.label1.set_visible(True)
    tick.label1.set_fontsize(8)

plt.xlabel('Position')
plt.ylabel('Time')


for time in range(41):
    for pos in range(100):
        plt.text(x=pos, y=time, s=str(int(link_density[time, pos])), ha='center', va='center', fontsize=1.5, color='red', fontweight='bold')

plt.grid(True, which='minor')
plt.tight_layout()
plt.savefig('./Corridor_example_7_fig_1.pdf')
plt.close()


plt.figure(figsize=(6, 5))
plt.plot(range(21), np.concatenate([[0], np.cumsum(link_density[0, 43:63])]), 'o--')
plt.plot(range(21), np.concatenate([[0], np.cumsum(link_density[8, 43:63])]), 'o--')
plt.plot(range(21), np.concatenate([[0], np.cumsum(link_density[16, 43:63])]), 'o--')
plt.plot(range(21), np.concatenate([[0], np.cumsum(link_density[24, 43:63])]), 'o--')
plt.plot(range(21), np.concatenate([[0], np.cumsum(link_density[32, 43:63])]), 'o--')

plt.legend(['time 0', 'time 8', 'time 16', 'time 24', 'time 32'])

plt.gca().set_xticks([0, 5, 10, 15, 20])
plt.gca().set_xticks(np.linspace(0, 20, 21), minor=True)

plt.gca().set_yticks([0, 200, 400, 600, 800, 1000, 1200])

plt.grid()
plt.tight_layout()
plt.savefig('./Corridor_example_7_fig_2.pdf')
plt.close()