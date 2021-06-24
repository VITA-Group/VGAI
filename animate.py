import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle

arch='vis_grnn'
v_max=3.0
scale=6.0
F=24
K=3
radius=1.5
n_agents = 50
seed=3
step=99
trace_dir = '{}/vinit{}_scale{}_F{}_K{}_radius_{}_N{}_seed{}/timeSteps_{}.pkl'.format(arch, int(v_max), scale, F, K, radius, n_agents, seed,step)
trace = pickle.load(open(trace_dir, "rb" ))

fig, ax = plt.subplots()

fig.set_tight_layout(True)




# Query the figure's on-screen size and DPI. Note that when saving the figure to

# a file, we need to provide a DPI for that separately.

print('fig size: {0} DPI, size in inches {1}'.format(

    fig.get_dpi(), fig.get_size_inches()))



# Plot a scatter that persists (isn't redrawn) and the initial line.

x = trace['init_state'][:,0]
y = trace['init_state'][:,1]
print(x.shape)
print(y.shape)
print('costs = ' + str(trace['costs'][0]))
print('final_cost = {}'.format(trace['costs'][0].sum()))
print('seperate mark')
#x = np.arange(0, 20, 0.1)
#ax.scatter(x, x + np.random.normal(0, 3.0, len(x)))
print(x)
print(y)
f = ax.scatter(x,y)
ax.set(xlim=(-10, 10), ylim=(-10, 10))
#line, = ax.plot(x, x - 5, 'r-', linewidth=2)
dec = 0
#for i in range(100):
#    x = trace['states'][:, :, i]
    #v = (x[:, 2] ** 2 + x[:, 3] ** 2) ** 0.5
#    v = (np.mean(x[:, 2]) ** 2 + np.mean(x[:, 3]) ** 2 ) ** 0.5
#    dec += v
    

final_cost = trace['costs'][0][30:].sum() / 10.0
print('final cost = {}'.format(final_cost)) 
 



def update(i):
    #fig.clear()

    label = 'timestep {0}'.format(i)

    print(label)

    # Update the line and the axes (with a new xlabel). Return a tuple of

    # "artists" that have to be redrawn for this frame.

    #print(trace['states'].shape)
    x = trace['states'][:,0,i]
    y = trace['states'][:,1,i]
    cost = trace['costs'][:, i]
    
    data = trace['states'][:,0:2,i]
    #print(data.shape)
    f.set_offsets(data)
    
    #line.set_ydata(x - 5 + i)
    #ax.set_xlabel(str(label) + '   cost   ' + str(costs[:,i]))
    ax.set_xlabel('label = {}, cost = {:.6f}'.format(label, cost[0]))
    #print('record cost = {}'.format(cost))
    #print('cal cost = {}'.format(np.sum(np.var(trace['states'][:, 2:4, i], axis=0))))
    
    return ax



if __name__ == '__main__':

    # FuncAnimation will call the 'update' function for each frame; here

    # animating over 10 frames, with an interval of 200ms between frames.

    anim = FuncAnimation(fig, update, frames=np.arange(0, step), interval=200)

    if len(sys.argv) > 1 and sys.argv[1] == 'save':
 
        save_path = '{}_vinit{}_scale{}_F{}_K{}_radius_{}_N{}_seed{}_t{}.gif'.format(arch, int(v_max), scale, F, K, radius, n_agents, seed,step)

        anim.save(save_path, dpi=80, writer='imagemagick')

    else:

        # plt.show() will just loop the animation forever.

        plt.show()

    print('final_cost = {}'.format(trace['costs'][0].sum()))
    f_str = str([x for x in list(trace['states'][:,0,0])])
    text_file = open("init_x.txt", "w")
    text_file.write(f_str)
    text_file.close()

    f_str = str([x for x in list(trace['states'][:,1,0])])
    text_file = open("init_y.txt", "w")
    text_file.write(f_str)
    text_file.close()

    f_str = str([x for x in list(trace['states'][:,0,98])])
    text_file = open("x98.txt", "w")
    text_file.write(f_str)
    text_file.close()

    f_str = str([x for x in list(trace['states'][:,1,98])])
    text_file = open("y98.txt", "w")
    text_file.write(f_str)
    text_file.close()

    f_str = str([x for x in list(trace['states'][:,0,99] -  trace['states'][:,0,98])])
    text_file = open("vx100.txt", "w")
    text_file.write(f_str)
    text_file.close()


    f_str = str([x for x in list(trace['states'][:,1,99] -  trace['states'][:,1,98])])
    text_file = open("vy100.txt", "w")
    text_file.write(f_str)
    text_file.close()

