import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import re

# Function to read the results from the file
def read_results(filename):
    x = []
    rho = []
    u = []
    p = []
    with open(filename, 'r') as file:
        # Skip the first row as it contains a single float number
        t = file.readline()
        for line in file:
            data = line.split()
            x.append(float(data[0]))
            rho.append(float(data[1]))
            u.append(float(data[2]))
            p.append(float(data[3]))
    return t, x, rho, u, p

# Custom sorting function to extract numerical value from filename
def numerical_sort(value):
    numbers = re.findall(r'\d+', value)
    return int(numbers[0]) if numbers else -1

for cgpu in ['cpu', 'gpu']:
    # Find all results files
    result_files = sorted(glob.glob("results/"+cgpu+"_t_*.dat"), key=numerical_sort)

    # Read the first file to get x values and initialize lists for rho, u, p
    _, x, _, _, _ = read_results(result_files[0])
    all_rho = []
    all_u = []
    all_p = []

    # Read all files and store the results
    for file in result_files:
        t, _, rho, u, p = read_results(file)
        all_rho.append(rho)
        all_u.append(u)
        all_p.append(p)

    # Create a figure and subplots
    fig, axs = plt.subplots(3, 1, figsize=(14, 8))

    fig.suptitle(cgpu+" implementation", fontsize=16)

    # Initialize plots
    line_rho, = axs[0].plot(x, all_rho[0], label='Density', color='blue')
    line_u, = axs[1].plot(x, all_u[0], label='Velocity', color='green')
    line_p, = axs[2].plot(x, all_p[0], label='Pressure', color='red')

    for ax in axs:
        ax.legend()
        ax.grid()

    axs[0].set_ylabel('Density')
    axs[1].set_ylabel('Velocity')
    axs[2].set_ylabel('Pressure')
    axs[2].set_xlabel('Position')

    axs[0].set_xlim([0, 1])
    axs[1].set_xlim([0, 1])
    axs[2].set_xlim([0, 1])

    axs[0].set_ylim([0.2, 1.4])
    axs[1].set_ylim([0, 2])
    axs[2].set_ylim([0, 4])

    # Animation update function
    def update(frame):
        line_rho.set_ydata(all_rho[frame])
        line_u.set_ydata(all_u[frame])
        line_p.set_ydata(all_p[frame])
        return line_rho, line_u, line_p

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(result_files), blit=True)

    # Save the animation
    ani.save("results/"+cgpu+"_results.mp4", writer='ffmpeg')
