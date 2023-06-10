import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

import matplotlib.cm as cm

# # Generate some example data
# y = np.linspace(0, 1, 100)  # Linearly spaced data
# x = np.logspace(0.1, 1, 100)  # Logarithmically spaced data

# # # Create the plot
# fig, ax = plt.subplots(figsize=(8, 6))
# # fig.set_size_inches(8, 10)  # Adjust the size as needed

# Create a figure with the size of (8, 8)
fig = plt.figure(figsize=(8, 8))

# Create an Axes object within the figure
ax = fig.add_subplot(1, 1, 1)

# plt.figure()
# plt.figure(figsize=(8, 4))

# Set the x-axis to linear scale and y-axis to logarithmic scale
# ax.plot(x, y)

sz=50
edge_c='black'
scatter_order=10

# use ViT-B/32
# ax.scatter(17,0.678,s=sz,label="CLIP-Mesh")
# ax.scatter(200*60,0.786,s=sz,label="DreamFields")
# ax.scatter(12*60,0.751,s=sz,label="DreamFusion")
# ax.scatter(16/60,0.154,s=sz,label="Point-E (40M, text-only)")
# ax.scatter(1,0.365,s=sz,label="Point-E (40M)")
# ax.scatter(1.2,0.403,s=sz,label="Point-E (300M)")
# ax.scatter(25/60,0.336,s=sz,label="Point-E (300M, text-only)")
# ax.scatter(1.5,0.411,s=sz,label="Point-E (1B)")
# ax.scatter(1.0,0.411,s=sz,label="Shap-E (300M)")
# ax.scatter(13/60,0.378,s=sz,label="Shap-E (300M, text-only)")

# ax.scatter(17,0.678,s=sz,label="CLIP-Mesh", edgecolor=edge_c, zorder=scatter_order, marker='D')
# ax.scatter(200*60,0.786,s=sz,label="DreamFields", edgecolor=edge_c, zorder=scatter_order, marker='^')
# ax.scatter(12*60,0.751,s=sz,label="DreamFusion", edgecolor=edge_c, zorder=scatter_order, marker='d')
# ax.scatter(16/60,0.154,s=sz,label="Point-E (40M, text-only)", edgecolor=edge_c, zorder=scatter_order)
# ax.scatter(1,0.365,s=sz,label="Point-E (40M)", edgecolor=edge_c, zorder=scatter_order)
# ax.scatter(1.2,0.403,s=sz,label="Point-E (300M)", edgecolor=edge_c, zorder=scatter_order)
# ax.scatter(25/60,0.336,s=sz,label="Point-E (300M, text-only)", edgecolor=edge_c, zorder=scatter_order)
# ax.scatter(1.5,0.411,s=sz,label="Point-E (1B)", edgecolor=edge_c, zorder=scatter_order)
# ax.scatter(1.0,0.411,s=sz,label="Shap-E (300M)", edgecolor=edge_c, zorder=scatter_order, marker='s')
# ax.scatter(13/60,0.378,s=sz,label="Shap-E (300M, text-only)", edgecolor=edge_c, zorder=scatter_order, marker='s')

# ax.set_yscale('linear')
# ax.set_xscale('log')

# plt.scatter(17,0.678,s=sz,label="CLIP-Mesh (V100)", edgecolor=edge_c, zorder=scatter_order, marker='D')
# plt.scatter(200*60,0.786,s=sz,label="DreamFields (V100)", edgecolor=edge_c, zorder=scatter_order, marker='^')
# plt.scatter(12*60,0.751,s=sz,label="DreamFusion (V100)", edgecolor=edge_c, zorder=scatter_order, marker='d')
# plt.scatter(16/60,0.154,s=sz,label="Point-E (40M, text-only, V100)", edgecolor=edge_c, zorder=scatter_order)
# plt.scatter(1,0.365,s=sz,label="Point-E (40M, V100)", edgecolor=edge_c, zorder=scatter_order)
# plt.scatter(1.2,0.403,s=sz,label="Point-E (300M, V100)", edgecolor=edge_c, zorder=scatter_order)
# plt.scatter(25/60,0.336,s=sz,label="Point-E (300M, text-only, V100)", edgecolor=edge_c, zorder=scatter_order)
# plt.scatter(1.5,0.411,s=sz,label="Point-E (1B, V100)", edgecolor=edge_c, zorder=scatter_order)
# plt.scatter(1.0,0.411,s=sz,label="Shap-E (300M, V100)", edgecolor=edge_c, zorder=scatter_order, marker='s')
# plt.scatter(13/60,0.378,s=sz,label="Shap-E (300M, text-only, V100)", edgecolor=edge_c, zorder=scatter_order, marker='s')

# plt.scatter(52,0.678,s=sz,label="CLIP-Mesh (T4)", edgecolor=edge_c, zorder=scatter_order, marker='D')
# plt.scatter(69/60,0.154,s=sz,label="Point-E (40M, text-only, T4)", edgecolor=edge_c, zorder=scatter_order)
# plt.scatter(3.3,0.365,s=sz,label="Point-E (40M, T4)", edgecolor=edge_c, zorder=scatter_order)
# plt.scatter(1.9,0.336,s=sz,label="Point-E (300M, text-only, T4)", edgecolor=edge_c, zorder=scatter_order)
# plt.scatter(49/60,0.378,s=sz,label="Shap-E (300M, text-only, T4)", edgecolor=edge_c, zorder=scatter_order, marker='s')

# plt.scatter(3.1,0.154,s=sz,label="Point-E (40M, text-only, Jetson)", edgecolor=edge_c, zorder=scatter_order)
# plt.scatter(9.7,0.365,s=sz,label="Point-E (40M, Jetson)", edgecolor=edge_c, zorder=scatter_order)
# plt.scatter(5.2,0.336,s=sz,label="Point-E (300M, text-only, Jetson)", edgecolor=edge_c, zorder=scatter_order)
# plt.scatter(2.5,0.378,s=sz,label="Shap-E (300M, text-only, Jetson)", edgecolor=edge_c, zorder=scatter_order, marker='s')

# Create a scatter plot with specified colors
# scatter = ax.scatter(x, y, c=colors, cmap='viridis')

# ax.scatter(17,0.678,s=sz,label="CLIP-Mesh (V100)", edgecolor=edge_c, zorder=scatter_order, marker='D')
# ax.scatter(200*60,0.786,s=sz,label="DreamFields (V100)", edgecolor=edge_c, zorder=scatter_order, marker='^')
# ax.scatter(12*60,0.751,s=sz,label="DreamFusion (V100)", edgecolor=edge_c, zorder=scatter_order, marker='d')
# ax.scatter(16/60,0.154,s=sz,label="Point-E (40M, text-only, V100)", edgecolor=edge_c, zorder=scatter_order)
# ax.scatter(1,0.365,s=sz,label="Point-E (40M, V100)", edgecolor=edge_c, zorder=scatter_order)
# ax.scatter(1.2,0.403,s=sz,label="Point-E (300M, V100)", edgecolor=edge_c, zorder=scatter_order)
# ax.scatter(25/60,0.336,s=sz,label="Point-E (300M, text-only, V100)", edgecolor=edge_c, zorder=scatter_order)
# ax.scatter(1.5,0.411,s=sz,label="Point-E (1B, V100)", edgecolor=edge_c, zorder=scatter_order)
# ax.scatter(1.0,0.411,s=sz,label="Shap-E (300M, V100)", edgecolor=edge_c, zorder=scatter_order, marker='s')
# ax.scatter(13/60,0.378,s=sz,label="Shap-E (300M, text-only, V100)", edgecolor=edge_c, zorder=scatter_order, marker='s')

# ax.scatter(52,0.678,s=sz,label="CLIP-Mesh (T4)", edgecolor=edge_c, zorder=scatter_order, marker='D')
# ax.scatter(69/60,0.154,s=sz,label="Point-E (40M, text-only, T4)", edgecolor=edge_c, zorder=scatter_order)
# ax.scatter(3.3,0.365,s=sz,label="Point-E (40M, T4)", edgecolor=edge_c, zorder=scatter_order)
# ax.scatter(1.9,0.336,s=sz,label="Point-E (300M, text-only, T4)", edgecolor=edge_c, zorder=scatter_order)
# ax.scatter(49/60,0.378,s=sz,label="Shap-E (300M, text-only, T4)", edgecolor=edge_c, zorder=scatter_order, marker='s')

# ax.scatter(3.1,0.154,s=sz,label="Point-E (40M, text-only, Jetson)", edgecolor=edge_c, zorder=scatter_order)
# ax.scatter(9.7,0.365,s=sz,label="Point-E (40M, Jetson)", edgecolor=edge_c, zorder=scatter_order)
# ax.scatter(5.2,0.336,s=sz,label="Point-E (300M, text-only, Jetson)", edgecolor=edge_c, zorder=scatter_order)
# ax.scatter(2.5,0.378,s=sz,label="Shap-E (300M, text-only, Jetson)", edgecolor=edge_c, zorder=scatter_order, marker='s')

data = [
    (17, 0.678, 'CLIP-Mesh (V100)', 'D', 'blue'),
    (200*60, 0.786, 'DreamFields (V100)', '^', 'red'),
    (12*60, 0.751, 'DreamFusion (V100)', 'd', 'green'),
    (16/60, 0.154, 'Point-E (40M, text-only, V100)', 'o', 'blue'),
    (1, 0.365, 'Point-E (40M, V100)', 'o', 'red'),
    (1.2, 0.403, 'Point-E (300M, V100)', 'o', 'green'),
    (25/60, 0.336, 'Point-E (300M, text-only, V100)', 'o', 'blue'),
    (1.5, 0.411, 'Point-E (1B, V100)', 'o', 'red'),
    (1.0, 0.411, 'Shap-E (300M, V100)', 's', 'green'),
    (13/60, 0.378, 'Shap-E (300M, text-only, V100)', 's', 'blue'),
    (52, 0.678, 'CLIP-Mesh (T4)', 'D', 'red'),
    (69/60, 0.154, 'Point-E (40M, text-only, T4)', 'o', 'green'),
    (3.3, 0.365, 'Point-E (40M, T4)', 'o', 'blue'),
    (1.9, 0.336, 'Point-E (300M, text-only, T4)', 'o', 'red'),
    (49/60, 0.378, 'Shap-E (300M, text-only, T4)', 's', 'green'),
    (3.1, 0.154, 'Point-E (40M, text-only, Jetson)', 'o', 'blue'),
    (9.7, 0.365, 'Point-E (40M, Jetson)', 'o', 'red'),
    (5.2, 0.336, 'Point-E (300M, text-only, Jetson)', 'o', 'green'),
    (2.5, 0.378, 'Shap-E (300M, text-only, Jetson)', 's', 'blue'),
]

# Get unique shapes
shapes = set([d[3] for d in data])

# Create a colormap based on the number of unique shapes
# cmap = ListedColormap(plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(shapes))))
cmap = ListedColormap(cm.get_cmap('tab10')(np.linspace(0, 1, len(shapes))))

for x, y, label, marker, color in data:
    shape_index = list(shapes).index(marker)
    ax.scatter(x, y, label=label, marker=marker, c=[shape_index], cmap=cmap, vmin=0, vmax=len(shapes), edgecolors='black')

ax.set_yscale('linear')
ax.set_xscale('log')

# plt.yscale('linear')
# plt.xscale('log')

# Set axis labels
# ax.set_ylabel('Linear Scale')
# ax.set_xlabel('Log Scale') # in minutes

# plt.set_ylim(0,1)

plt.ylim(0,1)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True, zorder=1)

# Set title
# ax.set_title('Plot with Logarithmic and Linear Scales')

# Set the legend
plt.legend(loc='lower left', bbox_to_anchor=(-0.09, 1.03),fontsize=12,ncol=2)
# plt.legend(loc='lower right',fontsize=12,ncol=2)

# Set the colorbar to show the color mapping
# plt.colorbar()

# fig = plt.gcf()  # Get the current figure
fig.set_size_inches(8, 4)  # Width: 8 inches, Height: 6 inches
fig.savefig('./quality-plot.png', bbox_inches='tight')
# fig.set_size_inches(8, 10)  # Adjust the size as needed
# plt.savefig('./quality-plot.png')
# Show the plot
plt.show()