
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = 17,14

x, y, z,dx,dy,dz = np.loadtxt('field2.irreg.txt',
                              skiprows=6,
                              unpack=True)


# =============================================================================
# using sqaure root to differentiate between displacements and useful for assigning colors
# =============================================================================
z=np.sqrt(dx*dx+dy*dy)

# =============================================================================
# Skipping 3 rows for smoother plot  
# =============================================================================
slice_interval = 4
skip = (slice(None, None, slice_interval))
Quiver=plt.quiver(x[skip],y[skip],dx[skip],dy[skip],z[skip]
                  ,units= 'xy'
                  ,angles='uv'
                  ,scale=6
                  ,pivot='tail'
                  ,cmap=plt.cm.coolwarm  
                  )
plt.title(" Movements of water particles caused by winds",fontsize=14)
plt.xlabel("X Equivalent vectors", fontsize=12)
plt.ylabel("Y Equivalent vectors", fontsize=12)

#plt.grid()

cbar= plt.colorbar(Quiver,shrink=0.9, aspect=30)
cbar.set_ticks([0.0, 0.5, 0.99])
cbar.set_ticklabels(["Low", "Medium", "High"])
cbar.set_label('Flow Velocity', fontsize=12)

# plt.show()
# =============================================================================
# 
plt.draw()
plt.savefig('Output.png')
# =============================================================================

