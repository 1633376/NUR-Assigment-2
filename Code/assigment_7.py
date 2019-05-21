import h5py
import matplotlib.pyplot as plt
import numpy as np
import mathlib.quadtree as qt

particles_data = h5py.File('./Data/colliding.hdf5')['PartType4']
particles_pos = particles_data['Coordinates']
particles_masses = particles_data['Masses']



tree = qt.QuadTree((0,0), 200, 12)

for particle in particles_pos:
    tree.add_boddy(particle)
tree.plot()


#plt.scatter(particles_pos[:,0],particles_pos[:,1],s=1)
#plt.show()

#print(particles_df['PartType4']['Coordinates'][0])