import h5py
import matplotlib.pyplot as plt
import numpy as np
import mathlib.quadtree as qt

particles_data = h5py.File('./Data/colliding.hdf5')['PartType4']
particles_pos = np.array(particles_data['Coordinates'])
particles_masses = np.array(particles_data['Masses'])

particle_info = np.zeros((len(particles_masses),4))
particle_info[:,3] = particles_masses
particle_info[:,2] = particles_pos[:,2]
particle_info[:,1] = particles_pos[:,1]
particle_info[:,0] = particles_pos[:,0]

tree = qt.QuadTree((0,0), 150, 12)

for particle in particle_info:
    tree.add_boddy(particle)
plt.scatter(particle_info[100,0],particle_info[100,1],c='red',s=100)
tree.plot()
tree.print_moments(particle_info[100,0],particle_info[100,1])



#plt.scatter(particles_pos[:,0],particles_pos[:,1],s=1)
#plt.show()

#print(particles_df['PartType4']['Coordinates'][0])