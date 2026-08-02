"""One-off script: rotates ../waypoints_2.csv by a fixed 6-degree offset and saves ../centre_line_2.csv."""
import numpy as np
import matplotlib.pyplot as plt
import math
file_csv = np.loadtxt('../waypoints_2.csv',delimiter=',')
theta = 6*(math.pi/180.)
x,y = file_csv[:,0]*math.cos(theta)-file_csv[:,1]*math.sin(theta),file_csv[:,1]*math.cos(theta)+file_csv[:,0]*math.sin(theta)
plt.plot(file_csv[:,0],file_csv[:,1])
plt.plot(x,y)
new_file_csv = np.array([x,y]).T
np.savetxt('../centre_line_2.csv',new_file_csv,delimiter=',')
plt.show()
