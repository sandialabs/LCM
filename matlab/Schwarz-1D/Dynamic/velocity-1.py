import numpy
from matplotlib.pyplot import *
V0=numpy.loadtxt("V.txt")

rc('font', family='serif', serif='cm10')
rc('font', family='serif', serif='cm10', size=18)
rc('text', usetex=True)

dimensions0 = V0.shape

steps0 = dimensions0[0]
nodes0 = dimensions0[1]

for step in range(1, steps0):
    plot(V0[0,:], V0[step, :], "r.-", markersize=6)

xlabel('Position', fontsize=22)
ylabel('Velocity', fontsize=22)
#axes().set_yticklabels(["X", "1", "2", "4", "8", "16", "32", "64"])
#axes().set_xticklabels(["Y", "32", "64", "128", "256", "512", "1024"])
#ylim([2, 32])
#legend(['Ideal (slope = -1)','10 load steps'], loc='upper right', fontsize=16)
#axes().set_aspect('equal')
axes().grid(True)
savefig("velocity-1.pdf", bbox_inches='tight')
