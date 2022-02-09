import numpy
from matplotlib.pyplot import *
A0=numpy.loadtxt("A.txt")

rc('font', family='serif', serif='cm10')
rc('font', family='serif', serif='cm10', size=18)
rc('text', usetex=True)

dimensions0 = A0.shape

steps0 = dimensions0[0]
nodes0 = dimensions0[1]

for step in range(1, steps0):
    plot(A0[0,:], A0[step, :], "r.-", markersize=6)

xlabel('Position', fontsize=22)
ylabel('Acceleration', fontsize=22)
#axes().set_yticklabels(["X", "1", "2", "4", "8", "16", "32", "64"])
#axes().set_xticklabels(["Y", "32", "64", "128", "256", "512", "1024"])
#ylim([2, 32])
#legend(['Ideal (slope = -1)','10 load steps'], loc='upper right', fontsize=16)
#axes().set_aspect('equal')
axes().grid(True)
savefig("acceleration-1.pdf", bbox_inches='tight')
