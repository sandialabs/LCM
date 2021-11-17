import numpy
from matplotlib.pyplot import *
V0=numpy.loadtxt("V0.txt")
V1=numpy.loadtxt("V1.txt")

rc('font', family='serif', serif='cm10')
rc('font', family='serif', serif='cm10', size=18)
rc('text', usetex=True)

dimensions0 = V0.shape
dimensions1 = V1.shape

steps0 = dimensions0[0]
nodes0 = dimensions0[1]

steps1 = dimensions1[0]
nodes1 = dimensions1[1]

for step in range(1, steps0):
    plot(V0[0,:], V0[step, :], "r-", markersize=6)

for step in range(1, steps1):
    plot(V1[0,:], V1[step, :], "g-", markersize=6)
    
xlim([0,1])
ylim([-200,200])
xlabel('Position', fontsize=14)
ylabel('Velocity', fontsize=14)
xticks(fontsize=14)
yticks(fontsize=14)
#axes().set_yticklabels(["X", "1", "2", "4", "8", "16", "32", "64"])
#axes().set_xticklabels(["Y", "32", "64", "128", "256", "512", "1024"])
#ylim([2, 32])
#legend(['Ideal (slope = -1)','10 load steps'], loc='upper right', fontsize=16)
#axes().set_aspect('equal')
axes().grid(True)
savefig("velocity-2.pdf", bbox_inches='tight')
