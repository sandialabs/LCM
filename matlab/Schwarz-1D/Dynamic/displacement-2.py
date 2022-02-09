import numpy
from matplotlib.pyplot import *
U0=numpy.loadtxt("U0.txt")
U1=numpy.loadtxt("U1.txt")

rc('font', family='serif', serif='cm10')
rc('font', family='serif', serif='cm10', size=18)
rc('text', usetex=True)

dimensions0 = U0.shape
dimensions1 = U1.shape

steps0 = dimensions0[0]
nodes0 = dimensions0[1]

steps1 = dimensions1[0]
nodes1 = dimensions1[1]

for step in range(1, steps0):
    plot(U0[0,:], U0[step, :], "r-", markersize=6)

for step in range(1, steps1):
    plot(U1[0,:], U1[step, :], "g-", markersize=6)
    
xlim([0,1])
ylim([-0.01,0.01])
xlabel('Position', fontsize=14)
ylabel('Displacement', fontsize=14)
xticks(fontsize=14)
yticks(fontsize=14)
#legend(['Implicit','Explicit'], loc='upper right', fontsize=16)
#axes().set_yticklabels(["X", "1", "2", "4", "8", "16", "32", "64"])
#axes().set_xticklabels(["Y", "32", "64", "128", "256", "512", "1024"])
#axes().set_aspect('equal')
axes().grid(True)
savefig("displacement-2.pdf", bbox_inches='tight')
