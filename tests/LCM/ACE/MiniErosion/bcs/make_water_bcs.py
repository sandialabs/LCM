#!/bin/python
import numpy as np
import os

# Open all files

#f0 = open("time.txt", "w")
#f1 = open("waterHs.txt", "w")
#f2 = open("waterHc.txt", "w")
#f3 = open("waterL.txt", "w")
#f4 = open("waterHA.txt", "w")
#f5 = open("waterH.txt", "w")
#f6 = open("waterA.txt", "w")
#f7 = open("dynP_h3600_Grid0.1_2018MINI.txt", "r")
#f8 = open("Ablufftemp.txt", "r")
f9 = open("Ablufftemp2.txt", "w")



#time = []
#f0.write("{")
#for i in range(101):
#    time.append(i*3600)
#    f0.write(str(time[i]) + ", ")
#f0.seek(0, os.SEEK_END); f0.seek(f0.tell() - 2, os.SEEK_SET)
#f0.truncate()
#f0.write("}")
#f0.close()

#waterHs = []
#f1.write("{")
#for i in range(101):
#    waterHs.append(np.sin(i)-0.8)
#    f1.write(str(waterHs[i]) + ", ")
#f1.seek(0, os.SEEK_END); f1.seek(f1.tell() - 2, os.SEEK_SET)
#f1.truncate()
#f1.write("}")
#f1.close()

#waterHc = []
#f2.write("{")
#for i in range(101):
#    waterHc.append(0.15*np.sin(i)+0.15)
#    f2.write(str(waterHc[i]) + ", ")
#f2.seek(0, os.SEEK_END); f2.seek(f2.tell() - 2, os.SEEK_SET)
#f2.truncate()
#f2.write("}")
#f2.close()

#waterL = []
#f3.write("{")
#for i in range(101):
#    waterL.append(abs(5.0*np.sin(i)+4.0))
#    f3.write(str(waterL[i]) + ", ")
#f3.seek(0, os.SEEK_END); f3.seek(f3.tell() - 2, os.SEEK_SET)
#f3.truncate()
#f3.write("}")
#f3.close()

#waterH = []
#lines = f7.readlines()
#new_array = lines[0][1:-1]
#new_array = new_array.split(',')
#print(new_array)
#f5.write("{")
#for i in range(101):
#    waterH.append(float(new_array[i])+0.55)
#    f5.write(str(waterH[i]) + ", ")
#f5.seek(0, os.SEEK_END); f5.seek(f5.tell() - 2, os.SEEK_SET)
#f5.truncate()
#f5.write("}")
#f5.close()
#print(waterH)

#waterA = []
#f6.write("{")
#for i in range(101):
#    waterA.append(0.15*np.sin(i)+0.30)
#    f6.write(str(waterA[i]) + ", ")
#f6.seek(0, os.SEEK_END); f6.seek(f6.tell() - 2, os.SEEK_SET)
#f6.truncate()
#f6.write("}")
#f6.close()
#print(waterA)

#waterHA = []
#f4.write("{")
#for i in range(101):
#    waterHA.append(waterH[i]+waterA[i])
#    f4.write(str(waterHA[i]) + ", ")
#f4.seek(0, os.SEEK_END); f4.seek(f4.tell() - 2, os.SEEK_SET)
#f4.truncate()
#f4.write("}")
#f4.close()
#print(waterHA)

bluffTemp = []
f9.write("{")
for i in range(101):
    bluffTemp.append(1.5*np.sin(i/2.0)+273.0)
    f9.write(str(bluffTemp[i]) + ", ")
f9.seek(0, os.SEEK_END); f9.seek(f9.tell() - 2, os.SEEK_SET)
f9.truncate()
f9.write("}")
f9.close()


















