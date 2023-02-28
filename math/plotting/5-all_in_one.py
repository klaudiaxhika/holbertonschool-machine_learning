#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.title("All in One",fontsize=10)

plt.subplot(321)
plt.plot(y0,'r')
plt.xlim(0,10)
plt.ylim(0,1050)
plt.yticks([0,500,1000],fontsize=10)
plt.xticks(fontsize=10)

plt.subplot(322)
plt.scatter(x1, y1, c='m')
plt.title("Men's Height vd Weight",fontsize=8)
plt.ylabel("Weight (lbs)",fontsize=8)
plt.xlabel("Height (in)",fontsize=8)
plt.xticks([60,70,80],fontsize=10)
plt.yticks([170,180,190],fontsize=10)

plt.subplot(323)
plt.plot(x3, y31)
plt.title('Exponential Decay of C-14',fontsize=8)
plt.xlabel('Time (years)',fontsize=8)
plt.ylabel('Fraction Remaining',fontsize=8)
plt.xlim(0,30000)
plt.yscale("log")
plt.xticks([0,10000,20000],fontsize=10)
plt.yticks(fontsize=10)

plt.subplot(324)
plt.plot(x3,y31,'r--', label='C-14')
plt.plot(x3,y32,'g', label='Ra-226')
plt.xlim(0,20000)
plt.ylim(0,1)
plt.xlabel('Time (years)',fontsize=8)
plt.ylabel('Fraction Remaining',fontsize=8)
plt.title('Exponential Decay of Radioactive Elements',fontsize=8)
plt.yticks([0.0,0.5,1.0],fontsize=10)
plt.xticks([0,5000,10000,15000,20000],fontsize=10)
plt.legend()

binwidth=10

plt.subplot(313)
plt.hist(student_grades, edgecolor='black', bins=range(0, 100 + binwidth, binwidth))
plt.xlim(0,100)
plt.ylim(0,30)
plt.xlabel('Grades',fontsize=8)
plt.ylabel('Number of Students',fontsize=8)
plt.title('Project A',fontsize=8)
plt.xticks([0,10,20,30,40,50,60,70,80,90,100],fontsize=10)
plt.yticks([0,10,20,30],fontsize=10)

plt.tight_layout()
plt.show()

