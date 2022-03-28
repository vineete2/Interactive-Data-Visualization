#%%

from matplotlib import pylab as plt, colors
import numpy as np
import math
dataArr = []
list_16bits = []
with open('slice150.raw', "rb") as f:
    dataArr = f.read()
#read binary 16-bit data
    list_16bits = [dataArr[i + 1] << 8 | dataArr[i] for i in range(0, len(dataArr), 2)]
print(list_16bits)
print(type(list_16bits))

array=np.reshape(list_16bits, (512,512))
from copy import copy, deepcopy
data = deepcopy(array)
#data = np.array(data)[np.indices.astype(int)]
#print(array)
#print(type(array))
#plt.imshow(array,cmap=plt.cm.bone)
#plt.title('Original Scan')
#plt.savefig('1_Original.png', dpi=300, bbox_inches='tight')
#plt.show()

#%%


array.shape
#profile line through line 256 of this 2D data set
fig1 = plt.figure()
myprof=plt.plot(array[255,])
plt.xlabel('data at line 256')
plt.ylabel('data values')
plt.title('profile line through line 256')
plt.legend(['line 256'])

plt.savefig('profileline256.png', dpi=300, bbox_inches='tight')
plt.close(fig1)
#%%

#mean value and the variance value of this 2D data set
print('Mean of 2D data set',array.mean())
print('Variance of 2D data set',array.var())


#%%
from matplotlib import pylab as plt3, colors
from scipy import stats
#dist = stats.norm()

fig2 = plt.figure()
#histogram of this 2D data set
#density = stats.gaussian_kde(array)
#x=plt.hist(array,bins=30)
myhist=plt.hist(array.ravel(),bins=30,histtype=u'step', density=True)
#gkde = stats.gaussian_kde()
#plt.plot(myhist,dist.pdf(myhist))
#plt.colorbar()
#plt.show()
#plt.plot(x, density(x))
#mu, sigma = scipy.stats.norm.fit(array)
#best_fit_line = scipy.stats.norm.pdf(x, mu, sigma)
#plt.plot(x, best_fit_line)

plt.title('histogram of 2D data set')
plt.xlabel('x values')
plt.ylabel('y values')

plt.savefig('hist.png', dpi=300, bbox_inches='tight')

plt.close(fig2)
#%%
from matplotlib import pylab as plt
#Rescale values to range between 0 and 255 using a linear transformation

print('Data Type: %s' % array.dtype)
print('Min: %.3f, Max: %.3f' % (array.min(), array.max()))
array = array.astype('float64')


array /= array.max()/255

print('Min: %.3f, Max: %.3f' % (array.min(), array.max()))
#print(array)
plt.imshow(array,cmap=plt.cm.bone)
plt.title('Rescaled and linear transformed the image')
plt.savefig('linearTransf.png', dpi=300, bbox_inches='tight')

#%%

#Different transformation using log
def ln(x):
    n = 2.0
    return n * ((x ** (1/n)) - 1)

out2array=ln(array)

plt.imshow(out2array#,norm=colors.PowerNorm(gamma=3)
           ,cmap=plt.cm.bone)
plt.title('Rescaled and log transformed the image')
plt.savefig('NonlinearTransf.png', dpi=300, bbox_inches='tight')

#%%

#create boxcar filter and input data
boxcar=np.array([1/121] * 121).reshape((11, 11))



#%%
sub1 = data[0:11,0:11]
rows, cols = (len(data), len(data))

#list1=[]
listsum = []

for r in range(rows-11):
    for c in range(cols-11):
        sub1 = data[r:r+11,c:c+11]
        #smoothdata=boxcar*sub1
        #list1.append(smoothdata[5][5])
        #print(list1)
        listsum.append(np.sum(sub1)/121)
    #print (sub1[r][c])

#smoothdata=boxcar*sub1
#op=smoothdata[5][5]

#print(len(list1))
#shape=int(math.sqrt(len(list1)))
#%%
#output1=np.reshape(list1, (shape,shape))
#plt.imshow(output1,cmap=plt.cm.bone)
#plt.title('After applying 11x11 boxcar smoothing filter')
#plt.savefig('3_boxcar.png', dpi=300, bbox_inches='tight')

fig4 = plt.figure()
print(len(listsum))
shape=int(math.sqrt(len(listsum)))
#%%
output1=np.reshape(listsum, (shape,shape))
plt.imshow(output1,cmap=plt.cm.bone)
plt.title('After applying 11x11 boxcar smoothing filter')
plt.savefig('3_boxcar.png', dpi=300, bbox_inches='tight')
plt.close(fig4)
#%%
#11x11 median filter
median=np.array([1] * 121).reshape((11, 11))
sub2 = data[0:11,0:11]
rows, cols = (len(data), len(data))

listMedian=[]

for r in range(rows-11):
    for c in range(cols-11):
        sub2 = data[r:r+11,c:c+11]
        smoothMedian=median*sub2
        listMedian.append(np.median(smoothMedian))
        #print(list1)
    #print (sub1[r][c])

#smoothdata=boxcar*sub1
#op=smoothdata[5][5]

print(len(listMedian))
shape=int(math.sqrt(len(listMedian)))

outputMedian=np.reshape(listMedian, (shape,shape))
plt.imshow(outputMedian,cmap=plt.cm.bone)
plt.title('After applying 11x11 median filter')
plt.savefig('4_median.png', dpi=300, bbox_inches='tight')