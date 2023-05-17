from turtle import distance
import imageio
import sys
# distance = sys.argv[1]

images = []
for j in range(200):
    print(j)
    images.append(imageio.imread('job/HR1-m-0/output/'+str(j).zfill(3)+'_flowfield_yz.png'))
imageio.mimsave('test-0.gif',images,fps=30)
