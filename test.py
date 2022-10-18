from turtle import distance
import imageio
import sys
# distance = sys.argv[1]

images = []
for j in range(10):
    print(j)
    images.append(imageio.imread('job/7D-yaw0/output/'+str(j).zfill(3)+'_flowfield_xz.png'))
imageio.mimsave('test.gif',images,fps=30)