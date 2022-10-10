from turtle import distance
import imageio
import sys
distance = sys.argv[1]

images = []
for j in range(200):
    print(j)
    images.append(imageio.imread('job/'+str(distance)+'D-yaw0/output/'+str(j).zfill(3)+'_flowfield_xz.png'))
imageio.mimsave(str(distance)+'D-yaw0.gif',images,fps=30)