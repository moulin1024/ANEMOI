from turtle import distance
import imageio
import sys
# distance = sys.argv[1]

images = []
for j in range(360):
    print(j)
    images.append(imageio.imread('job/8wt-m-mgm/output/'+str(j).zfill(3)+'_flowfield_yz.png'))
imageio.mimsave('test.gif',images,fps=30)
