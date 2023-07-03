from turtle import distance
import imageio
import sys
case = sys.argv[1]

images = []
for j in range(40):
    print(j)
    images.append(imageio.imread('job/'+case+'/output/'+str(j).zfill(3)+'_flowfield_yz.png'))
imageio.mimsave(case+'.gif',images,fps=24)
