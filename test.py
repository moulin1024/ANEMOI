import imageio
import sys
case = sys.argv[1]

images = []
for j in range(900):
    print(j)
    images.append(imageio.imread('job/'+case+'/output/'+str(j).zfill(3)+'_flowfield_xz.png'))
imageio.mimsave(case+'_xz_.gif',images,duration=20)

images = []
for j in range(900):
    print(j)
    images.append(imageio.imread('job/'+case+'/output/'+str(j).zfill(3)+'_flowfield_xy.png'))
imageio.mimsave(case+'_xy_.gif',images,duration=20)