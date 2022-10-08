import imageio
images = []
for j in range(10):
    print(j)
    images.append(imageio.imread('job/NREL-m/output/'+str(j).zfill(3)+'_flowfield_xz.png'))
imageio.mimsave('test.gif',images,fps=10)