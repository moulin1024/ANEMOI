import imageio
images = []
for j in range(97):
    print(j)
    images.append(imageio.imread('job/dyn-yaw-8wt-anime/output/'+str(j+1).zfill(3)+'_flowfield_xz.png'))
imageio.mimsave('test-400.gif',images,fps=24)