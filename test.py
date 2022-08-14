import imageio
images = []
for j in range(199):
    print(j)
    images.append(imageio.imread('job/NREL-m-dyn-yaw-30-400/output/'+str(j+1).zfill(3)+'_animation_xz.png')[400:-400,300:-300])
imageio.mimsave('test-400.gif', images,fps=30)