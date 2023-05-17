from turtle import distance
import imageio
import sys
# distance = sys.argv[1]

images = []
for j in range(100):
    print(j)
    images.append(imageio.imread('./fig/fx_'+str(j)+'.png'))
imageio.mimsave('test-0.gif',images,fps=30)
