import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import axes3d

image_dir = '/Users/gene/Teaching/ML4A/ml4a.github.io/images/'

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def relu(x):
	return x * (x > 0)


def save_fig(plt, fname, dpi_=50):
	fpath = image_dir+fname
	plt.savefig(fpath, dpi=dpi_, facecolor='w', edgecolor='w',
	        orientation='portrait', papertype=None, format=None,
	        transparent=False, bbox_inches=None, pad_inches=0.1,
	        frameon=None)

def make_sigmoid_fig():
	min_x = -10.0
	max_x = 10.0
	min_y = -0.0
	max_y = 1.0
	x = np.arange(min_x, max_x, 0.01)
	y = sigmoid(x)
	plt.clf()
	plt.figure(1)
	plt.title('sigmoid', fontsize=26)
	#plt.xlabel('z', fontsize=32)
	#plt.ylabel('$\sigma(z)$', fontsize=32)
	plt.plot(x, y, 'k', linewidth=4);
	plt.axis([min_x, max_x, min_y, max_y])
	plt.axhline(0, color='grey')
	plt.axvline(0, color='grey')
	plt.text(-9.6, 0.84, r'$\sigma(z) = \frac{1}{1 + e^{-z}}$', fontsize=36)
	plt.grid(True)
#	plt.show()
	print "save it"
	save_fig(plt, "sigmoid.png")


def make_relu_fig():
	min_x = -10.0
	max_x = 10.0
	min_y = -0.0
	max_y = 10.0
	x = np.arange(min_x, max_x, 0.01)
	y = relu(x)
	plt.clf()
	plt.figure(1)
	plt.title('ReLU', fontsize=24)
	#plt.xlabel('z', fontsize=32)
	#plt.ylabel('$\sigma(z)$', fontsize=32)
	plt.plot(x, y, 'k', linewidth=4);
	plt.axis([min_x, max_x, min_y, max_y])
	plt.axhline(0, color='grey')
	plt.axvline(0, color='grey')
	plt.text(-9.6, 8.4, r'$R(z) = max(0,\ \ z)$', fontsize=28)
	plt.grid(True)
#	plt.show()
	save_fig(plt, "relu.png")


make_relu_fig()


#98.5, 104.5 pressure
#humid 25 -> 70

def make_2d_lin_classifier():
	x_pos = [60, 62, 51, 55]
	y_pos = [98.6, 99.9, 97.6, 100.2]
	x_neg = [29, 40, 39, 46]
	y_neg = [101.7, 101.1, 103.2, 102.1]
	plt.plot(x_pos, y_pos, '+', mew=6, ms=25, c='red')
	plt.plot(x_neg, y_neg, '_', mew=6, ms=25, c='blue')
	plt.xlabel("humidity (%)")
	plt.ylabel("pressure (kPa)")
	plt.title("Rain")
	plt.grid(True)
	#plt.show()
	save_fig(plt, "lin_classifier_2d.png", 72)

	print "|**Humidity (%)**|**Pressure (kPa)**|**Rain?**|\n|==|==|==|"
	for i in range(0,4):
		print "|%d|%0.1f|%s|"%(x_neg[i], y_neg[i], '-')
		print "|%d|%0.1f|%s|"%(x_pos[i], y_pos[i], '+')



def make_3d_lin_classifier():
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	x_pos = [60, 62, 51, 55]
	y_pos = [98.6, 99.9, 97.6, 100.2]
	z_pos = [0.7, 0.9, 0.83, 0.88]
	x_neg = [29, 40, 39, 46]
	y_neg = [101.7, 101.1, 103.2, 102.1]
	z_neg = [0.2, 0.11, 0.19, 0.3]


	ax = fig.gca(projection='3d')
	X = np.arange(19, 90, 2)
	Y = np.arange(87, 114, 2)
	X, Y = np.meshgrid(X, Y)
	Z = -0.8 + 0.01*X + 0.01*Y
	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

	ax.scatter(x_pos, y_pos, z_pos, c='r', s=320, lw=5, marker='+')
	ax.scatter(x_neg, y_neg, z_neg, c='g', s=320, lw=5, marker='_')

	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	plt.draw()
	plt.show()


#make_3d_lin_classifier()

make_sigmoid_fig()


#os.system('mkdir '+image_dir+'temp/')

#ax = fig.add_subplot(111, projection='3d')
#X, Y, Z = axes3d.get_test_data(0.1)
#ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)



#for i, angle in enumerate(range(0, 360, 5)):
#	ax.view_init(30, angle)
#	plt.draw()
	#save_fig(plt, "temp/rot_%04d.png"%i, 72)

#os.system('ffmpeg -i '+image_dir+'temp/rot_%4d.png -c:v libx264 -r 30 -pix_fmt yuv420p '+image_dir+'out.mp4')
#os.system('rm -rf '+image_dir+'temp')
