
from PIL import Image
import numpy as np
import scipy.io
import tensorflow as tf
from functools import reduce
import scipy.misc

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

#*******************LOAD PICTURE*********************

def loadpic():
	
	imgcon = Image.open("content.jpg")
	imgsty = Image.open("style.jpg")
	
	datacon = imgcon.getdata()
	datasty = imgsty.getdata()
	
	datacon = np.reshape(datacon, (300, 300, 3))
	datasty = np.reshape(datasty, (300, 300, 3))
	
	datacon = np.array(datacon)
	datasty = np.array(datasty)
	'''
	
	datacon = scipy.misc.imread("content.jpg").astype(np.float)
	datasty = scipy.misc.imread("style.jpg").astype(np.float)
	'''
	return datacon, datasty
	
#********************CONSTRUCT VGG***********************

def vggnet(input_image):

	predata = scipy.io.loadmat('imagenet-vgg-verydeep-19.mat')
	weights = predata['layers'][0]
	
	def conv_layer(bottom, num):
		
		kernels, bias = weights[num][0][0][0][0]
		kernels = np.transpose(kernels, (1, 0, 2, 3))
		bias = bias.reshape(-1)
		
		cvnt = tf.nn.conv2d(bottom, tf.constant(kernels), strides = (1, 1, 1, 1), padding = 'SAME')
		ret = tf.nn.bias_add(cvnt, bias)
		return ret
		
	def pool_layer(bottom):
		
		ret = tf.nn.avg_pool(bottom, ksize = (1, 2, 2, 1), strides = (1, 2, 2, 1), padding = 'SAME')
		return ret
		
	def relu_layer(bottom):
		
		ret = tf.nn.relu(bottom)
		return ret
	
	vgg = {}
	
	vgg['conv1_1'] = conv_layer(input_image, 0)
	vgg['relu1_1'] = relu_layer(vgg['conv1_1'])
	vgg['conv1_2'] = conv_layer(vgg['relu1_1'], 2)
	vgg['relu1_2'] = relu_layer(vgg['conv1_2'])
	vgg['pool1'] = pool_layer(vgg['relu1_2'])
	
	vgg['conv2_1'] = conv_layer(vgg['pool1'], 5)
	vgg['relu2_1'] = relu_layer(vgg['conv2_1'])
	vgg['conv2_2'] = conv_layer(vgg['relu2_1'], 7)
	vgg['relu2_2'] = relu_layer(vgg['conv2_2'])
	vgg['pool2'] = pool_layer(vgg['relu2_2'])
	
	
	vgg['conv3_1'] = conv_layer(vgg['pool2'], 10)
	vgg['relu3_1'] = relu_layer(vgg['conv3_1'])
	vgg['conv3_2'] = conv_layer(vgg['relu3_1'], 12)
	vgg['relu3_2'] = relu_layer(vgg['conv3_2'])
	vgg['conv3_3'] = conv_layer(vgg['relu3_2'], 14)
	vgg['relu3_3'] = relu_layer(vgg['conv3_3'])
	vgg['conv3_4'] = conv_layer(vgg['relu3_3'], 16)
	vgg['relu3_4'] = relu_layer(vgg['conv3_4'])
	vgg['pool3'] = pool_layer(vgg['relu3_4'])
	
	vgg['conv4_1'] = conv_layer(vgg['pool3'], 19)
	vgg['relu4_1'] = relu_layer(vgg['conv4_1'])
	vgg['conv4_2'] = conv_layer(vgg['relu4_1'], 21)
	vgg['relu4_2'] = relu_layer(vgg['conv4_2'])
	vgg['conv4_3'] = conv_layer(vgg['relu4_2'], 23)
	vgg['relu4_3'] = relu_layer(vgg['conv4_3'])
	vgg['conv4_4'] = conv_layer(vgg['relu4_3'], 25)
	vgg['relu4_4'] = relu_layer(vgg['conv4_4'])
	vgg['pool4'] = pool_layer(vgg['relu4_4'])
	
	vgg['conv5_1'] = conv_layer(vgg['pool4'], 28)
	vgg['relu5_1'] = relu_layer(vgg['conv5_1'])
	vgg['conv5_2'] = conv_layer(vgg['relu5_1'], 30)
	vgg['relu5_2'] = relu_layer(vgg['conv5_2'])
	vgg['conv5_3'] = conv_layer(vgg['relu5_2'], 32)
	vgg['relu5_3'] = relu_layer(vgg['conv5_3'])
	vgg['conv5_4'] = conv_layer(vgg['relu5_3'], 34)
	vgg['relu5_4'] = relu_layer(vgg['conv5_4'])
	
	return vgg
	
#***********************STYLIZE IMAGE************************

def main():
	
	infodata = scipy.io.loadmat('imagenet-vgg-verydeep-19.mat')
	mean = infodata['normalization'][0][0][0]
	mean = np.mean(mean, axis=(0, 1))
	
	datacon, datasty = loadpic()
	shape = (1,) + datacon.shape
	
	infocon = {}
	infosty = {}
	compute_content = 'relu4_2'
	compute_style = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
	weightcon = 5
	weightsty = 0.2
	iterations = 1000
	
	
	g = tf.Graph()
	with g.as_default(), tf.Session() as sess:
		tempimage = tf.placeholder('float', shape = shape)
		tempvgg = vggnet(tempimage)
		qqq = np.array([datacon - mean])
		infocon[compute_content] = tempvgg[compute_content].eval(feed_dict = {tempimage: qqq})
		
	g = tf.Graph()
	with g.as_default(), tf.Session() as sess:
		tempimage = tf.placeholder('float', shape = shape)
		tempvgg = vggnet(tempimage)
		ppp = np.array([datasty - mean])
		for layer in compute_style:
			temp_style = tempvgg[layer].eval(feed_dict = {tempimage: ppp})
			temp_style = np.reshape(temp_style, (-1, temp_style.shape[3]))
			infosty[layer] = np.matmul(temp_style.T, temp_style) / temp_style.size
			
	g = tf.Graph()
	with g.as_default(), tf.Session() as sess:
		#initial = np.random.normal(size = shape, scale = 0.2)
		ini = tf.random_normal(shape) * 0.1
		via = tf.Variable(ini)
		center_vgg = vggnet(via)
		
		losscon = weightcon * (tf.nn.l2_loss(center_vgg[compute_content] - infocon[compute_content]) * 2 / infocon[compute_content].size)
		
		#losssty = 0.0
		losssty = []
		for layer in compute_style:
			'''
			ttt = np.array([initial])
			temp_center = center_vgg[layer].eval(feed_dict = {via: ttt[0]})
			temp_center = np.reshape(temp_center, (-1, temp_center.shape[3]))
			center_gram = np.matmul(temp_center.T, temp_center) / temp_center.size
			'''
			temp_center = center_vgg[layer]
			_, height, width, number = map(lambda i: i.value, temp_center.get_shape())
			sss = height * width * number
			tempnet = tf.reshape(temp_center, (-1, number))
			center_gram = tf.matmul(tf.transpose(tempnet), tempnet) / sss
			
			style_gram = infosty[layer]
			#jjj = tf.cast(tf.nn.l2_loss(center_gram - style_gram) * 2 / style_gram.size, tf.float32)
			#losssty = tf.add(losssty, jjj)
			uuu = tf.nn.l2_loss(center_gram - style_gram) * 2 / style_gram.size
			losssty.append(uuu)
		fi_losssty = reduce(tf.add, losssty)
		
		total_loss = losscon + weightsty * fi_losssty
		
		desceding = tf.train.AdamOptimizer(0.2).minimize(total_loss)
		
		with tf.Session() as sess:
			sess.run(tf.initialize_all_variables())
			for i in range(iterations):
				desceding.run()
				if i == iterations - 1:
					this_loss = total_loss.eval()
					output = via.eval()
					#output = np.reshape(output, (300, 300, 3))
					output = output.reshape(shape[1:])
					output = output + mean
					yield (
						output
						)
					#output = np.uint8(output*255)
					#!output = np.clip(output, 0, 255).astype(np.uint8)
					#image_output = Image.fromarray(output)
					#scipy.misc.imsave(save_path, image_output)
					#!scipy.misc.imsave(save_path, output)
					
					#image_output.show()

					
#**************************SAVE RESULT*****************************
					
def output():
	save_path = "./rst.jpg"
	for via in main():
		via = np.clip(via, 0, 255).astype(np.uint8)
		scipy.misc.imsave(save_path, via)
		
output()