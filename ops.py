
from glob import glob 
import os
import tensorflow as tf
import numpy as np
import cv2 
def block_patch(input, margin=5):
	shape = input.get_shape().as_list()
	
	#create patch in random size
	pad_size = tf.random_uniform([2], minval=15, maxval=25, dtype=tf.int32)
	patch = tf.zeros([pad_size[0], pad_size[1], shape[-1]], dtype=tf.float32)
 	
	h_ = tf.random_uniform([1], minval=margin, maxval=shape[0]-pad_size[0]-margin, dtype=tf.int32)[0]
	w_ = tf.random_uniform([1], minval=margin, maxval=shape[1]-pad_size[1]-margin, dtype=tf.int32)[0]
	
	padding = [[h_, shape[0]-h_-pad_size[0]], [w_, shape[1]-w_-pad_size[1]], [0, 0]]
	padded = tf.pad(patch, padding, "CONSTANT", constant_values=1)

	coord = h_, w_

	res = tf.multiply(input, padded)

	return res, padded, coord, pad_size

#function to get training data
def load_train_data(args):
	paths = os.path.join(args.data, "img_align_celeba/*.jpg")
	data_count = len(glob(paths))
	
	filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(paths))

	image_reader = tf.WholeFileReader()
	_, image_file = image_reader.read(filename_queue)
	images = tf.image.decode_jpeg(image_file, channels=3)
	

	#input image range from -1 to 1
	#center crop 32x32 since raw images are not center cropped.
	# images = tf.image.central_crop(images, 0.5)
	images = tf.image.resize_images(images ,[args.input_height, args.input_width])
	images = tf.image.convert_image_dtype(images, dtype=tf.float32) / 127.5 - 1
	
	orig_images = images
	images, mask, coord, pad_size = block_patch(images, margin=args.margin)
	mask = tf.reshape(mask, [args.input_height, args.input_height, 3])
	
	#flip mask values
	mask = -(mask - 1)
	images += mask

	orig_imgs, perturbed_imgs, mask, coord, pad_size = tf.train.shuffle_batch([orig_images, images, mask, coord, pad_size],
																			  batch_size=args.batch_size,
																			  capacity=args.batch_size*2,
																			  min_after_dequeue=args.batch_size
																			 )


	return orig_imgs, perturbed_imgs, mask, coord, pad_size, data_count

def load_test_data(args):
	paths = glob("./data/test/*.jpg")
	data_count = len(paths)

	filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(paths))

	image_reader = tf.WholeFileReader()
	_, image_file = image_reader.read(filename_queue)
	images = tf.image.decode_jpeg(image_file, channels=3)
	

	#input image range from -1 to 1
	# uncomment to center crop 
	# images = tf.image.central_crop(images, 0.5)
	images = tf.image.resize_images(images ,[args.input_height, args.input_width])
	images = tf.image.convert_image_dtype(images, dtype=tf.float32) / 127.5 - 1
	
	orig_images = images
	images, mask, coord, pad_size = block_patch(images, margin=args.margin)
	mask = tf.reshape(mask, [args.input_height, args.input_height, 3])

	#flip mask values
	mask = -(mask - 1)
	images += mask

	orig_imgs, mask, test_imgs = tf.train.batch([orig_images, mask, images],
												batch_size=args.batch_size,
												capacity=args.batch_size,
											    )


	return orig_imgs, test_imgs, mask, data_count


# #function to save images in tile
# #comment this function block if you don't have opencv
# def img_tile(epoch, args, imgs, aspect_ratio=1.0, tile_shape=None, border=1, border_color=0):
# 	if imgs.ndim != 3 and imgs.ndim != 4:
# 		raise ValueError('imgs has wrong number of dimensions.')
# 	n_imgs = imgs.shape[0]

# 	tile_shape = None
# 	# Grid shape
# 	img_shape = np.array(imgs.shape[1:3])
# 	if tile_shape is None:
# 		img_aspect_ratio = img_shape[1] / float(img_shape[0])
# 		aspect_ratio *= img_aspect_ratio
# 		tile_height = int(np.ceil(np.sqrt(n_imgs * aspect_ratio)))
# 		tile_width = int(np.ceil(np.sqrt(n_imgs / aspect_ratio)))
# 		grid_shape = np.array((tile_height, tile_width))
# 	else:
# 		assert len(tile_shape) == 2
# 		grid_shape = np.array(tile_shape)

# 	# Tile image shape
# 	tile_img_shape = np.array(imgs.shape[1:])
# 	tile_img_shape[:2] = (img_shape[:2] + border) * grid_shape[:2] - border

# 	# Assemble tile image
# 	tile_img = np.empty(tile_img_shape)
# 	tile_img[:] = border_color
# 	for i in range(grid_shape[0]):
# 		for j in range(grid_shape[1]):
# 			img_idx = j + i*grid_shape[1]
# 			if img_idx >= n_imgs:
# 				# No more images - stop filling out the grid.
# 				break
# 			img = imgs[img_idx]
# 			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 			yoff = (img_shape[0] + border) * i
# 			xoff = (img_shape[1] + border) * j
# 			tile_img[yoff:yoff+img_shape[0], xoff:xoff+img_shape[1], ...] = img

# 	cv2.imwrite(args.images_path+"/img_"+str(epoch) + ".jpg", (tile_img + 1)*127.5)
