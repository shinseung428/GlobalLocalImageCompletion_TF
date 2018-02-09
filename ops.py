
from glob import glob 
import os
import tensorflow as tf
import numpy as np
import cv2 
def block_patch(input, patch_size=32, margin=5):
	shape = input.get_shape().as_list()
	#for training images
	if len(shape) == 3:
		patch = tf.zeros([patch_size, patch_size, shape[-1]], dtype=tf.float32)
	 	larger_patch = tf.zeros([patch_size+margin, patch_size+margin, shape[-1]], dtype=tf.float32)

		rand_num = tf.random_uniform([2], minval=0, maxval=shape[0]-patch_size, dtype=tf.int32)
		h_, w_ = rand_num[0], rand_num[1]

		padding = [[h_, shape[0]-h_-patch_size], [w_, shape[1]-w_-patch_size], [0, 0]]
		padded = tf.pad(patch, padding, "CONSTANT", constant_values=1)

		coord = h_, w_

		res = tf.multiply(input, padded)
	#for generated images
	else:
		patch = tf.zeros([patch_size, patch_size, shape[-1]], dtype=tf.float32)
		larger_patch = tf.zeros([patch_size+margin, patch_size+margin, shape[-1]], dtype=tf.float32)
	 
		res = []
		for idx in range(0,shape[0]):
			rand_num = tf.random_uniform([2], minval=0, maxval=shape[0]-patch_size, dtype=tf.int32)
			h_, w_ = rand_num[0], rand_num[1]

			padding = [[h_, shape[0]-h_-patch_size], [w_, shape[1]-w_-patch_size], [0, 0]]
			padded = tf.pad(patch, padding, "CONSTANT", constant_values=1)

			coord = h_, w_

			res.append(tf.multiply(input[idx], padded))
		res = tf.stack(res)

	return res, padded, coord

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
	images = tf.image.central_crop(images, 0.5)
	images = tf.image.resize_images(images ,[args.input_height, args.input_width])
	images = tf.image.convert_image_dtype(images, dtype=tf.float32) / 127.5 - 1
	
	orig_images = images
	images, mask, coord = block_patch(images, patch_size=args.patch_size)
	mask = tf.reshape(mask, [args.input_height, args.input_height, 1])
	
	#flip mask values
	mask = -(mask - 1)
	
	orig_imgs, perturbed_imgs, mask, coord = tf.train.shuffle_batch([orig_images, images, mask, coord],
																		  batch_size=args.batch_size,
																		  capacity=args.batch_size*2,
																		  min_after_dequeue=args.batch_size
																		 )


	return orig_imgs, perturbed_imgs, mask, coord, data_count


#function to save images in tile
#comment this function block if you don't have opencv
def img_tile(epoch, args, imgs, aspect_ratio=1.0, tile_shape=None, border=1, border_color=0):
	if imgs.ndim != 3 and imgs.ndim != 4:
		raise ValueError('imgs has wrong number of dimensions.')
	n_imgs = imgs.shape[0]

	tile_shape = None
	# Grid shape
	img_shape = np.array(imgs.shape[1:3])
	if tile_shape is None:
		img_aspect_ratio = img_shape[1] / float(img_shape[0])
		aspect_ratio *= img_aspect_ratio
		tile_height = int(np.ceil(np.sqrt(n_imgs * aspect_ratio)))
		tile_width = int(np.ceil(np.sqrt(n_imgs / aspect_ratio)))
		grid_shape = np.array((tile_height, tile_width))
	else:
		assert len(tile_shape) == 2
		grid_shape = np.array(tile_shape)

	# Tile image shape
	tile_img_shape = np.array(imgs.shape[1:])
	tile_img_shape[:2] = (img_shape[:2] + border) * grid_shape[:2] - border

	# Assemble tile image
	tile_img = np.empty(tile_img_shape)
	tile_img[:] = border_color
	for i in range(grid_shape[0]):
		for j in range(grid_shape[1]):
			img_idx = j + i*grid_shape[1]
			if img_idx >= n_imgs:
				# No more images - stop filling out the grid.
				break
			img = imgs[img_idx]
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

			yoff = (img_shape[0] + border) * i
			xoff = (img_shape[1] + border) * j
			tile_img[yoff:yoff+img_shape[0], xoff:xoff+img_shape[1], ...] = img

	cv2.imwrite(args.images_path+"/img_"+str(epoch) + ".jpg", (tile_img + 1)*127.5)
