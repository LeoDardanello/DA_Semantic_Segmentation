import torch.nn as nn
import torch
from torchvision import transforms
from torch.nn import functional as F
from PIL import Image
import numpy as np
import pandas as pd
import random
import numbers
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from PIL import Image
import os
from tqdm import tqdm

###########################################################################################
def split_dataset(toSplit):
    indsToSplit = range(0, len(toSplit))
    splitting = train_test_split(indsToSplit, train_size = 0.75, random_state = 42, stratify = None, shuffle = True)
    train_indexes = splitting[0]
    val_indexes = splitting[1]
    return Subset(toSplit,train_indexes),Subset(toSplit,val_indexes)

def one_hot_custom(label, label_info):
    semantic_map = np.zeros(label.shape[:2],dtype=np.uint8)
    # print("sematic shape",semantic_map.shape)
    # print("label_shape",label.shape)
    for info in label_info:
        color = info[-3:]
        class_map = np.all(label == color.reshape(1, 1,3), axis=2)
        semantic_map[class_map] = info[0]
    return semantic_map  #return a numpy array

def get_label_info_custom(csv_path):
    # return label -> {label_name: [r_value, g_value, b_value, ...}
    ann = pd.read_csv(csv_path)
    label = []
    for iter, row in ann.iterrows():
        label_name = row["Name"]
        label_id = row["ID"]
        rgb_color = [row["R"],row["G"],row["B"]]
        label.append( [label_id] +  rgb_color)
    return np.array(label)

def from_RGB_to_LabelID(label_colored,path,height,width):
    label_info = get_label_info_custom('/content/DA_Semantic_Segmentation/GTA.csv') 
    index=1
    label_list=[]
    if not os.path.exists("/content/GTA5/TrainID"):
        os.makedirs("/content/GTA5/TrainID")
    print("Generating LabelID..")
    for l in tqdm(label_colored):
        file_path =f"/content/GTA5/TrainID/{str(index).zfill(5)}.png"
        label_list.append(f"TrainID/{str(index).zfill(5)}.png")
        if not os.path.exists(file_path):
            with open(path+l, 'rb') as f:
                img = Image.open(f)
                img=img.convert("RGB").resize((width, height), Image.NEAREST)
            conv_img=one_hot_custom(np.array(img),label_info)
            conv_img = Image.fromarray(conv_img)
            conv_img.convert('L').save(file_path)
        index+=1
    return label_list

class DataAugmentation(object):

    def __call__(self, image, label):
        method= np.random.choice([RandCrop(), HorizontalFlip(), Jitter()])
        return method(image, label)


class RandCrop(DataAugmentation):
	"""Crop the given Image and label at a random location.

	Args:
		size (sequence or int): Desired output size of the crop. If size is an
			int instead of sequence like (h, w), a square crop (size, size) is
			made.
		padding (int or sequence, optional): Optional padding on each border
			of the image. Default is 0, i.e no padding. If a sequence of length
			4 is provided, it is used to pad left, top, right, bottom borders
			respectively.
	"""
	def __init__(self, seed=42): 				# void class for padding
		super(RandCrop, self).__init__()
		self.size = None
		random.seed(seed)


	@staticmethod
	def get_params(img, output_size, h, w):
		tw, th = output_size
		if w == tw and h == th:
			return 0, 0, h, w
		i = random.randint(0, h - th)
		j = random.randint(0, w - tw)
		return (j, i, j+tw, i+th)


	def __call__(self, img, label):
		#with PIL as input
		w, h =label.size
		self.size = w//2, h//2

		crop_box= self.get_params(label, self.size, h, w)

		cropped_image = img.crop(crop_box).resize((w, h), Image.NEAREST)	
		cropped_label = label.crop(crop_box).resize((w, h), Image.NEAREST)	

		return cropped_image, cropped_label
    
	
	def __str__(self):
		return "Random crop"


class HorizontalFlip(DataAugmentation):
	"""Horizontal Flip of the given Image and label.

	Args:
				-
	"""
	def __init__(self):
		super(HorizontalFlip, self).__init__()

	def __call__(self, img, label):
		flipped_image = img.transpose(Image.FLIP_LEFT_RIGHT)
		flipped_label = label.transpose(Image.FLIP_LEFT_RIGHT)
		return flipped_image, flipped_label
	
	def __str__(self):
		return "Horizontal flip"
	
class Jitter(DataAugmentation):
	def __init__(self):
		super(Jitter).__init__()

	def __call__(self, img, label):
		jitter_transform = transforms.Compose([
			transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
		])
		# transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
		jittered_image = jitter_transform(img)
		return jittered_image, label
	
	def __str__(self):
		return "Jitter"

# bytescale and to_image functions from scipy.misc
def bytescale(data, cmin=None, cmax=None, high=255, low=0):

#     Byte scales an array (image).
#     Byte scaling means converting the input image to uint8 dtype and scaling
#     the range to ``(low, high)`` (default 0-255).
#     If the input image already has dtype uint8, no scaling is done.
#     This function is only available if Python Imaging Library (PIL) is installed.
#     Parameters
#     ----------
#     data : ndarray
#         PIL image data array.
#     cmin : scalar, optional
#         Bias scaling of small values. Default is ``data.min()``.
#     cmax : scalar, optional
#         Bias scaling of large values. Default is ``data.max()``.
#     high : scalar, optional
#         Scale max value to `high`.  Default is 255.
#     low : scalar, optional
#         Scale min value to `low`.  Default is 0.
#     Returns
#     -------
#     img_array : uint8 ndarray
#         The byte-scaled array.
#     Examples
#     --------
#     >>> from scipy.misc import bytescale
#     >>> img = np.array([[ 91.06794177,   3.39058326,  84.4221549 ],
#     ...                 [ 73.88003259,  80.91433048,   4.88878881],
#     ...                 [ 51.53875334,  34.45808177,  27.5873488 ]])
#     >>> bytescale(img)
#     array([[255,   0, 236],
#            [205, 225,   4],
#            [140,  90,  70]], dtype=uint8)
#     >>> bytescale(img, high=200, low=100)
#     array([[200, 100, 192],
#            [180, 188, 102],
#            [155, 135, 128]], dtype=uint8)
#     >>> bytescale(img, cmin=0, cmax=255)
#     array([[91,  3, 84],
#            [74, 81,  5],
#            [52, 34, 28]], dtype=uint8)

    if data.dtype == np.uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)

def toimage(arr, high=255, low=0, cmin=None, cmax=None, pal=None,
            mode=None, channel_axis=None):
#     Takes a numpy array and returns a PIL image.
#     This function is only available if Python Imaging Library (PIL) is installed.
#     The mode of the PIL image depends on the array shape and the `pal` and
#     `mode` keywords.
#     For 2-D arrays, if `pal` is a valid (N,3) byte-array giving the RGB values
#     (from 0 to 255) then ``mode='P'``, otherwise ``mode='L'``, unless mode
#     is given as 'F' or 'I' in which case a float and/or integer array is made.
#     .. warning::
#         This function uses `bytescale` under the hood to rescale images to use
#         the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
#         It will also cast data for 2-D images to ``uint32`` for ``mode=None``
#         (which is the default).
#     Notes
#     -----
#     For 3-D arrays, the `channel_axis` argument tells which dimension of the
#     array holds the channel data.
#     For 3-D arrays if one of the dimensions is 3, the mode is 'RGB'
#     by default or 'YCbCr' if selected.
#     The numpy array must be either 2 dimensional or 3 dimensional.

    data = np.asarray(arr)
    if np.iscomplexobj(data):
        raise ValueError("Cannot convert a complex-valued array.")
    shape = list(data.shape)
    valid = len(shape) == 2 or ((len(shape) == 3) and
                                ((3 in shape) or (4 in shape)))
    if not valid:
        raise ValueError("'arr' does not have a suitable array shape for "
                         "any mode.")
    if len(shape) == 2:
        shape = (shape[1], shape[0])  # columns show up first
        if mode == 'F':
            data32 = data.astype(np.float32)
            image = Image.frombytes(mode, shape, data32.tobytes())
            return image
        if mode in [None, 'L', 'P']:
            bytedata = bytescale(data, high=high, low=low,
                                 cmin=cmin, cmax=cmax)
            image = Image.frombytes('L', shape, bytedata.tobytes())
            if pal is not None:
                image.putpalette(np.asarray(pal, dtype=np.uint8).tobytes())
                # Becomes a mode='P' automagically.
            elif mode == 'P':  # default gray-scale
                pal = (np.arange(0, 256, 1, dtype=np.uint8)[:, np.newaxis] *
                       np.ones((3,), dtype=np.uint8)[np.newaxis, :])
                image.putpalette(np.asarray(pal, dtype=np.uint8).tobytes())
            return image
        if mode == '1':  # high input gives threshold for 1
            bytedata = (data > high)
            image = Image.frombytes('1', shape, bytedata.tobytes())
            return image
        if cmin is None:
            cmin = np.amin(np.ravel(data))
        if cmax is None:
            cmax = np.amax(np.ravel(data))
        data = (data*1.0 - cmin)*(high - low)/(cmax - cmin) + low
        if mode == 'I':
            data32 = data.astype(np.uint32)
            image = Image.frombytes(mode, shape, data32.tobytes())
        else:
            raise ValueError(_errstr)
        return image

    # if here then 3-d array with a 3 or a 4 in the shape length.
    # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
    if channel_axis is None:
        if (3 in shape):
            ca = np.flatnonzero(np.asarray(shape) == 3)[0]
        else:
            ca = np.flatnonzero(np.asarray(shape) == 4)
            if len(ca):
                ca = ca[0]
            else:
                raise ValueError("Could not find channel dimension.")
    else:
        ca = channel_axis

    numch = shape[ca]
    if numch not in [3, 4]:
        raise ValueError("Channel axis dimension is not valid.")

    bytedata = bytescale(data, high=high, low=low, cmin=cmin, cmax=cmax)
    if ca == 2:
        strdata = bytedata.tobytes()
        shape = (shape[1], shape[0])
    elif ca == 1:
        strdata = np.transpose(bytedata, (0, 2, 1)).tobytes()
        shape = (shape[2], shape[0])
    elif ca == 0:
        strdata = np.transpose(bytedata, (1, 2, 0)).tobytes()
        shape = (shape[2], shape[1])
    if mode is None:
        if numch == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'

    if mode not in ['RGB', 'RGBA', 'YCbCr', 'CMYK']:
        raise ValueError(_errstr)

    if mode in ['RGB', 'YCbCr']:
        if numch != 3:
            raise ValueError("Invalid array shape for mode.")
    if mode in ['RGBA', 'CMYK']:
        if numch != 4:
            raise ValueError("Invalid array shape for mode.")

    # Here we know data and mode is correct
    image = Image.frombytes(mode, shape, strdata)
    return image
	
# FDA: Fourier Domain Adaptation
class FourierDomainAdaptation(object):

	def __init__(self, beta=0.01):
		self.beta = beta

	def __call__(self, im_src, im_trg):
		im_src = np.asarray(im_src, np.float32)
		im_trg = np.asarray(im_trg, np.float32)

		im_src = im_src.transpose((2, 0, 1))
		im_trg = im_trg.transpose((2, 0, 1))

		src_in_trg = self.FDA_source_to_target_np( im_src, im_trg, L=self.beta )
		return src_in_trg.transpose((1, 2, 0))

	def low_freq_mutate_np(self, amp_src, amp_trg, L=0.1 ):
			a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
			a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

			_, h, w = a_src.shape
			b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
			c_h = np.floor(h/2.0).astype(int)
			c_w = np.floor(w/2.0).astype(int)

			h1 = c_h-b
			h2 = c_h+b+1
			w1 = c_w-b
			w2 = c_w+b+1

			a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
			a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
			return a_src

	def FDA_source_to_target_np(self, src_img, trg_img, L=0.1 ):
			# exchange magnitude
			# input: src_img, trg_img

			src_img_np = src_img #.cpu().numpy()
			trg_img_np = trg_img #.cpu().numpy()

			# get fft of both source and target
			fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
			fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )

			# extract amplitude and phase of both ffts
			amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
			amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

			# mutate the amplitude part of source with target
			amp_src_ = self.low_freq_mutate_np( amp_src, amp_trg, L=L )

			# mutated fft of source
			fft_src_ = amp_src_ * np.exp( 1j * pha_src )

			# get the mutated image
			src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
			src_in_trg = np.real(src_in_trg)

			return src_in_trg

#################################################################################################

def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=300, power=0.9):
	"""Polynomial decay of learning rate
		:param init_lr is base learning rate
		:param iter is a current iteration
		:param lr_decay_iter how frequently decay occurs, default is 1
		:param max_iter is number of maximum iterations
		:param power is a polymomial power

	"""
	# if iter % lr_decay_iter or iter > max_iter:
	# 	return optimizer

	lr = init_lr*(1 - iter/max_iter)**power
	optimizer.param_groups[0]['lr'] = lr
	return lr
	# return lr

def get_label_info(csv_path):
	# return label -> {label_name: [r_value, g_value, b_value, ...}
	ann = pd.read_csv(csv_path)
	label = {}
	for iter, row in ann.iterrows():
		label_name = row['name']
		r = row['r']
		g = row['g']
		b = row['b']
		class_11 = row['class_11']
		label[label_name] = [int(r), int(g), int(b), class_11]
	return label

def one_hot_it_new(label, label_info):
	# label is a tensor -> [3, H, W] with the RGB values for each pixel
	# label_info is an array with [train_id, r value, g value, b value] for each class
	# return a tensor with the semantic_map -> [class_num, H, W]
	# the semantic map has 19 channels, for each channel there is a binary map HxW where the pixels 
	# whose value is 1 means the pixel belongs to the class, otherwise the value is 0
    semantic_map = np.zeros((19, label.shape[1], label.shape[2]))

    for index, info in enumerate(label_info[:-1]):
        color = info[:3].reshape(3, 1, 1)
        equality = np.all(label == color, axis=0)
     
        semantic_map[index % 19][equality] = 1

    return torch.tensor(semantic_map)

def one_hot_it(label, label_info):
	# return semantic_map -> [H, W]
	semantic_map = np.zeros(label.shape[:-1])
	for index, info in enumerate(label_info):
		color = label_info[info]
		# colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
		equality = np.equal(label, color)
		class_map = np.all(equality, axis=-1)
		semantic_map[class_map] = index
		# semantic_map.append(class_map)
	# semantic_map = np.stack(semantic_map, axis=-1)
	return semantic_map


def one_hot_it_v11(label, label_info):
	# return semantic_map -> [H, W, class_num]
	semantic_map = np.zeros(label.shape[:-1])
	# from 0 to 11, and 11 means void
	class_index = 0
	for index, info in enumerate(label_info):
		color = label_info[info][:3]
		class_11 = label_info[info][3]
		if class_11 == 1:
			# colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
			equality = np.equal(label, color)
			class_map = np.all(equality, axis=-1)
			# semantic_map[class_map] = index
			semantic_map[class_map] = class_index
			class_index += 1
		else:
			equality = np.equal(label, color)
			class_map = np.all(equality, axis=-1)
			semantic_map[class_map] = 11
	return semantic_map

def one_hot_it_v11_dice(label, label_info):
	# return semantic_map -> [H, W, class_num]
	semantic_map = []
	void = np.zeros(label.shape[:2])
	for index, info in enumerate(label_info):
		color = label_info[info][:3]
		class_11 = label_info[info][3]
		if class_11 == 1:
			# colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
			equality = np.equal(label, color)
			class_map = np.all(equality, axis=-1)
			# semantic_map[class_map] = index
			semantic_map.append(class_map)
		else:
			equality = np.equal(label, color)
			class_map = np.all(equality, axis=-1)
			void[class_map] = 1
	semantic_map.append(void)
	semantic_map = np.stack(semantic_map, axis=-1).astype(np.float)
	return semantic_map

def reverse_one_hot(image):
	"""
	Transform a 2D array in one-hot format (depth is num_classes),
	to a 2D array with only 1 channel, where each pixel value is
	the classified class key.

	# Arguments
		image: The one-hot format image

	# Returns
		A 2D array with the same width and height as the input, but
		with a depth size of 1, where each pixel value is the classified
		class key.
	"""
	# w = image.shape[0]
	# h = image.shape[1]
	# x = np.zeros([w,h,1])

	# for i in range(0, w):
	#     for j in range(0, h):
	#         index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
	#         x[i, j] = index
	image = image.permute(1, 2, 0)
	x = torch.argmax(image, dim=-1)
	return x


def colour_code_segmentation(image, label_values):
	"""
    Given a 1-channel array of class keys, colour code the segmentation results.

    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """

	# w = image.shape[0]
	# h = image.shape[1]
	# x = np.zeros([w,h,3])
	# colour_codes = label_values
	# for i in range(0, w):
	#     for j in range(0, h):
	#         x[i, j, :] = colour_codes[int(image[i, j])]
	label_values = [label_values[key][:3] for key in label_values if label_values[key][3] == 1]
	label_values.append([0, 0, 0])
	colour_codes = np.array(label_values)
	x = colour_codes[image.astype(int)]

	return x

def compute_global_accuracy(pred, label):
	pred = pred.flatten()
	label = label.flatten()
	total = len(label)
	count = 0.0
	for i in range(total):
		if pred[i] == label[i]:
			count = count + 1.0
	return float(count) / float(total)

def fast_hist(a, b, n):
	'''
	a and b are predict and mask respectively
	n is the number of classes
	'''
	k = (a >= 0) & (a < n)
	return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
	epsilon = 1e-5
	return (np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)

class RandomCrop(object):
	"""Crop the given PIL Image at a random location.

	Args:
		size (sequence or int): Desired output size of the crop. If size is an
			int instead of sequence like (h, w), a square crop (size, size) is
			made.
		padding (int or sequence, optional): Optional padding on each border
			of the image. Default is 0, i.e no padding. If a sequence of length
			4 is provided, it is used to pad left, top, right, bottom borders
			respectively.
		pad_if_needed (boolean): It will pad the image if smaller than the
			desired size to avoid raising an exception.
	"""

	def __init__(self, size, seed, padding=0, pad_if_needed=False):
		if isinstance(size, numbers.Number):
			self.size = (int(size), int(size))
		else:
			self.size = size
		self.padding = padding
		self.pad_if_needed = pad_if_needed
		self.seed = seed

	@staticmethod
	def get_params(img, output_size, seed):
		"""Get parameters for ``crop`` for a random crop.

		Args:
			img (PIL Image): Image to be cropped.
			output_size (tuple): Expected output size of the crop.

		Returns:
			tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
		"""
		random.seed(seed)
		w, h = img.size
		th, tw = output_size
		if w == tw and h == th:
			return 0, 0, h, w
		i = random.randint(0, h - th)
		j = random.randint(0, w - tw)
		return i, j, th, tw

	def __call__(self, img):
		"""
		Args:
			img (PIL Image): Image to be cropped.

		Returns:
			PIL Image: Cropped image.
		"""
		if self.padding > 0:
			img = torchvision.transforms.functional.pad(img, self.padding)

		# pad the width if needed
		if self.pad_if_needed and img.size[0] < self.size[1]:
			img = torchvision.transforms.functional.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
		# pad the height if needed
		if self.pad_if_needed and img.size[1] < self.size[0]:
			img = torchvision.transforms.functional.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))

		i, j, h, w = self.get_params(img, self.size, self.seed)

		return torchvision.transforms.functional.crop(img, i, j, h, w)

	def __repr__(self):
		return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

def cal_miou(miou_list, csv_path):
	# return label -> {label_name: [r_value, g_value, b_value, ...}
	ann = pd.read_csv(csv_path)
	miou_dict = {}
	cnt = 0
	for iter, row in ann.iterrows():
		label_name = row['name']
		class_11 = int(row['class_11'])
		if class_11 == 1:
			miou_dict[label_name] = miou_list[cnt]
			cnt += 1
	return miou_dict, np.mean(miou_list)

class OHEM_CrossEntroy_Loss(nn.Module):
	def __init__(self, threshold, keep_num):
		super(OHEM_CrossEntroy_Loss, self).__init__()
		self.threshold = threshold
		self.keep_num = keep_num
		self.loss_function = nn.CrossEntropyLoss(reduction='none')

	def forward(self, output, target):
		loss = self.loss_function(output, target).view(-1)
		loss, loss_index = torch.sort(loss, descending=True)
		threshold_in_keep_num = loss[self.keep_num]
		if threshold_in_keep_num > self.threshold:
			loss = loss[loss>self.threshold]
		else:
			loss = loss[:self.keep_num]
		return torch.mean(loss)

def group_weight(weight_group, module, norm_layer, lr):
	group_decay = []
	group_no_decay = []
	for m in module.modules():
		if isinstance(m, nn.Linear):
			group_decay.append(m.weight)
			if m.bias is not None:
				group_no_decay.append(m.bias)
		elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
			group_decay.append(m.weight)
			if m.bias is not None:
				group_no_decay.append(m.bias)
		elif isinstance(m, norm_layer) or isinstance(m, nn.GroupNorm):
			if m.weight is not None:
				group_no_decay.append(m.weight)
			if m.bias is not None:
				group_no_decay.append(m.bias)

	assert len(list(module.parameters())) == len(group_decay) + len(
		group_no_decay)
	weight_group.append(dict(params=group_decay, lr=lr))
	weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
	return weight_group
