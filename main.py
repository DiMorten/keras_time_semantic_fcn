from utils import *
from keras.datasets import cifar10
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout, Conv2DTranspose
# from keras.callbacks import ModelCheckpoint , EarlyStopping
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
import keras

import numpy as np
from sklearn.utils import shuffle
import cv2
from skimage.util import view_as_windows
import argparse
import tensorflow as tf

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import metrics

import glob

from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,classification_report
# Local
from densnet import DenseNetFCN
from metrics import fmeasure,categorical_accuracy
import deb
from keras_weighted_categorical_crossentropy import weighted_categorical_crossentropy, sparse_accuracy_ignoring_last_label
from keras.models import load_model

parser = argparse.ArgumentParser(description='')
parser.add_argument('-tl', '--t_len', dest='t_len',
					type=int, default=7, help='t len')
parser.add_argument('-cn', '--class_n', dest='class_n',
					type=int, default=11, help='class_n')
parser.add_argument('-chn', '--channel_n', dest='channel_n',
					type=int, default=2, help='channel number')

parser.add_argument('-pl', '--patch_len', dest='patch_len',
					type=int, default=32, help='patch len')
parser.add_argument('-pstr', '--patch_step_train', dest='patch_step_train',
					type=int, default=32, help='patch len')
parser.add_argument('-psts', '--patch_step_test', dest='patch_step_test',
					type=int, default=None, help='patch len')

parser.add_argument('-db', '--debug', dest='debug',
					type=int, default=1, help='patch len')
parser.add_argument('-ep', '--epochs', dest='epochs',
					type=int, default=8000, help='patch len')
parser.add_argument('-pt', '--patience', dest='patience',
					type=int, default=300, help='patience')

parser.add_argument('-bstr', '--batch_size_train', dest='batch_size_train',
					type=int, default=32, help='patch len')
parser.add_argument('-bsts', '--batch_size_test', dest='batch_size_test',
					type=int, default=32, help='patch len')

parser.add_argument('-em', '--eval_mode', dest='eval_mode',
					default='metrics', help='Test evaluate mode: metrics or predict')
parser.add_argument('-is', '--im_store', dest='im_store',
					default=True, help='Store sample test predicted images')
parser.add_argument('-eid', '--exp_id', dest='exp_id',
					default='default', help='Experiment id')

parser.add_argument('-path', '--path', dest='path',
					default='../data/', help='Experiment id')

args = parser.parse_args()

if args.patch_step_test==None:
	args.patch_step_test=args.patch_len

deb.prints(args.patch_step_test)

# ================= Generic class for init values =============================================== #
class NetObject(object):

	def __init__(self, patch_len=32, patch_step_train=32,patch_step_test=32, path="../data/", im_name_train="Image_Train.tif", im_name_test="Image_Test.tif", label_name_train="Reference_Train.tif", label_name_test="Reference_Test.tif", channel_n=2, debug=1,exp_id="skip_connections",
		t_len=7,class_n=11):
		self.patch_len = patch_len
		self.path = {"v": path, 'train': {}, 'test': {}}
		self.image = {'train': {}, 'test': {}}
		self.patches = {'train': {}, 'test': {}}

		self.patches['train']['step']=patch_step_train
		self.patches['test']['step']=patch_step_test        
		self.path['train']['in'] = path + 'train_test/train/ims/'
		self.path['test']['in'] = path + 'train_test/test/ims/'
		self.path['train']['label'] = path + 'train_test/train/labels/'
		self.path['test']['label'] = path + 'train_test/test/labels/'
		self.channel_n = channel_n
		self.debug = debug
		self.class_n = class_n
		self.report={'best':{}, 'val':{}}
		self.report['exp_id']=exp_id
		self.report['best']['text_name']='result_'+exp_id+'.txt'
		self.report['best']['text_path']='../results/'+self.report['best']['text_name']
		self.report['best']['text_history_path']='../results/'+'history.txt'
		self.report['val']['history_path']='../results/'+'history_val.txt'
		
		self.t_len=t_len
# ================= Dataset class implements data loading, patch extraction, metric calculation and image reconstruction =======#
class Dataset(NetObject):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.im_gray_idx_to_rgb_table=[[0,[0,0,255],29],
									[1,[0,255,0],150],
									[2,[0,255,255],179],
									[3,[255,255,0],226],
									[4,[255,255,255],255]]
		if self.debug >= 1:
			print("Initializing Dataset instance")

	def create(self):
		self.image["train"], self.patches["train"] = self.subset_create(
			self.path['train'],self.patches["train"]['step'])
		self.image["test"], self.patches["test"] = self.subset_create(
			self.path['test'],self.patches["test"]['step'])

		if self.debug:
			deb.prints(self.image["train"]['in'].shape)
			deb.prints(self.image["train"]['label'].shape)

			deb.prints(self.image["test"]['in'].shape)
			deb.prints(self.image["test"]['label'].shape)

	def create_load(self):

		self.patches_list={'train':{},'test':{}}
		#self.patches_list['train']['ims']=glob.glob(self.path['train']['in']+'*.npy')
		#self.patches_list['train']['label']=glob.glob(self.path['train']['label']+'*.npy')

		#self.patches_list['test']['ims']=glob.glob(self.path['test']['in']+'*.npy')
		#self.patches_list['test']['label']=glob.glob(self.path['test']['label']+'*.npy')
		self.patches['test']['label'],self.patches_list['test']['label']=self.folder_load(self.path['test']['label'])
		
		self.patches['train']['in'],self.patches_list['train']['ims']=self.folder_load(self.path['train']['in'])
		self.patches['train']['label'],self.patches_list['train']['label']=self.folder_load(self.path['train']['label'])
		self.patches['test']['in'],self.patches_list['test']['ims']=self.folder_load(self.path['test']['in'])
		deb.prints(self.patches['train']['in'].shape)
		deb.prints(self.patches['test']['in'].shape)
		deb.prints(self.patches['train']['label'].shape)
		
		print("Switching to one hot")
		self.patches['train']['label']=self.batch_label_to_one_hot(self.patches['train']['label'])
		self.patches['test']['label']=self.batch_label_to_one_hot(self.patches['test']['label'])

		deb.prints(len(self.patches_list['test']['label']))
		deb.prints(len(self.patches_list['test']['ims']))
		deb.prints(self.patches['train']['in'].shape)
		deb.prints(self.patches['train']['in'].dtype)
		
		deb.prints(self.patches['train']['label'].shape)
		
		self.patches['train']['n']=self.patches['train']['in'].shape[0]
		self.patches['train']['idx']=range(self.patches['train']['n'])

	def batch_label_to_one_hot(self,im):
		im_one_hot=np.zeros((im.shape[0],im.shape[1],im.shape[2],self.class_n))
		print(im_one_hot.shape)
		print(im.shape)
		for clss in range(0,self.class_n):
			im_one_hot[:,:,:,clss][im[:,:,:]==clss]=1
		return im_one_hot

	def folder_load(self,folder_path):
		paths=glob.glob(folder_path+'*.npy')
		files=[]
		deb.prints(len(paths))
		for path in paths:
			#print(path)
			files.append(np.load(path))
		return np.asarray(files),paths
	def subset_create(self, path,patch_step):
		image = self.image_load(path)
		image['label_rgb']=image['label'].copy()
		image['label'] = self.label2idx(image['label'])
		patches = self.patches_extract(image,patch_step)
		return image, patches

	def image_load(self, path):
		image = {}
		image['in'] = cv2.imread(path['in'], -1)
		image['label'] = np.expand_dims(cv2.imread(path['label'], 0), axis=2)
		count,unique=np.unique(image['label'],return_counts=True)
		print("label count,unique",count,unique)
		image['label_rgb']=cv2.imread(path['label'], -1)
		return image

	def patches_extract(self, image, patch_step):

		patches = {}
		patches['in'],_ = self.view_as_windows_multichannel(
			image['in'], (self.patch_len, self.patch_len, self.channel_n), step=patch_step)
		patches['label'],patches['label_partitioned_shape'] = self.view_as_windows_multichannel(
			image['label'], (self.patch_len, self.patch_len, 1), step=patch_step)

		# ===================== Switch labels to one-hot ===============#

		if self.debug >= 2:
			deb.prints(patches['label'].shape)

		if flag['label_one_hot']:

			# Get the vectorized integer label
			patches['label_h'] = np.reshape(
				patches['label'], (patches['label'].shape[0], patches['label'].shape[1]*patches['label'].shape[2]))
			deb.prints(patches['label_h'].shape)

			# Init the one-hot vectorized label
			patches['label_h2'] = np.zeros(
				(patches['label_h'].shape[0], patches['label_h'].shape[1], self.class_n))

			# Get the one-hot vectorized label
			for sample_idx in range(0, patches['label_h'].shape[0]):
				for loc_idx in range(0, patches['label_h'].shape[1]):
					patches['label_h2'][sample_idx, loc_idx,
										patches['label_h'][sample_idx][loc_idx]] = 1

			# Get the image one-hot labels
			patches['label'] = np.reshape(patches['label_h2'], (patches['label_h2'].shape[0],
																patches['label'].shape[1], patches['label'].shape[2], self.class_n))

			if self.debug >= 2:
				deb.prints(patches['label_h2'].shape)

		# ============== End switch labels to one-hot =============#
		if self.debug:
			deb.prints(patches['label'].shape)
			deb.prints(patches['in'].shape)

		return patches

	def label2idx(self, image_label):
		unique = np.unique(image_label)
		idxs = np.array(range(0, unique.shape[0]))
		for val, idx in zip(unique, idxs):
			image_label[image_label == val] = idx
		return image_label

	def view_as_windows_multichannel(self, arr_in, window_shape, step=1):
		out = np.squeeze(view_as_windows(arr_in, window_shape, step=step))
		partitioned_shape=out.shape

		deb.prints(out.shape)
		out = np.reshape(out, (out.shape[0] * out.shape[1],) + out.shape[2::])
		return out,partitioned_shape

#=============== METRICS CALCULATION ====================#
	def ims_flatten(self,ims):
		return np.reshape(ims,(np.prod(ims.shape[0:-1]),ims.shape[-1])).astype(np.float64)

	def average_acc(self,y_pred,y_true):
		correct_per_class=np.zeros(self.class_n)
		correct_all=y_pred.argmax(axis=1)[y_pred.argmax(axis=1)==y_true.argmax(axis=1)]
		for clss in range(0,self.class_n):
			correct_per_class[clss]=correct_all[correct_all==clss].shape[0]
		if self.debug>=1:
			deb.prints(correct_per_class)

		pred_unique,pred_class_count=np.unique(y_pred.argmax(axis=1),return_counts=True)
		deb.prints(pred_class_count)
		deb.prints(pred_unique+1)


		unique,per_class_count=np.unique(y_true.argmax(axis=1),return_counts=True)
		deb.prints(per_class_count)
		deb.prints(unique+1)
		per_class_count_all=np.zeros(self.class_n)
		for clss,count in zip(unique,per_class_count):
			per_class_count_all[clss]=count
		per_class_acc=np.divide(correct_per_class[1:].astype('float32'),per_class_count_all[1:].astype('float32'))
		average_acc=np.average(per_class_acc)
		return average_acc,per_class_acc
	def flattened_to_im(self,data_h,im_shape):
		return np.reshape(data_h,im_shape)

	def probabilities_to_one_hot(self,vals):
		out=np.zeros_like(vals)
		out[np.arange(len(vals)), vals.argmax(1)] = 1
		return out
	def assert_equal(self,val1,val2):
		return np.equal(val1,val2)


	def metrics_get(self,data,ignore_bcknd=True,debug=1): #requires batch['prediction'],batch['label']
		

		# ==========================IMGS FLATTEN ==========================================#
		data['prediction_h'] = self.ims_flatten(data['prediction'])
		data['prediction_h']=self.probabilities_to_one_hot(data['prediction_h'])
				
		data['label_h'] = self.ims_flatten(data['label']) #(self.batch['test']['size']*self.patch_len*self.patch_len,self.class_n


		if ignore_bcknd==True:
			data['prediction_h']=data['prediction_h'][:,1:]
			data['label_h']=data['label_h'][:,1:]

			if debug>0:
				deb.prints(data['label_h'].shape)
				deb.prints(data['prediction_h'].shape)
			#indices_to_keep=data['prediction_h']
			#data['prediction_h']=data['prediction_h'][:,data['prediction_h']!=0]
			data['prediction_h']=data['prediction_h'][~np.all(data['label_h'] == 0, axis=1)]
			data['label_h']=data['label_h'][~np.all(data['label_h'] == 0, axis=1)]
			
			#for row in range(0,data['label_h'].shape[0]):
			#	if np.sum(data['label_h'][row,:])==0:
			#		np.delete(data['label_h'],row,0)
			#		np.delete(data['prediction_h'],row,0)


		if debug>=1: 
			deb.prints(data['prediction_h'].dtype)
			deb.prints(data['label_h'].dtype)
			deb.prints(data['prediction_h'].shape)
			deb.prints(data['label_h'].shape)
			deb.prints(data['label_h'][0])
			deb.prints(data['prediction_h'][0])

		#============= TEST UNIQUE PRINTING==================#
		unique,count=np.unique(data['label_h'].argmax(axis=1),return_counts=True)
		print("Metric real unique+1,count",unique+1,count)
		unique,count=np.unique(data['prediction_h'].argmax(axis=1),return_counts=True)
		print("Metric prediction unique+1,count",unique+1,count)
		
		#========================METRICS GET================================================#
		metrics={}
		metrics['f1_score']=f1_score(data['label_h'],data['prediction_h'],average='macro')
		metrics['f1_score_weighted']=f1_score(data['label_h'],data['prediction_h'],average='weighted')
		
		metrics['overall_acc']=accuracy_score(data['label_h'],data['prediction_h'])
		metrics['confusion_matrix']=confusion_matrix(data['label_h'].argmax(axis=1),data['prediction_h'].argmax(axis=1))
		metrics['per_class_acc']=(metrics['confusion_matrix'].astype('float') / metrics['confusion_matrix'].sum(axis=1)[:, np.newaxis]).diagonal()
		
		metrics['average_acc']=np.average(metrics['per_class_acc'][~np.isnan(metrics['per_class_acc'])])

		
		#=====================IMG RECONSTRUCT============================================#
		if ignore_bcknd!=True:
			data_label_reconstructed=self.flattened_to_im(data['label_h'],data['label'].shape)
			data_prediction_reconstructed=self.flattened_to_im(data['prediction_h'],data['label'].shape)
		
			deb.prints(data_label_reconstructed.shape)
			np.testing.assert_almost_equal(data['label'],data_label_reconstructed)
			print("Is label reconstructed equal to original",np.array_equal(data['label'],data_label_reconstructed))
			print("Is prediction reconstructed equal to original",np.array_equal(data['prediction'].argmax(axis=3),data_prediction_reconstructed.argmax(axis=3)))

		if self.debug>=2: print(metrics['per_class_acc'])

		return metrics

	def metrics_write_to_txt(self,metrics,loss,epoch=0,path=None):
		#with open(self.report['best']['text_path'], "w") as text_file:
		#    text_file.write("Overall_acc,average_acc,f1_score: {0},{1},{2},{3}".format(str(metrics['overall_acc']),str(metrics['average_acc']),str(metrics['f1_score']),str(epoch)))
		#deb.prints(loss)
		#deb.prints(loss[0])
		#deb.prints(loss[1])
		
		with open(path, "a") as text_file:
			#text_file.write("{0},{1},{2},{3}\n".format(str(epoch),str(metrics['overall_acc']),str(metrics['average_acc']),str(metrics['f1_score'])))
			text_file.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14}\n".format(str(epoch),
				str(metrics['overall_acc']),str(metrics['average_acc']),str(metrics['f1_score']),str(metrics['f1_score_weighted']),str(loss[0]),str(loss[1]),
				str(metrics['per_class_acc'][0]),str(metrics['per_class_acc'][1]),str(metrics['per_class_acc'][2]),
				str(metrics['per_class_acc'][3]),str(metrics['per_class_acc'][4]),str(metrics['per_class_acc'][5]),
				str(metrics['per_class_acc'][6]),str(metrics['per_class_acc'][7])))
			
			
	def metrics_per_class_from_im_get(self,name='im_reconstructed_rgb_test_predictionplen64_3.png',folder='../results/reconstructed/',average=None):
		data={}
		metrics={}
		deb.prints(folder+name)
		data['prediction']=cv2.imread(folder+name,0)[0:-30,0:-2]
		data['label']=cv2.imread(folder+'im_reconstructed_rgb_test_labelplen64_3.png',0)[0:-30,0:-2]

		data['prediction']=np.reshape(data['prediction'],-1)
		data['label']=np.reshape(data['label'],-1)
		
		metrics['f1_score_per_class']=f1_score(data['prediction'],data['label'],average=average)
		print(metrics)


# =================== Image reconstruct =======================#

	def im_reconstruct(self,subset='test',mode='prediction'):
		h,w,_=self.image[subset]['label'].shape
		print(self.patches[subset]['label_partitioned_shape'])
		deb.prints(self.patches[subset][mode].shape)
		
		h_blocks,w_blocks,patch_len,_=self.patches[subset]['label_partitioned_shape']

		patches_block=np.reshape(self.patches[subset][mode].argmax(axis=3),(h_blocks,w_blocks,patch_len,patch_len))


		self.im_reconstructed=np.squeeze(np.zeros_like(self.image[subset]['label']))

		h_block_len=int(self.image[subset]['label'].shape[0]/h_blocks)
		w_block_len=int(self.image[subset]['label'].shape[1]/w_blocks)
		
		count=0

		for w_block in range(0,w_blocks):
			for h_block in range(0,h_blocks):
				y=int(h_block*h_block_len)
				x=int(w_block*w_block_len)
				#print(y)
				#print(x)				
				#deb.prints([y:y+self.patch_len])
				self.im_reconstructed[y:y+self.patch_len,x:x+self.patch_len]=patches_block[h_block,w_block,:,:]
				count+=1

		self.im_reconstructed_rgb=self.im_gray_idx_to_rgb(self.im_reconstructed)
		if self.debug>=3: 
			deb.prints(count)
			deb.prints(self.im_reconstructed_rgb.shape)

		cv2.imwrite('../results/reconstructed/im_reconstructed_rgb_'+subset+'_'+mode+self.report['exp_id']+'.png',self.im_reconstructed_rgb.astype(np.uint8))

	def im_gray_idx_to_rgb(self,im):
		out=np.zeros((im.shape+(3,)))
		for chan in range(0,3):
			for clss in range(0,self.class_n):
				out[:,:,chan][im==clss]=np.array(self.im_gray_idx_to_rgb_table[clss][1][chan])
		deb.prints(out.shape)
		out=cv2.cvtColor(out.astype(np.uint8),cv2.COLOR_RGB2BGR)
		return out
	def val_set_get(self,mode='stratified',validation_split=0.2):
		clss_train_unique,clss_train_count=np.unique(self.patches['train']['label'].argmax(axis=3),return_counts=True)
		deb.prints(clss_train_count)
		self.patches['val']={'n':int(self.patches['train']['n']*validation_split)}
		
		#===== CHOOSE VAL IDX
		#mode='stratified'
		if mode=='random':
			self.patches['val']['idx']=np.random.choice(self.patches['train']['idx'],self.patches['val']['n'],replace=False)
			

			self.patches['val']['in']=self.patches['train']['in'][self.patches['val']['idx']]
			self.patches['val']['label']=self.patches['train']['label'][self.patches['val']['idx']]
		
		elif mode=='stratified':
			while True:
				self.patches['val']['idx']=np.random.choice(self.patches['train']['idx'],self.patches['val']['n'],replace=False)
				self.patches['val']['in']=self.patches['train']['in'][self.patches['val']['idx']]
				self.patches['val']['label']=self.patches['train']['label'][self.patches['val']['idx']]
		
				clss_val_unique,clss_val_count=np.unique(self.patches['val']['label'].argmax(axis=3),return_counts=True)
				
				if not np.array_equal(clss_train_unique,clss_val_unique):
					deb.prints(clss_train_unique)
					deb.prints(clss_val_unique)
					
					pass
				else:
					percentages=clss_val_count/clss_train_count
					deb.prints(percentages)
					#if np.any(percentages<0.1) or np.any(percentages>0.3):
					if np.any(percentages>0.23):
					
						pass
					else:
						break
		elif mode=='random_v2':
			while True:

				self.patches['val']['idx']=np.random.choice(self.patches['train']['idx'],self.patches['val']['n'],replace=False)
				

				self.patches['val']['in']=self.patches['train']['in'][self.patches['val']['idx']]
				self.patches['val']['label']=self.patches['train']['label'][self.patches['val']['idx']]
				clss_val_unique,clss_val_count=np.unique(self.patches['val']['label'].argmax(axis=3),return_counts=True)
						
				deb.prints(clss_train_unique)
				deb.prints(clss_val_unique)

				deb.prints(clss_train_count)
				deb.prints(clss_val_count)

				clss_train_count_in_val=clss_train_count[np.isin(clss_train_unique,clss_val_unique)]
				percentages=clss_val_count/clss_train_count_in_val
				deb.prints(percentages)
				#if np.any(percentages<0.1) or np.any(percentages>0.3):
				if np.any(percentages>0.26):
					pass
				else:
					break				

		deb.prints(self.patches['val']['idx'].shape)

		
		deb.prints(self.patches['val']['in'].shape)
		#deb.prints(data.patches['val']['label'].shape)
		
		self.patches['train']['in']=np.delete(self.patches['train']['in'],self.patches['val']['idx'],axis=0)
		self.patches['train']['label']=np.delete(self.patches['train']['label'],self.patches['val']['idx'],axis=0)
		#deb.prints(data.patches['train']['in'].shape)
		#deb.prints(data.patches['train']['label'].shape)
# ========== NetModel object implements model graph definition, train/testing, early stopping ================ #

class NetModel(NetObject):
	def __init__(self, batch_size_train=32, batch_size_test=200, epochs=30000, 
		patience=30, eval_mode='metrics', val_set=True, *args, **kwargs):

		super().__init__(*args, **kwargs)
		if self.debug >= 1:
			print("Initializing Model instance")
		self.val_set=val_set
		self.metrics = {'train': {}, 'test': {}, 'val':{}}
		self.batch = {'train': {}, 'test': {}}
		self.batch['train']['size'] = batch_size_train
		self.batch['test']['size'] = batch_size_test
		self.eval_mode = eval_mode
		self.epochs = epochs
		self.early_stop={'best':0,
					'count':0,
					'signal':False,
					'patience':patience}

		with open(self.report['best']['text_history_path'], "w") as text_file:
			text_file.write("epoch,oa,aa,f1,class_acc\n")

		with open(self.report['val']['history_path'], "w") as text_file:
			text_file.write("epoch,oa,aa,f1,class_acc\n")


	def transition_down(self, pipe, filters):
		pipe = Conv2D(filters, (3, 3), strides=(2, 2), padding='same')(pipe)
		pipe = keras.layers.BatchNormalization(axis=3)(pipe)
		pipe = Activation('relu')(pipe)
		#pipe = Conv2D(filters, (1, 1), padding='same')(pipe)
		#pipe = keras.layers.BatchNormalization(axis=3)(pipe)
		#pipe = Activation('relu')(pipe)
		
		return pipe

	def dense_block(self, pipe, filters):
		pipe = Conv2D(filters, (3, 3), padding='same')(pipe)
		pipe = keras.layers.BatchNormalization(axis=3)(pipe)
		pipe = Activation('relu')(pipe)
		return pipe

	def transition_up(self, pipe, filters):
		pipe = Conv2DTranspose(filters, (3, 3), strides=(
			2, 2), padding='same')(pipe)
		pipe = keras.layers.BatchNormalization(axis=3)(pipe)
		pipe = Activation('relu')(pipe)
		#pipe = Dropout(0.2)(pipe)
		#pipe = Conv2D(filters, (1, 1), padding='same')(pipe)
		#pipe = keras.layers.BatchNormalization(axis=3)(pipe)
		#pipe = Activation('relu')(pipe)
		return pipe

	def concatenate_transition_up(self, pipe1, pipe2, filters):
		pipe = keras.layers.concatenate([pipe1, pipe2], axis=3)
		pipe = self.transition_up(pipe, filters)
		return pipe

	def build(self):
		in_im = Input(shape=(self.t_len,self.patch_len, self.patch_len, self.channel_n))
		filters = 64

		#x = keras.layers.Permute((1,2,0,3))(in_im)
		x = keras.layers.Permute((2,3,1,4))(in_im)
		
		x = Reshape((self.patch_len, self.patch_len,self.t_len*self.channel_n), name='predictions')(x)
		#pipe = {'fwd': [], 'bckwd': []}
		c = {'init_up': 0, 'up': 0}
		pipe=[]

		# ================== Transition Down ============================ #
		pipe.append(self.transition_down(x, filters))  # 0 16x16
		pipe.append(self.transition_down(pipe[-1], filters*2))  # 1 8x8
		pipe.append(self.transition_down(pipe[-1], filters*4))  # 2 4x4
		pipe.append(self.transition_down(pipe[-1], filters*8))  # 2 4x4
		c['down']=len(pipe)-1 # Last down-layer idx
		
		# =============== Dense block; no transition ================ #
		#pipe.append(self.dense_block(pipe[-1], filters*16))  # 3 4x4

		# =================== Transition Up ============================= #
		c['up']=c['down'] # First up-layer idx 
		pipe.append(self.concatenate_transition_up(pipe[-1], pipe[c['up']], filters*8))  # 4 8x8
		c['up']-=1
		pipe.append(self.concatenate_transition_up(pipe[-1], pipe[c['up']], filters*4))  # 4 8x8
		c['up']-=1
		pipe.append(self.concatenate_transition_up(pipe[-1], pipe[c['up']], filters*2))  # 5
		c['up']-=1
		pipe.append(self.concatenate_transition_up(pipe[-1], pipe[c['up']], filters))  # 6

		out = Conv2D(self.class_n, (1, 1), activation='softmax',
					 padding='same')(pipe[-1])
		self.graph = Model(in_im, out)
		print(self.graph.summary())

	def build(self):
		in_im = Input(shape=(self.t_len,self.patch_len, self.patch_len, self.channel_n))
		filters = 64

		#x = keras.layers.Permute((1,2,0,3))(in_im)
		x = keras.layers.Permute((2,3,1,4))(in_im)
		
		x = Reshape((self.patch_len, self.patch_len,self.t_len*self.channel_n), name='predictions')(x)
		#pipe = {'fwd': [], 'bckwd': []}
		

		x = Conv2D(48, (3, 3), activation='relu',
					 padding='same')(x)



		c = {'init_up': 0, 'up': 0}
		pipe=[]

		# ================== Transition Down ============================ #
		pipe.append(self.transition_down(x, filters))  # 0 16x16
		pipe.append(self.transition_down(pipe[-1], filters*2))  # 1 8x8
		#pipe.append(self.transition_down(pipe[-1], filters*4))  # 2 4x4
		c['down']=len(pipe)-1 # Last down-layer idx
		
		# =============== Dense block; no transition ================ #
		#pipe.append(self.dense_block(pipe[-1], filters*16))  # 3 4x4

		# =================== Transition Up ============================= #
		c['up']=c['down'] # First up-layer idx 
		#pipe.append(self.concatenate_transition_up(pipe[-1], pipe[c['up']], filters*4))  # 4 8x8
		#c['up']-=1
		pipe.append(self.concatenate_transition_up(pipe[-1], pipe[c['up']], filters*2))  # 5
		c['up']-=1
		pipe.append(self.concatenate_transition_up(pipe[-1], pipe[c['up']], filters))  # 6

		out = Conv2D(self.class_n, (1, 1), activation='softmax',
					 padding='same')(pipe[-1])

		self.graph = Model(in_im, out)
		print(self.graph.summary())

	def build(self):
		deb.prints(self.t_len)
		in_im = Input(shape=(self.t_len,self.patch_len, self.patch_len, self.channel_n))

		#x = keras.layers.Permute((1,2,0,3))(in_im)
		x = keras.layers.Permute((2,3,1,4))(in_im)
		
		x = Reshape((self.patch_len, self.patch_len,self.t_len*self.channel_n), name='predictions')(x)
		out = DenseNetFCN((32, 32, self.t_len*self.channel_n), nb_dense_block=2, growth_rate=16, dropout_rate=0.2,
						nb_layers_per_block=2, upsampling_type='deconv', classes=self.class_n, 
						activation='softmax', batchsize=32,input_tensor=x)
		self.graph = Model(in_im, out)
		print(self.graph.summary())
	def compile(self, optimizer, loss='binary_crossentropy', metrics=['accuracy',metrics.categorical_accuracy],loss_weights=None):
		loss_weighted=weighted_categorical_crossentropy(loss_weights)
		#sparse_accuracy_ignoring_last_label()
		self.graph.compile(loss=loss_weighted, optimizer=optimizer, metrics=metrics)
		#self.graph.compile(loss=sparse_accuracy_ignoring_last_label, optimizer=optimizer, metrics=metrics)
		#self.graph.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
	def loss_weights_estimate(self,data):
		unique,count=np.unique(data.patches['train']['label'].argmax(axis=3),return_counts=True)
		unique=unique[1:] # No bcknd
		count=count[1:].astype(np.float32)
		weights_from_unique=np.max(count)/count
		deb.prints(weights_from_unique)
		deb.prints(np.max(count))
		deb.prints(count)
		deb.prints(unique)
		self.loss_weights=np.zeros(self.class_n)
		for clss in range(1,self.class_n): # class 0 is bcknd. Leave it in 0
			
			if clss in unique:
				self.loss_weights[clss]=weights_from_unique[unique==clss]
			else:
				self.loss_weights[clss]=0
		deb.prints(self.loss_weights)

	def test(self,data):
		data.patches['train']['batch_n'] = data.patches['train']['in'].shape[0]//self.batch['train']['size']
		data.patches['test']['batch_n'] = data.patches['test']['in'].shape[0]//self.batch['test']['size']

		batch = {'train': {}, 'test': {}}
		self.batch['train']['n'] = data.patches['train']['in'].shape[0] // self.batch['train']['size']
		self.batch['test']['n'] = data.patches['test']['in'].shape[0] // self.batch['test']['size']

		data.patches['test']['prediction']=np.zeros_like(data.patches['test']['label'])
		deb.prints(data.patches['test']['label'].shape)
		deb.prints(self.batch['test']['n'])

		self.metrics['test']['loss'] = np.zeros((1, 2))

		data.patches['test']['prediction']=np.zeros_like(data.patches['test']['label'])
		self.batch_test_stats=True

		for batch_id in range(0, self.batch['test']['n']):
			idx0 = batch_id*self.batch['test']['size']
			idx1 = (batch_id+1)*self.batch['test']['size']

			batch['test']['in'] = data.patches['test']['in'][idx0:idx1]
			batch['test']['label'] = data.patches['test']['label'][idx0:idx1]

			if self.batch_test_stats:
				self.metrics['test']['loss'] += self.graph.test_on_batch(
					batch['test']['in'], batch['test']['label'])		# Accumulated epoch

			data.patches['test']['prediction'][idx0:idx1]=self.graph.predict(batch['test']['in'],batch_size=self.batch['test']['size'])

		#====================METRICS GET================================================#
		deb.prints(data.patches['test']['label'].shape)		
		deb.prints(idx1)
		print("Epoch={}".format(epoch))	
		
		# Average epoch loss
		self.metrics['test']['loss'] /= self.batch['test']['n']
			
		# Get test metrics
		metrics=data.metrics_get(data.patches['test'],debug=1)
		print('oa={}, aa={}, f1={}, f1_wght={}'.format(metrics['overall_acc'],
			metrics['average_acc'],metrics['f1_score'],metrics['f1_score_weighted']))
	def train(self, data):

		# Random shuffle
		##data.patches['train']['in'], data.patches['train']['label'] = shuffle(data.patches['train']['in'], data.patches['train']['label'], random_state=0)

		# Normalize
		##data.patches['train']['in'] = normalize(data.patches['train']['in'].astype('float32'))
		##data.patches['test']['in'] = normalize(data.patches['test']['in'].astype('float32'))

		# Computing the number of batches
		data.patches['train']['batch_n'] = data.patches['train']['in'].shape[0]//self.batch['train']['size']
		data.patches['test']['batch_n'] = data.patches['test']['in'].shape[0]//self.batch['test']['size']

		deb.prints(data.patches['train']['batch_n'])

		self.train_loop(data)

	def early_stop_check(self,metrics,epoch,most_important='average_acc'):

		if metrics[most_important]>=self.early_stop['best']:
			self.early_stop['best']=metrics[most_important]
			self.early_stop['count']=0
			print("Best metric updated")
			
			#data.im_reconstruct(subset='test',mode='prediction')
		else:
			self.early_stop['count']+=1
			if self.early_stop["count"]>=self.early_stop["patience"]:
				self.early_stop["signal"]=True
			else:
				self.early_stop["signal"]=False
			
			
	def train_loop(self, data):
		print('Start the training')
		cback_tboard = keras.callbacks.TensorBoard(
			log_dir='../summaries/', histogram_freq=0, batch_size=self.batch['train']['size'], write_graph=True, write_grads=False, write_images=False)
		txt={'count':0,'val':{},'test':{}}
		txt['val']={'metrics':[],'epoch':[],'loss':[]}
		txt['test']={'metrics':[],'epoch':[],'loss':[]}
		
		
		#========= VAL INIT


		
		count,unique=np.unique(data.patches['val']['label'].argmax(axis=3),return_counts=True)
		print("Val label count,unique",count,unique)

		count,unique=np.unique(data.patches['train']['label'].argmax(axis=3),return_counts=True)
		print("Train count,unique",count,unique)
		
		count,unique=np.unique(data.patches['test']['label'].argmax(axis=3),return_counts=True)
		print("Test count,unique",count,unique)
		
		#==================== ESTIMATE BATCH NUMBER===============================#
		batch = {'train': {}, 'test': {}}
		self.batch['train']['n'] = data.patches['train']['in'].shape[0] // self.batch['train']['size']
		self.batch['test']['n'] = data.patches['test']['in'].shape[0] // self.batch['test']['size']

		data.patches['test']['prediction']=np.zeros_like(data.patches['test']['label'])
		deb.prints(data.patches['test']['label'].shape)
		deb.prints(self.batch['test']['n'])
		

		#if self.train_mode==

		#data.im_reconstruct(subset='test',mode='label')
		#for epoch in [0,1]:
		#==============================START TRAIN/TEST LOOP============================#
		for epoch in range(self.epochs):

			self.metrics['train']['loss'] = np.zeros((1, 2))
			self.metrics['test']['loss'] = np.zeros((1, 2))

			# Random shuffle the data
			##data.patches['train']['in'], data.patches['train']['label'] = shuffle(data.patches['train']['in'], data.patches['train']['label'])

			#=============================TRAIN LOOP=========================================#
			for batch_id in range(0, self.batch['train']['n']):
				
				idx0 = batch_id*self.batch['train']['size']
				idx1 = (batch_id+1)*self.batch['train']['size']

				batch['train']['in'] = data.patches['train']['in'][idx0:idx1]
				batch['train']['label'] = data.patches['train']['label'][idx0:idx1]

				self.metrics['train']['loss'] += self.graph.train_on_batch(
					batch['train']['in'], batch['train']['label'])		# Accumulated epoch

			# Average epoch loss
			self.metrics['train']['loss'] /= self.batch['train']['n']

			self.train_predict=True
			#if self.train_predict:



			#================== VAL LOOP=====================#
			data.patches['val']['prediction']=np.zeros_like(data.patches['val']['label'])
			self.metrics['val']['loss'] = self.graph.test_on_batch(
					data.patches['val']['in'], data.patches['val']['label'])
			data.patches['val']['prediction']=self.graph.predict(data.patches['val']['in'])

			if self.val_set:
				# Get val metrics

				metrics_val=data.metrics_get(data.patches['val'],debug=0)

				self.early_stop_check(metrics_val,epoch)
				if epoch==1000 or epoch==700 or epoch==500 or epoch==1200:
					self.early_stop['signal']=True
				else:
					self.early_stop['signal']=False
				if self.early_stop['signal']==True:
					self.graph.save('model_'+str(epoch)+'.h5')

				metrics_val['per_class_acc'].setflags(write=1)
				metrics_val['per_class_acc'][np.isnan(metrics_val['per_class_acc'])]=-1
				print(metrics_val['per_class_acc'])
				
				if epoch % 50 == 0:
					print("Writing val...")
					#print(txt['val']['metrics'])
					for i in range(len(txt['val']['metrics'])):
						data.metrics_write_to_txt(txt['val']['metrics'][i],np.squeeze(txt['val']['loss'][i]),
							txt['val']['epoch'][i],path=self.report['val']['history_path'])
					txt['val']['metrics']=[]
					txt['val']['loss']=[]
					txt['val']['epoch']=[]
				else:
					txt['val']['metrics'].append(metrics_val)
					txt['val']['loss'].append(self.metrics['val']['loss'])
					txt['val']['epoch'].append(epoch)

			
			#==========================TEST LOOP================================================#
			data.patches['test']['prediction']=np.zeros_like(data.patches['test']['label'])
			self.batch_test_stats=True

			for batch_id in range(0, self.batch['test']['n']):
				idx0 = batch_id*self.batch['test']['size']
				idx1 = (batch_id+1)*self.batch['test']['size']

				batch['test']['in'] = data.patches['test']['in'][idx0:idx1]
				batch['test']['label'] = data.patches['test']['label'][idx0:idx1]

				if self.batch_test_stats:
					self.metrics['test']['loss'] += self.graph.test_on_batch(
						batch['test']['in'], batch['test']['label'])		# Accumulated epoch

				data.patches['test']['prediction'][idx0:idx1]=self.graph.predict(batch['test']['in'],batch_size=self.batch['test']['size'])

			#====================METRICS GET================================================#
			deb.prints(data.patches['test']['label'].shape)		
			deb.prints(idx1)
			print("Epoch={}".format(epoch))	
			
			# Average epoch loss
			self.metrics['test']['loss'] /= self.batch['test']['n']
			
			# Get test metrics
			metrics=data.metrics_get(data.patches['test'],debug=1)
			
			# Check early stop and store results if they are the best
			if epoch % 50 == 0:
				print("Writing to file...")
				for i in range(len(txt['test']['metrics'])):

					data.metrics_write_to_txt(txt['test']['metrics'][i],np.squeeze(txt['test']['loss'][i]),
						txt['test']['epoch'][i],path=self.report['best']['text_history_path'])
				txt['test']['metrics']=[]
				txt['test']['loss']=[]
				txt['test']['epoch']=[]
				self.graph.save('my_model.h5')

			else:

				txt['test']['metrics'].append(metrics)
				txt['test']['loss'].append(self.metrics['test']['loss'])
				txt['test']['epoch'].append(epoch)

			#data.metrics_write_to_txt(metrics,np.squeeze(self.metrics['test']['loss']),
			#	epoch,path=self.report['best']['text_history_path'])
			#self.test_metrics_evaluate(data.patches['test'],metrics,epoch)
			#if self.early_stop['signal']==True:
			#	break


			deb.prints(metrics['confusion_matrix'])
			#metrics['average_acc'],metrics['per_class_acc']=self.average_acc(data['prediction_h'],data['label_h'])
			deb.prints(metrics['per_class_acc'])
			deb.prints(metrics_val['per_class_acc'])
			
			print('oa={}, aa={}, f1={}, f1_wght={}'.format(metrics['overall_acc'],
				metrics['average_acc'],metrics['f1_score'],metrics['f1_score_weighted']))
			print('val oa={}, aa={}, f1={}, f1_wght={}'.format(metrics_val['overall_acc'],
				metrics_val['average_acc'],metrics_val['f1_score'],metrics_val['f1_score_weighted']))
			print("Loss. Train={}, Val={}, Test={}".format(self.metrics['train']['loss'],
				self.metrics['val']['loss'],self.metrics['test']['loss']))

			#====================END METRICS GET===========================================#


flag = {"data_create": 2, "label_one_hot": True}
if __name__ == '__main__':
	#
	data = Dataset(patch_len=args.patch_len, patch_step_train=args.patch_step_train,
		patch_step_test=args.patch_step_test,exp_id=args.exp_id,
		path=args.path, t_len=args.t_len, class_n=args.class_n)
	if flag['data_create']==1:
		data.create()
	elif flag['data_create']==2:
		data.create_load()

	deb.prints(data.patches['train']['label'].shape)

	# === SELECT VALIDATION SET FROM TRAIN SET
	val_set=True
	#val_set_mode='stratified'
	val_set_mode='stratified'
	if val_set:
		data.val_set_get(val_set_mode,0.15)
	# ===
	
	deb.prints(data.patches['train']['label'].shape)
	deb.prints(data.patches['val']['label'].shape)
	deb.prints(data.patches['test']['label'].shape)
	
	unique,count=np.unique(data.patches['test']['label'].argmax(axis=3),return_counts=True)
	deb.prints(unique)
	deb.prints(count)
	data.label_unique=unique.copy()
	adam = Adam(lr=0.0001, beta_1=0.9)
	
	model = NetModel(epochs=args.epochs, patch_len=args.patch_len,
					 patch_step_train=args.patch_step_train, eval_mode=args.eval_mode,
					 batch_size_train=args.batch_size_train,batch_size_test=args.batch_size_test,
					 patience=args.patience,t_len=args.t_len,class_n=args.class_n,path=args.path,
					 val_set=val_set)
	model.build()
	model.loss_weights_estimate(data)
	#model.loss_weights=np.array([0.10259888, 0.2107262 , 0.1949083 , 0.20119307, 0.08057474,
	#   0.20999881]
	#model.loss_weights=np.array([0,0.04274219, 0.12199843, 0.11601452, 0.12202774, 0.12183601,                                      
	#   0.1099085 , 0.11723573, 0.00854844, 0.12208636, 0.11760209]).astype(np.float64)
	#model.loss_weights=np.array([0,1,1,1,1,1,1,1,1,1,1,1]).astype(np.float64)/11
	#model.loss_weights=np.array([0.        , 0.06051054, 0.13370499, 0.13283712, 0.13405423,
	#   0.        , 0.13397788, 0.11706449, 0.12805041, 0.03190986,
	#   0.        , 0.12789048]).astype(np.float64)
	#model.loss_weights=np.array([0,1.39506639e+00, 2.60304567e+02, 1.03202335e+02, 1.93963056e+04,0,0,
	 #  6.00161586e+00, 1.66971628e+01, 1.00000000e+00, 0,1.70606546e+01]).astype(np.float64)
	
	#model.loss_weights=np.array([0,1.41430758e+00, 2.70356529e+02, 9.87119740e+01, 2.17569417e+05,0,
 #2.43094320e+03, 5.97588208e+00, 1.65553794e+01, 1.00000000e+00,0,
 #1.69903102e+01])

	#model.loss_weights=np.array([0,1.37713256e+00,2.45637517e+02,6.08387646e+01,2.01024432e+03,0,3.79562360e+02, 6.26613648e+00, 1.70359689e+01, 1.00000000e+00,3.90646218e+03 ,1.59325845e+01])
	# Estimated with test
	#model.loss_weights=np.array([0,1.37852055e+00, 2.45986531e+02, 6.10172192e+01, 1.97027386e+03,0,3.71352450e+02 ,6.26956560e+00, 1.70878077e+01 ,1.00000000e+00,4.62502597e+03, 1.59184248e+01])
	# Estimated with train
	# This is an okay fcn
	###model.loss_weights=np.array([0, 1.42610349e+00  , 7.30082405e+02  , 1.75681165e+01 ,  1.11196404e+03, 0,3.93620317e+02 ,  8.51592741e+00  , 2.28322375e+01 ,  1.00000000e+00,2.34818768e+03  , 2.45846645e+01])
	#####model.loss_weights=np.array([0,1,1,1,1,0,1,1,1,1,1,1])
	#======end cv seq1
	##model.loss_weights=np.ones(12)
	##model.loss_weights[0]=0
	##model.loss_weights/=11

	#=========== cv se12

	#####model.loss_weights=np.array([0,2.87029782e+02 ,1.15257798e+02,0,0,0 ,5.51515771e+01 ,1.45716824e+01, 3.90684535e+01 ,1.00000000e+00 ,4.01800573e+03 ,4.20670477e+01])
	#model.loss_weights=np.array([0,])
	#=========== Hannover

	#model.loss_weights=np.array([0,3.32893347, 2.62162162, 1.06386569 ,1.95959596, 1.    ,     7.92583281,2.20570229, 1.17444351])

	metrics=['accuracy']
	#metrics=['accuracy',fmeasure,categorical_accuracy]
	model.compile(loss='binary_crossentropy',
				  optimizer=adam, metrics=metrics,loss_weights=model.loss_weights)
	model_load=False
	if model_load:
		model=load_model('/home/lvc/Documents/Jorg/sbsr/fcn_model/results/seq2_true_norm/models/model_1000.h5')
		model.test(data)
	
	if args.debug:
		deb.prints(np.unique(data.patches['train']['label']))
		deb.prints(data.patches['train']['label'].shape)
	model.train(data)
