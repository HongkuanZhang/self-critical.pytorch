from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
from lmdbdict import lmdbdict
from lmdbdict.methods import DUMPS_FUNC, LOADS_FUNC
import os
import numpy as np
import numpy.random as npr
import random
from functools import partial

import torch
import torch.utils.data as data

import multiprocessing
import six

# 这个类就是读取图像feature的loader
class HybridLoader:
    """
    If db_path is a director, then use normal file loading
    If lmdb, then load from lmdb
    The loading method depend on extention.

    in_memory: if in_memory is True, we save all the features in memory
               For individual np(y|z)s, we don't need to do that because the system will do this for us.
               Should be useful for lmdb or h5.
               (Copied this idea from vilbert)
    """
    def __init__(self, db_path, ext, in_memory=False):
        # image feature存储的地址
        self.db_path = db_path
        # 文件后缀
        self.ext = ext
        if self.ext == '.npy':
            self.loader = lambda x: np.load(six.BytesIO(x))
        else:
            def load_npz(x):
                x = np.load(six.BytesIO(x))
                return x['feat'] if 'feat' in x else x['z']  # normally it should be 'feat', but under cocotest_bu, the key is saved to be 'z' mistakenly.
            self.loader = load_npz
        if db_path.endswith('.lmdb'):
            self.db_type = 'lmdb'
            self.lmdb = lmdbdict(db_path, unsafe=True)
            self.lmdb._key_dumps = DUMPS_FUNC['ascii']
            self.lmdb._value_loads = LOADS_FUNC['identity']
        elif db_path.endswith('.pth'): # Assume a key,value dictionary
            self.db_type = 'pth'
            self.feat_file = torch.load(db_path)
            self.loader = lambda x: x
            print('HybridLoader: ext is ignored')
        elif db_path.endswith('h5'):
            self.db_type = 'h5'
            self.loader = lambda x: np.array(x).astype('float32')
        else:
            # 执行这个，因为我们给的db_path是data/cocotalk/这样的一个地址类型而不是文件
            self.db_type = 'dir'

        # 这个很好理解就是如果是in_memory则存储之前load过的图像feature，初始化的时候是空的
        # 之后随着loading慢慢加入进来
        self.in_memory = in_memory
        if self.in_memory:
            self.features = {}
    
    def get(self, key):
        # 从loader中加载数据
        # 如果是有存储memory，则先看看要加载的数据是否在memory中
        # 是的话直接加载
        if self.in_memory and key in self.features:
            # We save f_input because we want to save the
            # compressed bytes to save memory
            f_input = self.features[key]
        elif self.db_type == 'lmdb':
            f_input = self.lmdb[key]
        elif self.db_type == 'pth':
            f_input = self.feat_file[key]
        elif self.db_type == 'h5':
            f_input = h5py.File(self.db_path, 'r')[key]
        else:
            # 如果没有memory或者不在memory中，则从源文件读取
            # 地址+图像id+后缀来读取
            f_input = open(os.path.join(self.db_path, key + self.ext), 'rb').read()

        # 当前加载feature如果不在memory中则加入进来，后面可以直接调用
        if self.in_memory and key not in self.features:
            self.features[key] = f_input

        # load image
        # 通过上面的np.load(six.ByteIO(x))方法加载打开的文件，得到最后的feature。
        feat = self.loader(f_input)

        return feat

class Dataset(data.Dataset):
    
    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        # 这个类是继承dataset类的自己定义的dataset
        self.opt = opt
        self.seq_per_img = opt.seq_per_img
        
        # feature related options
        self.use_fc = getattr(opt, 'use_fc', True)
        self.use_att = getattr(opt, 'use_att', True)
        self.use_box = getattr(opt, 'use_box', 0)
        self.norm_att_feat = getattr(opt, 'norm_att_feat', 0)
        self.norm_box_feat = getattr(opt, 'norm_box_feat', 0)

        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_json)
        # 参数里重要的有：info为json文件，包含ix_to_word为字典，以及images为每个图像的信息({'id','split','file_path'})
        self.info = json.load(open(self.opt.input_json))
        if 'ix_to_word' in self.info:
            self.ix_to_word = self.info['ix_to_word']
            self.vocab_size = len(self.ix_to_word)
            print('vocab size is ', self.vocab_size)
        
        # open the hdf5 file
        print('DataLoader loading h5 file: ', opt.input_fc_dir, opt.input_att_dir, opt.input_box_dir, opt.input_label_h5)
        """
        Setting input_label_h5 to none is used when only doing generation.
        For example, when you need to test on coco test set.
        """
        # h5_label_file为包含所有split的所有encoded captions的h5文件
        if self.opt.input_label_h5 != 'none':
            self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')
            # load in the sequence data
            seq_size = self.h5_label_file['labels'].shape
            self.label = self.h5_label_file['labels'][:]
            self.seq_length = seq_size[1]
            print('max sequence length in data is', self.seq_length)
            # load the pointers in full to RAM (should be small enough)
            
            # strat_ix和end_ix包含每个图片对应的cap的起始和终结index
            # 注意这里start是从1开始，结束也是start+cap数，后面会解释为什么不是0开始。
            self.label_start_ix = self.h5_label_file['label_start_ix'][:]
            self.label_end_ix = self.h5_label_file['label_end_ix'][:]
        else:
            self.seq_length = 1

        self.data_in_memory = getattr(opt, 'data_in_memory', False)
        
        # 加载图像feature loader
        self.fc_loader = HybridLoader(self.opt.input_fc_dir, '.npy', in_memory=self.data_in_memory)
        self.att_loader = HybridLoader(self.opt.input_att_dir, '.npz', in_memory=self.data_in_memory)
        self.box_loader = HybridLoader(self.opt.input_box_dir, '.npy', in_memory=self.data_in_memory)

        self.num_images = len(self.info['images']) # self.label_start_ix.shape[0]
        print('read %d image features' %(self.num_images))

        # separate out indexes for each of the provided splits
        # 将各个图像的三种类型的信息添加到对应split string的字典中
        self.split_ix = {'train': [], 'val': [], 'test': []}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if not 'split' in img:
                self.split_ix['train'].append(ix)
                self.split_ix['val'].append(ix)
                self.split_ix['test'].append(ix)
            elif img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
            elif opt.train_only == 0: # restval
                self.split_ix['train'].append(ix)

        print('assigned %d images to split train' %len(self.split_ix['train']))
        print('assigned %d images to split val' %len(self.split_ix['val']))
        print('assigned %d images to split test' %len(self.split_ix['test']))

    def get_captions(self, ix, seq_per_img):
        # 这个函数用来获得图像的全部caps
        # 这里反正是保证每个图像对应seq_per_img数量的GT caps
        # 之前说的start_index从1开始也是因为对于flickr数据会有每个图片对于caps数量不同的情况
        # 对于coco其实这里处理之后完全没影响
        # fetch the sequence labels
        ix1 = self.label_start_ix[ix] - 1 #label_start_ix starts from 1
        ix2 = self.label_end_ix[ix] - 1
        ncap = ix2 - ix1 + 1 # number of captions available for this image
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

        if ncap < seq_per_img:
            # we need to subsample (with replacement)
            seq = np.zeros([seq_per_img, self.seq_length], dtype = 'int')
            for q in range(seq_per_img):
                ixl = random.randint(ix1,ix2)
                seq[q, :] = self.label[ixl, :self.seq_length]
        else:
            ixl = random.randint(ix1, ix2 - seq_per_img + 1)
            seq = self.label[ixl: ixl + seq_per_img, :self.seq_length]

        return seq

    def collate_func(self, batch, split):
        seq_per_img = self.seq_per_img

        fc_batch = []
        att_batch = []
        label_batch = []

        wrapped = False

        infos = []
        gts = []

        for sample in batch:
            # fetch image
            tmp_fc, tmp_att, tmp_seq, \
                ix, it_pos_now, tmp_wrapped = sample
            if tmp_wrapped:
                wrapped = True

            fc_batch.append(tmp_fc)
            att_batch.append(tmp_att)
            
            # tmp_label形状为(5，max_len)
            tmp_label = np.zeros([seq_per_img, self.seq_length + 2], dtype = 'int')
            if hasattr(self, 'h5_label_file'):
                # if there is ground truth
                # tmp_seq为一个图像对应的5个caps，所以是把tmp_seq的每一行赋给tmp_label的每一行，不过tmp_label的每行的开头和结尾都为0，估计后面会插入start和end字符？
                tmp_label[:, 1 : self.seq_length + 1] = tmp_seq
            
            # label_batch加入tmp_label
            label_batch.append(tmp_label)

            # Used for reward evaluation
            if hasattr(self, 'h5_label_file'):
                # if there is ground truth
                # 每个图片对应的所有GT caps给到gts
                gts.append(self.label[self.label_start_ix[ix] - 1: self.label_end_ix[ix]])
            else:
                gts.append([])
        
            # record associated info as well
            info_dict = {}
            # 图像在图像list中的index
            info_dict['ix'] = ix
            # 图像的id
            info_dict['id'] = self.info['images'][ix]['id']
            # 图像的file_path
            info_dict['file_path'] = self.info['images'][ix].get('file_path', '')
            infos.append(info_dict)

        # #sort by att_feat length
        # fc_batch, att_batch, label_batch, gts, infos = \
        #     zip(*sorted(zip(fc_batch, att_batch, np.vsplit(label_batch, batch_size), gts, infos), key=lambda x: len(x[1]), reverse=True))
        # 这里是个zip()和zip(*)常常联用的压缩-解压缩法
        # 先是zip压缩，然后对压缩的内容通过sort进行排序，通过zip(*sort())得到解压缩返回给各个变量
        # 但是这里没看懂key=lambda x:0是怎么个意思，一般都是Lambda x:x[0]这样的让zip中的第0个参数作为排序的key之类的，但是这里是个常数0，没看懂
        # 感觉上面他注释的那个反而是合理的。。
        fc_batch, att_batch, label_batch, gts, infos = \
            zip(*sorted(zip(fc_batch, att_batch, label_batch, gts, infos), key=lambda x: 0, reverse=True))
        data = {}
        # 把list中的sample数据stack成一个batch的数据
        data['fc_feats'] = np.stack(fc_batch)
        
        # merge att_feats
        # 原来att_features是对于每个图片的数量都是不同的，也就是X*C的形状，X对每个图片都不等
        # 所以这里显示求得batch中的最大的X，即max_att_len，然后创建一个(batch_size,max_len,C)的全0向量作为batch向量
        # 之后把每个X*C放入到里面，长度小于X的都自动被0 pad了，等于X的就直接赋值过去
        max_att_len = max([_.shape[0] for _ in att_batch])
        data['att_feats'] = np.zeros([len(att_batch), max_att_len, att_batch[0].shape[1]], dtype = 'float32')
        for i in range(len(att_batch)):
            data['att_feats'][i, :att_batch[i].shape[0]] = att_batch[i]
        
        # 因为上面用0去pad了，后面计算的时候需要忽略那些pad位置，因此这里生成mask
        # mask的形状是和上面的前两维度相同的(b_s,max_len)，然后第二维度根据实际长度，从0到实际长度位置都为1，之后的位置都为0，类似于下面的样子
        # e.g. data['att_masks'] = [[1,1,1,1,0,0],[1,1,1,0,0,0],[1,1,0,0,0,0]]
        data['att_masks'] = np.zeros(data['att_feats'].shape[:2], dtype='float32')
        for i in range(len(att_batch)):
            data['att_masks'][i, :att_batch[i].shape[0]] = 1
        
        # 如果attention features数量都相等则不需要mask，numpy的size方法可以得到所有维度之和，如(2,4,5)形状的array的size为40
        # set att_masks to None if attention features have same length
        if data['att_masks'].sum() == data['att_masks'].size:
            data['att_masks'] = None

        data['labels'] = np.vstack(label_batch)
        # generate mask
        # 为label生成mask，即每个caption都通过pad 0达到了最大长度
        # 所以这里先统计每个caption中非0元素，然后生成[1,1,1,..,0,0]的mask向量
        # 其中，1的数量为非0元素(caption实际长度)+2(start和end)，0的数量为max_len+2减去1的数量
        nonzeros = np.array(list(map(lambda x: (x != 0).sum()+2, data['labels'])))
        mask_batch = np.zeros([data['labels'].shape[0], self.seq_length + 2], dtype = 'float32')
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
        data['masks'] = mask_batch
        
        # 本来labels和masks分别是包含一个batch中所有数据的labels和masks，而没有对每个sample的5个labels和masks分割开来(即形状为(b_s*5,-1))
        # 因此这里通过reshape变为(b_s,5,-1)的形状,labels的-1应该等于max_len，而mask的-1等于max_len+2
        data['labels'] = data['labels'].reshape(len(batch), seq_per_img, -1)
        data['masks'] = data['masks'].reshape(len(batch), seq_per_img, -1)

        data['gts'] = gts # all ground truth captions of each images
        data['bounds'] = {'it_pos_now': it_pos_now, # the it_pos_now of the last sample
                          'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos

        # data是个字典，keys有fc_feats,att_feats,att_masks,labels,masks,gts,bounds,infos
        # 每个key对应的value是一个batch的数据，为torch tensor
        data = {k:torch.from_numpy(v) if type(v) is np.ndarray else v for k,v in data.items()} # Turn all ndarray to torch tensor

        return data

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        这里得到的是每个sample，batch数据是把多个sample给到collate_fn来得到的
        我们平时一般是用的default的collate，即每个sample是一个tensor或者{key:tensor}
        则默认一个batch的数据就是把batch size的tensor集合起来的tensor，即外侧增加batch维度
        最后这个getitem输出的tuple会给到collate_fn，它会负责把数据batch起来
        """
        # 这个index不是一个数，而是从sampler返回的三元组，详细见Mysampler部分
        ix, it_pos_now, wrapped = index #self.split_ix[index]
        # Transformer的情况是用att不用fc，所以执行下面这个
        if self.use_att:
            # 获取图像feature
            att_feat = self.att_loader.get(str(self.info['images'][ix]['id']))
            # Reshape为K*C形状，K为feature数(不固定，每个图片对应的K不同，后面要padding)，C为feature维度(channel数量)
            # Reshape to K x C
            att_feat = att_feat.reshape(-1, att_feat.shape[-1])
            if self.norm_att_feat:
                att_feat = att_feat / np.linalg.norm(att_feat, 2, 1, keepdims=True)
            if self.use_box:
                box_feat = self.box_loader.get(str(self.info['images'][ix]['id']))
                # devided by image width and height
                x1,y1,x2,y2 = np.hsplit(box_feat, 4)
                h,w = self.info['images'][ix]['height'], self.info['images'][ix]['width']
                box_feat = np.hstack((x1/w, y1/h, x2/w, y2/h, (x2-x1)*(y2-y1)/(w*h))) # question? x2-x1+1??
                if self.norm_box_feat:
                    box_feat = box_feat / np.linalg.norm(box_feat, 2, 1, keepdims=True)
                att_feat = np.hstack([att_feat, box_feat])
                # sort the features by the size of boxes
                att_feat = np.stack(sorted(att_feat, key=lambda x:x[-1], reverse=True))
        else:
            att_feat = np.zeros((0,0), dtype='float32')
        # fc不使用，所以为空向量(np.zeros((0))为形状为0的空向量)
        if self.use_fc:
            try:
                fc_feat = self.fc_loader.get(str(self.info['images'][ix]['id']))
            except:
                # Use average of attention when there is no fc provided (For bottomup feature)
                fc_feat = att_feat.mean(0)
        else:
            fc_feat = np.zeros((0), dtype='float32')
        # 获取图像对应的多个caps
        if hasattr(self, 'h5_label_file'):
            seq = self.get_captions(ix, self.seq_per_img)
        else:
            seq = None
        
        return (fc_feat,
                att_feat, seq,
                ix, it_pos_now, wrapped)

    def __len__(self):
        return len(self.info['images'])

# train文件首先会调用这个得到DataLoader
# 这个class会返回各个split的dataloader
# class返回的loader参数包括opts，batch_size，dataset(这里有点特别)，以及各个split对应的dataloader
class DataLoader:
    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.dataset = Dataset(opt)

        # Initialize loaders and iters
        self.loaders, self.iters = {}, {}
        for split in ['train', 'val', 'test']:
            if split == 'train':
                # 就是返回一个sampler，对于train数据会是随机sample
                sampler = MySampler(self.dataset.split_ix[split], shuffle=True, wrap=True)
            else:
                # 对于test和eval是按顺序sample
                sampler = MySampler(self.dataset.split_ix[split], shuffle=False, wrap=False)
            # 注意！！这里的data.DataLoader中的data是torch.utils.data，而不是data文件夹
            # 所以这里的DataLoader不是307行的那个，而是pytorch官方的loader
            self.loaders[split] = data.DataLoader(dataset=self.dataset,
                                                  batch_size=self.batch_size,
                                                  sampler=sampler,
                                                  pin_memory=True,
                                                  num_workers=4, # 4 is usually enough
                                                  collate_fn=partial(self.dataset.collate_func, split=split), # 会把split传入给collate方程
                                                  drop_last=False)
            self.iters[split] = iter(self.loaders[split])

    def get_batch(self, split):
        try:
            data = next(self.iters[split])
        except StopIteration:
            self.iters[split] = iter(self.loaders[split])
            data = next(self.iters[split])
        return data

    def reset_iterator(self, split):
        self.loaders[split].sampler._reset_iter()
        self.iters[split] = iter(self.loaders[split])

    def get_vocab_size(self):
        return self.dataset.get_vocab_size()

    @property
    def vocab_size(self):
        return self.get_vocab_size()

    def get_vocab(self):
        return self.dataset.get_vocab()

    def get_seq_length(self):
        return self.dataset.get_seq_length()

    @property
    def seq_length(self):
        return self.get_seq_length()

    def state_dict(self):
        def get_prefetch_num(split):
            if self.loaders[split].num_workers > 0:
                return (self.iters[split]._send_idx - self.iters[split]._rcvd_idx) * self.batch_size
            else:
                return 0
        return {split: loader.sampler.state_dict(get_prefetch_num(split)) \
                    for split, loader in self.loaders.items()}

    def load_state_dict(self, state_dict=None):
        if state_dict is None:
            return
        for split in self.loaders.keys():
            self.loaders[split].sampler.load_state_dict(state_dict[split])

# 这个mysampler会决定给到__getitem__的变量idx是什么样子
class MySampler(data.sampler.Sampler):
    def __init__(self, index_list, shuffle, wrap):
        self.index_list = index_list
        self.shuffle = shuffle
        self.wrap = wrap
        # wrap是一个信号，告诉sampler当sample全部batch的时候是否发出停止iteration的信号
        # 对于train dataset设置为True即停止不发出信号，对于test设置为False即停止要发出停止信号
        # if wrap, there will be not stop iteration called
        # wrap True used during training, and wrap False used during test.
        self._reset_iter()

    def __iter__(self):
        return self

    def __next__(self):
        # 在没有完成sample一个epoch中的所有example的时候，wrapped为False
        # 即返回给getitem的wrapped变量为False
        wrapped = False
        # 当iter_counter等于index_list长度，即sample最后一个batch的时候
        # 根据self.wrap决定是变更wrapped为True还是发出停止信号
        if self.iter_counter == len(self._index_list):
            # 初始化index list，对于train split会打乱
            self._reset_iter()
            if self.wrap:
                wrapped = True
            else:
                raise StopIteration()
        if len(self._index_list) == 0: # overflow when 0 samples
            return None
        # Sampler会返回三个值，一个是当前iter对应的index，一个是进行的iter数量，最后一个是wrapped信号
        elem = (self._index_list[self.iter_counter], self.iter_counter+1, wrapped)
        self.iter_counter += 1
        return elem

    def next(self):
        return self.__next__()

    # 初始化时候会调用，即对split的index list进行shuffle或者保持原样，然后返回。
    def _reset_iter(self):
        if self.shuffle:
            rand_perm = npr.permutation(len(self.index_list))
            self._index_list = [self.index_list[_] for _ in rand_perm]
        else:
            self._index_list = self.index_list

        self.iter_counter = 0

    def __len__(self):
        return len(self.index_list)

    def load_state_dict(self, state_dict=None):
        if state_dict is None:
            return
        self._index_list = state_dict['index_list']
        self.iter_counter = state_dict['iter_counter']

    def state_dict(self, prefetched_num=None):
        prefetched_num = prefetched_num or 0
        return {
            'index_list': self._index_list,
            'iter_counter': self.iter_counter - prefetched_num
        }

    
