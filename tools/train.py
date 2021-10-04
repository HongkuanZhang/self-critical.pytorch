from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import time
import os
from six.moves import cPickle
import traceback
from collections import defaultdict

import captioning.utils.opts as opts
import captioning.models as models
from captioning.data.dataloader import *
import skimage.io
import captioning.utils.eval_utils as eval_utils
import captioning.utils.misc as utils
from captioning.utils.rewards import init_scorer, get_self_critical_reward
from captioning.modules.loss_wrapper import LossWrapper


def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def train(opt):

    ################################
    # Build dataloader
    ################################
    # 得到多个split的DataLoader
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    ##########################
    # Initialize infos
    ##########################
    infos = {
        'iter': 0,
        'epoch': 0,
        'loader_state_dict': None,
        'vocab': loader.get_vocab(),
    }
    # Load old infos(if there is) and check if models are compatible
    # 这里是继续训练时候需要加载以前的训练信息，我们暂时不考虑这个
    if opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl')):
        with open(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl'), 'rb') as f:
            infos = utils.pickle_load(f)
            saved_model_opt = infos['opt']
            need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert getattr(saved_model_opt, checkme) == getattr(opt, checkme), "Command line argument and saved model disagree on '%s' " % checkme
    infos['opt'] = opt

    #########################
    # Build logger
    #########################
    # naive dict logger
    histories = defaultdict(dict)
    # 这里也是继续训练需要执行的内容，不用管
    if opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
        with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl'), 'rb') as f:
            histories.update(utils.pickle_load(f))

    # tensorboard logger
    tb_summary_writer = SummaryWriter(opt.checkpoint_path)

    ##########################
    # Build model
    ##########################
    # 得到ix_to_word字典
    opt.vocab = loader.get_vocab()
    # 得到模型
    model = models.setup(opt).cuda()
    del opt.vocab
    # 加载之前训练的checkpoints
    # Load pretrained weights:
    if opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from, 'model.pth')):
        model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model.pth')))
    
    # Wrap generation model with loss function(used for training)
    # This allows loss function computed separately on each machine
    # 这里主要是通过Wrap和data parallel让各个machine独立计算loss
    lw_model = LossWrapper(model, opt)
    # Wrap with dataparallel
    dp_model = torch.nn.DataParallel(model)
    dp_model.vocab = getattr(model, 'vocab', None)  # nasty
    
    # wrap之后的模型名称
    dp_lw_model = torch.nn.DataParallel(lw_model)

    ##########################
    #  Build optimizer
    ##########################
    # 初始化optimizer，根据opts可以初始化为”loss不下降时会降低学习率的“optimizer
    if opt.noamopt:
        assert opt.caption_model in ['transformer', 'bert', 'm2transformer'], 'noamopt can only work with transformer'
        optimizer = utils.get_std_opt(model, optim_func=opt.optim, factor=opt.noamopt_factor, warmup=opt.noamopt_warmup)
    elif opt.reduce_on_plateau:
        optimizer = utils.build_optimizer(model.parameters(), opt)
        optimizer = utils.ReduceLROnPlateau(optimizer,
                                            factor=opt.reduce_on_plateau_factor,
                                            patience=opt.reduce_on_plateau_patience)
    else:
        optimizer = utils.build_optimizer(model.parameters(), opt)
    # Load the optimizer
    if opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from,"optimizer.pth")):
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))

    #########################
    # Get ready to start
    #########################
    iteration = infos['iter']
    epoch = infos['epoch']
    # For back compatibility
    if 'iterators' in infos:
        infos['loader_state_dict'] = {split: {'index_list': infos['split_ix'][split], 'iter_counter': infos['iterators'][split]} for split in ['train', 'val', 'test']}
    # 这里暂时不懂，但是由于默认infos中的'loader_state_dict'为None，所以返回也为None
    # 补充:新联过程中info词典中的loader_state_dict会被赋值并且最后和模型一起保存
    # 所以对于从0训练的情况这个值为0，而如果是继续训练则会从之前训练的checkpoint中读取值，如124行所示
    loader.load_state_dict(infos['loader_state_dict'])
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)
    if opt.noamopt:
        optimizer._step = iteration
    # flag indicating finish of an epoch
    # Always set to True at the beginning to initialize the lr or etc.
    epoch_done = True
    # Assure in training mode
    dp_lw_model.train()

    # Start training
    # 开始训练！！！
    try:
        while True:
            # Stop if reaching max epochs
            if epoch >= opt.max_epochs and opt.max_epochs != -1:
                break

            if epoch_done:
                # 如果不是自动降低lr的情况需要根据设定的开始降低LR的epoch数来手动降低LR
                if not opt.noamopt and not opt.reduce_on_plateau:
                    # Assign the learning rate
                    # 如果当前epoch数大于应该降低LR的epoch数
                    if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                        # 当前epoch已经比设定多出frac个epochs
                        frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                        # 因此要降低的倍数为decay_rate的frac个次方(每个epoch的lr变为上一个epoch的lr*decay rate)，并赋予为current_lr
                        decay_factor = opt.learning_rate_decay_rate  ** frac
                        opt.current_lr = opt.learning_rate * decay_factor
                    else:
                        opt.current_lr = opt.learning_rate
                    # optimizer的lr设定为更新后的lr
                    utils.set_lr(optimizer, opt.current_lr) # set the decayed rate
                
                # 如果设置了schedule sampling，且当前epoch触发了sampling则执行
                # Assign the scheduled sampling prob
                if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                    # sampling的概率随着epoch增加成倍增加(每相隔scheduled_sampling_increase_every多个epochs，概率增加一个prob)
                    frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                    opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                    model.ss_prob = opt.ss_prob

                # If start self critical training
                # 是否触发sc training，是的话要用到cached_tokens来计算cider
                if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                    sc_flag = True
                    init_scorer(opt.cached_tokens)
                else:
                    sc_flag = False
                
                # If start structure loss training
                # 暂时没太懂这个loss是什么，感兴趣可以看modules/losses中的class StructureLosses 
                if opt.structure_after != -1 and epoch >= opt.structure_after:
                    struc_flag = True
                    init_scorer(opt.cached_tokens)
                else:
                    struc_flag = False
                
                # 这里也没太懂，貌似是扔掉最坏的模型？
                # 看后面懂了，并不是扔掉模型，而是在计算loss的时候扔掉一部分值最小的loss(只保留top的losses)
                # 这个flag要和drop_worse_rate同步设置，然后会扔掉int(长度*rate)的最低值。
                if opt.drop_worst_after != -1 and epoch >= opt.drop_worst_after:
                    drop_worst_flag = True
                else:
                    drop_worst_flag = False

                epoch_done = False
                    
            start = time.time()
            # 如果进行warm_up，若warmup epoch为N，则第一个epoch的学习率是lr*1/N，然后每个epoch增长lr*1/N，第N个epoch增长为lr
            if opt.use_warmup and (iteration < opt.noamopt_warmup):
                opt.current_lr = opt.learning_rate * (iteration+1) / opt.noamopt_warmup
                utils.set_lr(optimizer, opt.current_lr)
            
            # Load data from train split (0)
            #########################
            # 加载batch数据！
            #########################
            
            # 这里后面可以看看它定义的DataLoader的get_batch方法
            data = loader.get_batch('train')
            print('Read data:', time.time() - start)

            torch.cuda.synchronize()
            start = time.time()
            
            # 读取fc_feats，att_feats，labels，masks和att_masks数据
            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
            tmp = [_ if _ is None else _.cuda() for _ in tmp]
            fc_feats, att_feats, labels, masks, att_masks = tmp
            
            #########################
            # 将数据输入给模型！
            #########################
            optimizer.zero_grad()
            # 5个数据以及3个flag给到模型，不太清除gts那个是什么，还是需要看get_batch方法
            # 计算模型输出
            model_out = dp_lw_model(fc_feats, att_feats, labels, masks, att_masks, data['gts'], torch.arange(0, len(data['gts'])), sc_flag, struc_flag, drop_worst_flag)

            # 计算loss
            if not drop_worst_flag:
                loss = model_out['loss'].mean()
            else:
                # drop_worst的情况下去掉最小的一些loss值再求平均值
                loss = model_out['loss']
                loss = torch.topk(loss, k=int(loss.shape[0] * (1-opt.drop_worst_rate)), largest=False)[0].mean()

            loss.backward()
            
            # 进行梯度剪裁
            if opt.grad_clip_value != 0:
                getattr(torch.nn.utils, 'clip_grad_%s_' %(opt.grad_clip_mode))(model.parameters(), opt.grad_clip_value)
            
            optimizer.step()
            train_loss = loss.item()
            
            torch.cuda.synchronize()
            end = time.time()
            
            # 根据三个flag打印loss
            # 可以看到structure和sc的情况只是从model_out中读取对应信息，说明默认已经计算了这些loss/reward
            if struc_flag:
                print("iter {} (epoch {}), train_loss = {:.3f}, lm_loss = {:.3f}, struc_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(iteration, epoch, train_loss, model_out['lm_loss'].mean().item(), model_out['struc_loss'].mean().item(), end - start))
            elif not sc_flag:
                print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(iteration, epoch, train_loss, end - start))
            else:
                print("iter {} (epoch {}), avg_reward = {:.3f}, time/batch = {:.3f}" \
                    .format(iteration, epoch, model_out['reward'].mean(), end - start))

            # Update the iteration and epoch
            # 训练step每次计算输出后都更新
            iteration += 1
            # 估计加载数据时候，加载最后一个batch的话返回的data中的['bounds']['wrapped']会是True，则说明epoch完成，更新epoch
            if data['bounds']['wrapped']:
                epoch += 1
                epoch_done = True

            # Write the training loss summary
            # 如果iteration满足log的steps数，则开始进行loss logging
            if (iteration % opt.losses_log_every == 0):
                tb_summary_writer.add_scalar('train_loss', train_loss, iteration)
                if opt.noamopt:
                    opt.current_lr = optimizer.rate()
                elif opt.reduce_on_plateau:
                    opt.current_lr = optimizer.current_lr
                tb_summary_writer.add_scalar('learning_rate', opt.current_lr, iteration)
                tb_summary_writer.add_scalar('scheduled_sampling_prob', model.ss_prob, iteration)
                if sc_flag:
                    tb_summary_writer.add_scalar('avg_reward', model_out['reward'].mean(), iteration)
                elif struc_flag:
                    tb_summary_writer.add_scalar('lm_loss', model_out['lm_loss'].mean().item(), iteration)
                    tb_summary_writer.add_scalar('struc_loss', model_out['struc_loss'].mean().item(), iteration)
                    tb_summary_writer.add_scalar('reward', model_out['reward'].mean().item(), iteration)
                    tb_summary_writer.add_scalar('reward_var', model_out['reward'].var(1).mean(), iteration)

                histories['loss_history'][iteration] = train_loss if not sc_flag else model_out['reward'].mean()
                histories['lr_history'][iteration] = opt.current_lr
                histories['ss_prob_history'][iteration] = model.ss_prob

            # update infos
            # 更新infos词典
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['loader_state_dict'] = loader.state_dict()
            
            # make evaluation on validation set, and save model
            # 根据iteration或者是(epoch结束时要保存的设置)触发保存model
            if (iteration % opt.save_checkpoint_every == 0 and not opt.save_every_epoch) or \
                (epoch_done and opt.save_every_epoch):
                # eval model
                eval_kwargs = {'split': 'val',
                                'dataset': opt.input_json}
                eval_kwargs.update(vars(opt))
                
                # 这里后面看看，eval_split会对模型进行操作得到val_loss以及predictions
                val_loss, predictions, lang_stats = eval_utils.eval_split(
                    dp_model, lw_model.crit, loader, eval_kwargs)

                # 如果是reduce_on_plateau，则进行scheduler_step
                if opt.reduce_on_plateau:
                    if 'CIDEr' in lang_stats:
                        optimizer.scheduler_step(-lang_stats['CIDEr'])
                    else:
                        optimizer.scheduler_step(val_loss)
                # Write validation result into summary
                tb_summary_writer.add_scalar('validation loss', val_loss, iteration)
                if lang_stats is not None:
                    for k,v in lang_stats.items():
                        tb_summary_writer.add_scalar(k, v, iteration)
                histories['val_result_history'][iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

                # Save model if is improving on validation result
                # language_eval表示是否在eval set上进行生成文本的评价，然后用这个评价判断模型是否表现更好进而决定是否保存
                # 如果是1则为进行评价(BLEU,ROUGE等)，然后下面选择的是用CIDEr来评价
                if opt.language_eval == 1:
                    current_score = lang_stats['CIDEr']
                else:
                    # 这里表示的是loss加负号作为分数，即loss(正数)越小则-loss越大
                    # 并不是current score减去loss！那是 - = 不是 = - 。。。
                    current_score = - val_loss

                best_flag = False

                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True

                # Dump miscalleous informations
                infos['best_val_score'] = best_val_score
                
                # 保存的时候还是用model这个变量而不是lw_model呢
                utils.save_checkpoint(opt, model, infos, optimizer, histories)
                
                # 如果每个save point都保存模型，则进行保存，名字上附加epoch数(如果指定每个epoch都保存)或者iteration数
                if opt.save_history_ckpt:
                    utils.save_checkpoint(opt, model, infos, optimizer,
                        append=str(epoch) if opt.save_every_epoch else str(iteration))

                # 保存最好的模型并在名称加上best字符
                if best_flag:
                    utils.save_checkpoint(opt, model, infos, optimizer, append='best')

    except (RuntimeError, KeyboardInterrupt):
        # 以外退出或者键盘干扰退出都会保存模型
        print('Save ckpt on exception ...')
        utils.save_checkpoint(opt, model, infos, optimizer)
        print('Save ckpt done.')
        stack_trace = traceback.format_exc()
        print(stack_trace)


opt = opts.parse_opt()
train(opt)
