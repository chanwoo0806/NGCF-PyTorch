'''
Created on March 24, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
'''

import torch
import torch.optim as optim

from NGCF import NGCF
from utility.helper import *
from utility.batch_test import *

import warnings
warnings.filterwarnings('ignore')
import os
import time
from tensorboardX import SummaryWriter


if __name__ == '__main__':
    print(f'''
          Dataset: {args.dataset}
          Epoch: {args.epoch}
          Patience: {args.patience} on NDCG@20
          Embedding Dim: {args.embed_size}
          Learning Rate: {args.lr}
          Batch Size: {args.batch_size}
          Num of Layers: {len(eval(args.layer_size))}
          L2 Norm: {eval(args.regs)[0]}
          ''')
    # args.device = torch.device('cuda:' + str(args.gpu_id))
    args.device = torch.device('cuda')

    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

    args.node_dropout = eval(args.node_dropout)
    # args.mess_dropout = eval(args.mess_dropout)
    args.mess_dropout = [0.1]*len(eval(args.layer_size))

    model = NGCF(data_generator.n_users,
                 data_generator.n_items,
                 norm_adj,
                 args).to(args.device)

    t0 = time.time()

    if args.tensorboard:
        ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
        TB_PATH = os.path.join(os.path.join(ROOT_PATH, f"tensorboard/{args.dataset}"), time.strftime("%m-%d-%Hh%Mm%Ss")) \
            + f"-l{len(eval(args.layer_size))}-d{args.embed_size}-r{eval(args.regs)[0]}"
        writer = SummaryWriter(TB_PATH)
    """
    *********************************************************
    Train.
    """
    cur_best_pre_0, stopping_step = 0, 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    for epoch in range(args.epoch):
        t1 = time.time()
        loss, mf_loss, emb_loss = 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample()
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users,
                                                                           pos_items,
                                                                           neg_items,
                                                                           drop_flag=args.node_dropout_flag)

            batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
                                                                              pos_i_g_embeddings,
                                                                              neg_i_g_embeddings)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss

        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time.time() - t1, loss, mf_loss, emb_loss)
                print(perf_str)
                
            if args.tensorboard:
                writer.add_scalar("Train/Loss", loss, epoch)
            continue

        t2 = time.time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(model, users_to_test, drop_flag=False)

        t3 = time.time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)
            
        if args.tensorboard:
            writer.add_scalar("Train/Loss", loss, epoch)
            for i, k in enumerate(eval(args.Ks)):
                writer.add_scalar(f"Test/Recall_{k}", ret['recall'][i], epoch)
                writer.add_scalar(f"Test/Precision_{k}", ret['precision'][i], epoch)
                writer.add_scalar(f"Test/NDCG_{k}", ret['ndcg'][i], epoch)
                writer.add_scalar(f"Test/HitRatio_{k}", ret['hit_ratio'][i], epoch)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['ndcg'][-1], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=args.patience)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['ndcg'][-1] == cur_best_pre_0 and args.save_flag == 1:
            torch.save(model.state_dict(), args.weights_path + str(epoch) + '.pkl')
            print('save the weights in path: ', args.weights_path + str(epoch) + '.pkl')
    
    if args.tensorboard:
        writer.close()

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_ndcg_0 = max(ndcgs[:, -1])
    idx = list(ndcgs[:, -1]).index(best_ndcg_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time.time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)