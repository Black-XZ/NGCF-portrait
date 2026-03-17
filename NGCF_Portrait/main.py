import torch
import torch.optim as optim
import pandas as pd

from NGCF import *
from utility.helper import *
from utility.batch_test import *

import warnings

warnings.filterwarnings('ignore')
from time import time

if __name__ == '__main__':

    import os

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # 根据是否有 GPU 设置设备
    if torch.cuda.is_available():
        args.device = torch.device('cuda:' + str(args.gpu_id))
    else:
        args.device = torch.device('cpu')

    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

    # 获取用户聚类信息
    user_clusters = data_generator.user_clusters
    user_clusters_value = data_generator.user_clusters_value
    user_feature_dim = data_generator.user_feature_dim

    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)

    model = NGCF(data_generator.n_users,
                 data_generator.n_items,
                 norm_adj,
                 args,
                 user_clusters,
                 user_clusters_value,
                 user_feature_dim).to(args.device)

    t0 = time()
    """
    *********************************************************
    Train.
    """
    cur_best_pre_0, stopping_step = 0, 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    epoch_data = []

    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss = 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample()
            pos_ratings = torch.tensor([data_generator.R[u - 1, i - 1] for u, i in zip(users, pos_items)],
                                       dtype=torch.float,
                                       device=args.device)

            # 调用forward
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users,
                                                                           pos_items,
                                                                           neg_items,
                                                                           drop_flag=args.node_dropout_flag)

            batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
                                                                              pos_i_g_embeddings,
                                                                              neg_i_g_embeddings,
                                                                              pos_weights=pos_ratings)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss

        if (epoch + 1) % 1 != 0:  # 每隔 1 或 10 （自己取值）轮运行一次continue后的代码
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    (epoch + 1), time() - t1, loss, mf_loss, emb_loss)
                print(perf_str)
            continue

        t2 = time()

        users_to_test = list(data_generator.test_set.keys())
        ret = test(model, users_to_test, drop_flag=False)

        t3 = time()

        # 提取 ndcg@10 和 hr@10 的值
        ndcg_at_10 = ret['ndcg'][0]
        hr_at_10 = ret['hit_ratio'][0]

        print(f"Epoch {(epoch + 1)} [Train Time: {t2 - t1:.1f}s, Test Time: {t3 - t2:.1f}s]:")
        print(f"  Loss: {loss:.5f}, NDCG@10: {ndcg_at_10:.5f}, HR@10: {hr_at_10:.5f}")

        # 将当前 epoch 的数据保存
        epoch_data.append([(epoch + 1), loss.item(), ndcg_at_10, hr_at_10])
        epoch_df = pd.DataFrame(epoch_data, columns=['Epoch', 'Loss', 'NDCG@10', 'HR@10'])

        # 导出为 CSV 文件
        csv_path = 'epoch_performance.csv'
        epoch_df.to_csv(csv_path, index=False)
        print(f"Epoch performance data has been saved to {csv_path}")

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       ((epoch + 1), t2 - t1, t3 - t2, loss, mf_loss, emb_loss, ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=5)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        # if should_stop == True:
        #     break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            torch.save(model.state_dict(), args.weights_path + str(epoch) + '.pkl')
            print('save the weights in path: ', args.weights_path + str(epoch) + '.pkl')

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 ((idx + 1), time() - t0,
                  '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)

    # 作图
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use('Agg')

    Ks = eval(args.Ks)

    # 提取 ndcg@10 和 hr@10 的值
    ndcg_at_10 = ndcgs[:, 0]  # 第一列是 K=10 时的 ndcg 值
    hr_at_10 = hit[:, 0]  # 第一列是 K=10 时的 hr 值

    # 打印 ndcg@10 和 hr@10 的最佳值
    best_ndcg_at_10 = max(ndcg_at_10)
    best_hr_at_10 = max(hr_at_10)
    print(f"Best NDCG@10: {best_ndcg_at_10:.5f}")
    print(f"Best HR@10: {best_hr_at_10:.5f}")

    # 绘制 ndcg、hr 和 recall 随 K 变化的图像
    plt.figure(figsize=(12, 8))

    # 绘制 NDCG 曲线
    plt.subplot(2, 2, 1)
    plt.plot(Ks, ndcgs[idx], marker='o', label=f'Best Epoch {idx+1}')
    plt.xlabel('K')
    plt.ylabel('NDCG')
    plt.title('NDCG@K')
    plt.legend()

    # 绘制 HR 曲线
    plt.subplot(2, 2, 2)
    plt.plot(Ks, hit[idx], marker='o', label=f'Best Epoch {idx+1}')
    plt.xlabel('K')
    plt.ylabel('HR')
    plt.title('HR@K')
    plt.legend()

    # 绘制 Recall 曲线
    plt.subplot(2, 2, 3)
    plt.plot(Ks, recs[idx], marker='o', label=f'Best Epoch {idx+1}')
    plt.xlabel('K')
    plt.ylabel('Recall')
    plt.title('Recall@K')
    plt.legend()

    # 调整布局
    plt.tight_layout()
    plt.savefig(f'NGCFp_output_{args.epoch}.png')
    plt.show()
    plt.close()
    print('\nThe graph has been saved.')
