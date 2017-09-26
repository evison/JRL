import pandas as pd
import numpy as np
import tensorflow as tf


path = 'recommend'
file = open(path, 'rb')
all = file.read()
recom_items = eval(all)


path = 'true_test'
file = open(path, 'rb')
all = file.read()
true_purchased = eval(all)



def top_k(pre_top_k, true_top_k):
    user_number = len(pre_top_k)
    correct = []
    co_length = []
    tr_length = []
    pre_length = []
    p = []
    r = []
    f = []
    hit_number = 0
    for i in range(user_number):
        temp = []
        for j in pre_top_k[i]:
            if j in true_top_k[i]:
                temp.append(j)
        if len(temp):
            hit_number = hit_number + 1.0
        co_length.append(len(temp))
        pre_length.append(len(pre_top_k[i]))
        tr_length.append(len(true_top_k[i]))
        correct.append(temp)

    #print co_length
    real_user = 0

    for i in range(user_number):
        if tr_length[i] == 0:
            p_t = 0.0
        else:
            real_user += 1
            p_t = co_length[i] / float(pre_length[i])
        if tr_length[i] == 0:
            r_t = 0.0
        else:
            r_t = co_length[i] / float(tr_length[i])
        p.append(p_t)
        r.append(r_t)
        if p_t != 0 or r_t != 0:
            f.append(2.0 * p_t * r_t / (p_t + r_t))
        else:
            f.append(0.0)


    print real_user
    hit_ratio = hit_number / user_number
    print np.array(p).mean(),np.array(r).mean(),np.array(f).mean(),hit_ratio
    return p, r, f, hit_ratio
def NDCG_k(recommend_list, purchased_list):
    user_number = len(recommend_list)
    u_ndgg = []
    for i in range(user_number):
        temp = 0
        Z_u = 0
        for j in range(min(len(recommend_list[i]), len(purchased_list[i]))):
            Z_u = Z_u + 1 / np.log2(j + 2)
        for j in range(len(recommend_list[i])):
            if recommend_list[i][j] in purchased_list[i]:
                temp = temp + 1 / np.log2(j + 2)

        if Z_u == 0:
            temp = 0
        else:
            temp = temp / Z_u
        u_ndgg.append(temp)
    NDCG = np.array(u_ndgg).mean()
    print NDCG
    return NDCG


ground = []
pre = []
for (k,v) in true_purchased.items():
    ground.append(v[::-1])
    pre.append(recom_items[k][::-1])


top_d = [5, 10, 20, 30, 40, 50]
for i in range(len(top_d)):
    pre_d = np.array(pre)[:, :top_d[i]]
    top_k(pre_d, ground)
    NDCG_k(pre_d, ground)



'''
p = 0.000252
'''
