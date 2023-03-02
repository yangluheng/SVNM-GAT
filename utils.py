import datetime
import pickle
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import os
import multiprocessing
import scipy.spatial

def get_target(triples,file_paths):
    def read_dict(file_paths):
        ent2id_dict = {}
        ids = []
        for file_path in file_paths:
            id = set()
            with open(file_path, "r", encoding="utf-8") as fr:
                for line in fr:
                    params = line.strip("\n").split("\t")
                    ent2id_dict[params[1]] = int(params[0])
                    id.add(int(params[0]))
            ids.append(id)
        return ent2id_dict, ids
    ent2id_dict, ids = read_dict([file_paths + "/ent_ids_" + str(i) for i in range(1,3)])
    
    r_hs, r_ts = {}, {}
    for (h, r, t) in triples:
        if r not in r_hs:
            r_hs[r] = set()
        if r not in r_ts:
            r_ts[r] = set()
        r_hs[r].add(h)
        r_ts[r].add(t)
    assert len(r_hs) == len(r_ts)
    return r_hs, r_ts, ids

def normalize_adj(adj): #邻接矩阵正交化
    adj = sp.coo_matrix(adj)    #构建稀疏矩阵
    rowsum = np.array(adj.sum(1))   #矩阵行求和
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()   #求和的-0.5次方，d_inv_sqrt：[0.024000768036865967,0.05572782125753528],19054
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.   #用于检查数字是否为无穷大(正数或负数)，如果是inf，转换成0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)   #构建对角形矩阵，d_mat_inv_sqrt:[[0.005320654540360169],[1.0]],19054
    return d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).T

def load_triples(file_name):
    triples = []
    entity = set()
    rel = set([0])
    time = set([0])
    for line in open(file_name,'r'):
        params = line.split()
        if len(params) == 5:
         ##   head, r, tail, ts, te = int(params[0]), int(params[1]), int(params[2]), 0,0
            head, r, tail, ts, te = int(params[0]), int(params[1]), int(params[2]), int(params[3]), int(params[4])###      
            entity.add(head); entity.add(tail); rel.add(r+1); time.add(ts+1); time.add(te+1)
            triples.append([head,r+1,tail,ts+1,te+1])
        else:
         ##   head, r, tail, t = int(params[0]), int(params[1]), int(params[2]), 0### by setting all timestamps to 0, we get TU-GNN
            head, r, tail, t = int(params[0]), int(params[1]), int(params[2]), int(params[3])###
            entity.add(head); entity.add(tail); rel.add(r+1); time.add(t+1)###  ebti
            triples.append([head,r+1,tail,t+1])####
    return entity,rel,triples,time

def load_alignment_pair(file_name):
    alignment_pair = []
    c = 0
    for line in open(file_name,'r'):
        e1,e2 = line.split()
        alignment_pair.append((int(e1),int(e2)))
    return alignment_pair

def get_matrix(triples,entity,rel,time):    #{0, 1, 2, 3, 4] 494
        ent_size = max(entity)+1    #19054
        rel_size = (max(rel) + 1)
        time_size = (max(time)+1)   ###
        print(ent_size,rel_size,time_size)  #19054 494 4018
        adj_matrix = sp.lil_matrix((ent_size,ent_size)) #建立(19054, 19054)空矩阵，实体的邻接矩阵
        adj_features = sp.lil_matrix((ent_size,ent_size))
        radj = []   #四元组邻接矩阵，1230210，[[0, 155, 1, 3604], [155, 0, 495, 3604]]
        rel_in = np.zeros((ent_size,rel_size))  #实体-关系的矩阵，(19054, 494)，[[0.  0.  81]]
        rel_out = np.zeros((ent_size,rel_size)) #实体-关系的矩阵，(19054, 494)



     
        for i in range(max(entity)+1):  #实体的邻接矩阵，对角线设为1
            adj_features[i,i] = 1

        r_h = {}
        r_t = {}
        if len(triples[0])<5:
            time_link = np.zeros((ent_size,time_size))  #（19054，4018），实体-时间矩阵
            # time_link = np.zeros((ent_size,time_size),dtype="float32")  #（19054，4018），实体-时间矩阵
            for h,r,t,tau in triples:   #构造实体的邻接矩阵
                adj_matrix[h,t] = 1; adj_matrix[t,h] = 1;
                adj_features[h,t] = 1; adj_features[t,h] = 1;
                radj.append([h,t,r,tau]); radj.append([t,h,r+rel_size,tau]);    #引入反转关系的四元组列表，反转关系：r+rel_size
                time_link[h][tau] +=1 ; time_link[t][tau] +=1   #头实体对应的时间点，尾实体对应的时间点
                rel_out[h][r] += 1; rel_in[t][r] += 1

                #每种关系对应的头尾实体
                r_h[r] = h              #{1: 11, 2: 410, 3}
                r_t[r] = t
        else:
            time_link = np.zeros((ent_size,time_size))
            for h,r,t,ts,te in triples:
                adj_matrix[h,t] = 1; adj_matrix[t,h] = 1;
                adj_features[h,t] = 1; adj_features[t,h] = 1;
                radj.append([h,t,r,ts]); radj.append([t,h,r+rel_size,te]);
                time_link[h][te] +=1 ; time_link[h][te] +=1 #头实体对应的开始时间
                time_link[t][ts] +=1 ; time_link[t][te] +=1 #尾实体对应的结束时间
                rel_out[h][r] += 1; rel_in[t][r] += 1

                # 每种关系对应的头尾实体
                r_h[r] = h
                r_t[r] = t
        # print("头尾：",r_h)
        # print("头尾：",r_t)

        count = -1  #198261
        s = set()   #{'10386 10313'}
        d = {}  #d，{0:32, 1:13},  198262  节点度
        r_index,t_index,r_val = [],[],[]
        #r_index:[[0,499],[0,507]],1230210，同一头尾实体的关系的数量      t_index:[[0,1594],[0,970]], 1230210，同一头尾实体的时间数量     r_val:[1,1,1,1],每个四元组的关系    1230210
        for h,t,r,tau in sorted(radj,key=lambda x: x[0]*10e10+x[1]*10e5):   #x:x[]字母可以随意修改，排序方式按照中括号[]里面的维度进行排序，[0]按照第一维排序，[2]按照第三维排序,这是按第一维排序
            if ' '.join([str(h),str(t)]) in s:
                r_index.append([count,r])
                t_index.append([count,tau])
                r_val.append(1)
                d[count] += 1
            else:
                count += 1
                d[count] = 1
                s.add(' '.join([str(h),str(t)]))    #'0 4'
                r_index.append([count,r])   #[0, 499]
                t_index.append([count,tau]) #[0, 1594]
                r_val.append(1)
        for i in range(len(r_index)):   #计算正交化的度
            r_val[i] /= d[r_index[i][0]]
            

        time_features  = time_link
        time_features = normalize_adj(sp.lil_matrix(time_features))     #csr_matrix,  (19054, 4018),    (0, 6)	0.0005760368663594471  (0, 12)	0.0011520737327188942
        rel_features = np.concatenate([rel_in,rel_out],axis=1)
        # rel_features = np.concatenate([rel_in,rel_out],axis=1,dtype="float32")
        adj_features = normalize_adj(adj_features)  #(19054, 19054),  (0,0) 0.003999999999999999,(0,4)  0.003999999999999999
        rel_features = normalize_adj(sp.lil_matrix(rel_features))   #(19054, 988),  (0,2) 2.830936473785528e-05 (0,3) 1.0000000000000002
        return adj_matrix,r_index,r_val,t_index,adj_features,rel_features,time_features



#
# def gcn_load_data(input_folder, is_two=False, is_three=False, is_four=False):
#     kg1, kg2, sup_ent1, sup_ent2, ref_ent1, ref_ent2, total_tri_num, total_e_num, total_r_num, rel_id_mapping = \
#         read_dbp15k_input(input_folder)
#     linked_ents = set(sup_ent1 + sup_ent2 + ref_ent1 + ref_ent2)
#     enhanced_triples1, enhanced_triples2 = enhance_triples(kg1, kg2, sup_ent1, sup_ent2)
#     ori_triples = kg1.triple_list + kg2.triple_list
#     triples = remove_unlinked_triples(ori_triples + list(enhanced_triples1) + list(enhanced_triples2), linked_ents)
#     rel_ht_dict = generate_rel_ht(triples)
#
#     saved_data_path = input_folder + 'alinet_saved_data.pkl'
#     if os.path.exists(saved_data_path):
#         print('load saved adj data from', saved_data_path)
#         adj = pickle.load(open(saved_data_path, 'rb'))
#     else:
#         one_adj, _ = no_weighted_adj(total_e_num, triples, is_two_adj=False)
#         adj = [one_adj]
#         two_hop_triples1, two_hop_triples2 = None, None
#         three_hop_triples1, three_hop_triples2 = None, None
#         if is_two:
#             two_hop_triples1 = generate_2hop_triples(kg1, linked_ents=linked_ents)
#             two_hop_triples2 = generate_2hop_triples(kg2, linked_ents=linked_ents)
#             triples = two_hop_triples1 | two_hop_triples2
#             two_adj, _ = no_weighted_adj(total_e_num, triples, is_two_adj=False)
#             adj.append(two_adj)
#         if is_three:
#             three_hop_triples1 = generate_3hop_triples(kg1, two_hop_triples1, linked_ents=linked_ents)
#             three_hop_triples2 = generate_3hop_triples(kg2, two_hop_triples2, linked_ents=linked_ents)
#             triples = three_hop_triples1 | three_hop_triples2
#             three_adj, _ = no_weighted_adj(total_e_num, triples, is_two_adj=False)
#             adj.append(three_adj)
#         if is_four:
#             four_hop_triples1 = generate_3hop_triples(kg1, three_hop_triples1, linked_ents=linked_ents)
#             four_hop_triples2 = generate_3hop_triples(kg2, three_hop_triples2, linked_ents=linked_ents)
#             triples = four_hop_triples1 | four_hop_triples2
#             four_adj, _ = no_weighted_adj(total_e_num, triples, is_two_adj=False)
#             adj.append(four_adj)
#         print('save adj data to', saved_data_path)
#         pickle.dump(adj, open(saved_data_path, 'wb'))
#
#     return adj, kg1, kg2, sup_ent1, sup_ent2, ref_ent1, ref_ent2, total_tri_num, \
#            total_e_num, total_r_num, rel_id_mapping, rel_ht_dict




import networkx as nx

def find123Nei(G, node):
    nodes = list(nx.nodes(G))
    nei1_li = []
    nei2_li = []
    nei3_li = []
    for FNs in list(nx.neighbors(G, node)):  # find 1_th neighbors
        nei1_li .append(FNs)

    for n1 in nei1_li:
        for SNs in list(nx.neighbors(G, n1)):  # find 2_th neighbors
            nei2_li.append(SNs)
    nei2_li = list(set(nei2_li) - set(nei1_li))
    if node in nei2_li:
        nei2_li.remove(node)

    for n2 in nei2_li:
        for TNs in nx.neighbors(G, n2):
            nei3_li.append(TNs)
    nei3_li = list(set(nei3_li) - set(nei2_li) - set(nei1_li))
    if node in nei3_li:
        nei3_li.remove(node)

    return nei1_li, nei2_li, nei3_li

h = nx.Graph()
# h.add_nodes_from(list(range(1, 8)))
h.add_edges_from([(1, 2), (1, 3), (1, 5), (1, 4), (2, 8), (2, 6), (3, 6), (4, 7),(6,6)])
neighs = find123Nei(h,1)
print(neighs)


def load_data(lang,train_ratio = 1000):      #200
    """
    entity1: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  9517
    rel1 :{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 248
    triples1:[[0, 1, 155, 3604], [1, 2, 4, 61]] 307552
    time1 :{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 4018
    """
    entity1,rel1,triples1,time1 = load_triples(lang + 'triples_1')
    entity2,rel2,triples2,time2 = load_triples(lang + 'triples_2')
    
    """
    train_pair: [(4, 9553), (8, 9533)] 1000（k=25）
    dev_pair:   [(5405, 13847)] 8366
    """
    train_pair = load_alignment_pair(lang + 'sup_pairs')
    dev_pair = load_alignment_pair(lang + 'ref_pairs')
    dev_pair = train_pair[train_ratio:]+dev_pair
    train_pair = train_pair[:train_ratio]
    #adj_matrix:(0,0)0.0007137758743754462  (0,4)1.0    (19054, 19054)
    #r_index：[[0, 499], [0, 507]]   1230210，同一头尾实体的关系的数量
    #r_val:[1,1,1,1],每个四元组的关系    1230210
    #t_index:t_index:[[0,1594],[0,970]], 1230210，同一头尾实体的时间数量
    #adj_features,rel_features,time_features:正交化的实体、关系、时间的邻接矩阵
    adj_matrix,r_index,r_val,t_index,adj_features,rel_features,time_features = get_matrix(triples1+triples2,
    entity1.union(entity2),rel1.union(rel2),time1.union(time2))

    #多跳
    # one_adj, _ = no_weighted_adj(total_e_num, triples, is_two_adj=False)
    # adj = [one_adj]
    # two_hop_triples1, two_hop_triples2 = None, None
    # three_hop_triples1, three_hop_triples2 = None, None
    # if is_two:
    #     two_hop_triples1 = generate_2hop_triples(kg1, linked_ents=linked_ents)
    #     two_hop_triples2 = generate_2hop_triples(kg2, linked_ents=linked_ents)
    #     triples = two_hop_triples1 | two_hop_triples2
    #     two_adj, _ = no_weighted_adj(total_e_num, triples, is_two_adj=False)
    #     adj.append(two_adj)
    # if is_three:
    #     three_hop_triples1 = generate_3hop_triples(kg1, two_hop_triples1, linked_ents=linked_ents)
    #     three_hop_triples2 = generate_3hop_triples(kg2, two_hop_triples2, linked_ents=linked_ents)
    #     triples = three_hop_triples1 | three_hop_triples2
    #     three_adj, _ = no_weighted_adj(total_e_num, triples, is_two_adj=False)
    #     adj.append(three_adj)
    # if is_four:
    #     four_hop_triples1 = generate_3hop_triples(kg1, three_hop_triples1, linked_ents=linked_ents)
    #     four_hop_triples2 = generate_3hop_triples(kg2, three_hop_triples2, linked_ents=linked_ents)
    #     triples = four_hop_triples1 | four_hop_triples2
    #     four_adj, _ = no_weighted_adj(total_e_num, triples, is_two_adj=False)
    #     adj.append(four_adj)

    
    return np.array(train_pair),np.array(dev_pair),adj_matrix,np.array(r_index),np.array(r_val),np.array(t_index),adj_features,rel_features,time_features###

def get_hits(vec, test_pair, wrank = None, top_k=(1, 5, 10)):
    Lvec = np.array([vec[e1] for e1, e2 in test_pair])
    Rvec = np.array([vec[e2] for e1, e2 in test_pair])
    
    Lvec = Lvec / np.linalg.norm(Lvec,axis=-1,keepdims=True)
    Rvec = Rvec / np.linalg.norm(Rvec,axis=-1,keepdims=True)
    sim_o = -Lvec.dot(Rvec.T)
    sim = sim_o.argsort(-1)
    if wrank is not None:
        srank = np.zeros_like(sim)
        for i in range(srank.shape[0]):
            for j in range(srank.shape[1]):
                srank[i,sim[i,j]] = j
        rank = np.max(np.concatenate([np.expand_dims(srank,-1),np.expand_dims(wrank,-1)],-1),axis=-1)
        sim = rank.argsort(-1)
    top_lr = [0] * len(top_k)
    MRR_lr = 0
    for i in range(Lvec.shape[0]):
        rank = sim[i, :]
        rank_index = np.where(rank == i)[0][0]
        MRR_lr += 1/(rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    top_rl = [0] * len(top_k)
    MRR_rl = 0
    sim = sim_o.argsort(0)
    for i in range(Rvec.shape[0]):
        rank = sim[:,i]
        rank_index = np.where(rank == i)[0][0]
        MRR_rl += 1/(rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1
    print('For each left:')
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    print('MRR: %.3f' % (MRR_lr / Lvec.shape[0]))  
    print('For each right:')
    for i in range(len(top_rl)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))
    print('MRR: %.3f' % (MRR_rl / Rvec.shape[0]))



def get_hits1(vec, vec_r, l1, M0, ref_data, rel_type, test_pair, sim_e, sim_r, wrank = None,top_k=(1, 3, 10)):
    ref = set()  # 对齐实体

    time11 = time.localtime(time.time())
    time11 = time.strftime("%Y-%m-%d %H:%M:%S", time11)
    print("开始时间：", time11)
    start = datetime.datetime.now()

    for pair in ref_data:  # 对齐实体
        ref.add((pair[0], pair[1]))
    r_num = vec_r.shape[0] // 2  # 关系向量长度/2，分开两种关系，一种未加入反转
    print(r_num)
    kg = {}  # 节点对应的关系-节点
    rel_ent = {}  # 关系对应的实体
    for tri in M0:  # 加入反转的图，邻接矩阵
        if tri[0] == tri[2]:  # 5729	13	5729
            continue
        if tri[0] not in kg:  # 先创建集合
            kg[tri[0]] = set()
        if tri[2] not in kg:
            kg[tri[2]] = set()
        if tri[1] not in rel_ent:
            rel_ent[tri[1]] = set()

        kg[tri[0]].add((tri[1], tri[2]))  # 头结点对应的关系-尾节点
        kg[tri[2]].add((tri[1], tri[0]))  # 尾节点对应的关系-头节点
        rel_ent[tri[1]].add((tri[0], tri[2]))  # 关系对应的头节点和尾节点
        # print(kg)
        # print(rel_ent)

    L = np.array([e1 for e1, e2 in test_pair])  # 从测试实体对中选择图1节点
    R = np.array([e2 for e1, e2 in test_pair])  # 从测试实体对中选择图2节点
    Lvec = vec[L]  # 实体向量
    Rvec = vec[R]  # 实体向量
    # print(L)
    # print(R)
    # print(Lvec)
    # print(Rvec)

    sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
    print("实体sim", sim)
    if sim_e is None:
        sim_e = sim

    if wrank is not None:
        srank = np.zeros_like(sim)
        for i in range(srank.shape[0]):
            for j in range(srank.shape[1]):
                srank[i,sim[i,j]] = j
        rank = np.max(np.concatenate([np.expand_dims(srank,-1),np.expand_dims(wrank,-1)],-1),axis=-1)
        sim = rank.argsort(-1)

    R_set = {}  # 图2节点对应的图1节点和相似度

    for i in range(len(L)):  # 图1测试节点
        j = sim_e[i, :].argsort()[0]  # 找实体对应相似度最小的下标
        if sim_e[i, j] >= 5:  # 大于等于给定阈值
            continue
        if j in R_set and sim_e[i, j] < R_set[j][1]:
            ref.remove((L[R_set[j][0]], R[j]))  # 更新对齐实体对，移除原来相似度较大的实体对
            ref.add((L[i], R[j]))  # 放入较小的
            R_set[j] = (i, sim_e[i, j])  # 更新图2节点对应的图1节点和相似度
        if j not in R_set:
            ref.add((L[i], R[j]))  # 放入实体对
            R_set[j] = (i, sim_e[i, j])  # 更新图2节点对应的图1节点和相似度

    if sim_r is None:  # 计算关系距离
        vec_r = vec_r.numpy()
        sim_r = scipy.spatial.distance.cdist(vec_r[:l1], vec_r[l1:r_num], metric='cityblock')
    print("关系sim", sim_r)

    ref_r = set()  # 关系和反转关系的相似度
    for i in range(l1):  # l1图1关系数量
        j = sim_r[i, :].argsort()[0]  # 关系对应相似度最小的下标
        if sim_r[i, j] < 3:  # 小于指定阈值
            ref_r.add((i, j + l1))  # 更新
            ref_r.add((i + r_num, j + l1 + r_num))

    e_index = 0
    for i in range(len(L)):  # 图1节点
        rank = sim[i, :].argsort()[:800]  # 前100
        e_index = len(rank)
        for j in rank:
            if R[j] in kg:  # 如果前100个有图2节点在图中
                match_num = 0  # 匹配数量
                for n_1 in kg[L[i]]:  # 图1节点对应的关系-尾实体
                    for n_2 in kg[R[j]]:  # 图2节点对应的关系-头结点
                        if (n_1[1], n_2[1]) in ref and (n_1[0], n_2[0]) in ref_r:  # 图1尾实体和图2头实体在对齐数据集，图1关系和图2关系在对齐
                            w = rel_type[str(n_1[0]) + ' ' + str(n_1[1])] * rel_type[str(n_2[0]) + ' ' + str(n_2[1])]
                            match_num += w  # 匹配数量加1 公式（6）（7）（8）
                sim[i, j] -= 10 * match_num / (len(kg[L[i]]) + len(kg[R[j]]) + match_num)  # 更新相似度  公式（9）

    # file.writelines("********************************")
    # sim_r = scipy.spatial.distance.cdist(vec_r[:l1], vec_r[l1:r_num], metric='cityblock')
    # print(sim_r)
    # r_index = 0
    # for i in range(l1):  # 图1关系数量
    #     rank = sim_r[i, :].argsort()[:200]  # 前20
    #     r_index = len(rank)
    #     for j in rank:
    #         if i in rel_ent and j + l1 in rel_ent:  # 如果图1关系在对齐关系和图2关系在对齐关系
    #             match_num = 0
    #             for n_1 in rel_ent[i]:  # 图1关系对应的头尾节点
    #                 for n_2 in rel_ent[j + l1]:  # 图2关系对应的头尾节点
    #                     if (n_1[0], n_2[0]) in ref and (n_1[1], n_2[1]) in ref:  # 图1头实体和图2头实体在对齐数据集
    #                         match_num += 1  # 匹配数量加1 公式（6）（7）（8）
    #             sim_r[i, j] -= 200 * match_num / (len(rel_ent[i]) + len(rel_ent[j + l1]) + match_num)  # 更新 公式（10）


    mrr_l = []  # 图1MRR
    mrr_r = []  # 图2MRR
    mr_l = []  # 图1MR
    mr_r = []  # 图2MR

    top_lr = [0] * len(top_k)  # hit@k比率，图1对图2
    for i in range(Lvec.shape[0]):  # 图1
        rank = sim[i, :].argsort()
        rank_index = np.where(rank == i)[0][0]  # 返回满足条件的最小下标
        # print(rank)
        mrr_l.append(1.0 / (rank_index + 1))  # 放入MRR
        mr_l.append(rank_index + 1)
        for j in range(len(top_k)):  # 计算hit@k
            if rank_index < top_k[j]:
                top_lr[j] += 1  # 排名前k数量加1

    top_rl = [0] * len(top_k)  # hit@k比率，图2对图1
    # print(top_rl)

    for i in range(Rvec.shape[0]):
        rank = sim[:, i].argsort()
        rank_index = np.where(rank == i)[0][0]  # 返回满足条件的
        mrr_r.append(1.0 / (rank_index + 1))  # 放入MRR
        mr_r.append(rank_index + 1)
        for j in range(len(top_k)):  # 计算hit@k
            if rank_index < top_k[j]:
                top_rl[j] += 1  # 排名前k数量加1

    # file.write("sim_e：" + str(sim_e))
    # file.write("sim_r：" + str(sim_r))

    print('Entity Alignment (left):')

    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    print('MRR: %.4f' % (np.mean(mrr_l)))
    print('MR: %.4f' % (np.mean(mr_l)))

    print('Entity Alignment (right):')

    for i in range(len(top_rl)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))
    print('MRR: %.4f' % (np.mean(mrr_r)))
    print('MR: %.4f' % (np.mean(mr_r)))

    time2 = time.localtime(time.time())
    time2 = time.strftime("%Y-%m-%d %H:%M:%S", time2)
    print("结束时间：", time2)
    end = datetime.datetime.now()
    print("用时", end - start)
    return sim, sim_r, ref
