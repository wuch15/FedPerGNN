import numpy as np

def generate_batch_data_random(batch_size,train_user_index,trainu,traini,history,trainlabel,user_neighbor_emb):
    idx = np.array(list(train_user_index.keys()))
    np.random.shuffle(idx)
    batches = [idx[range(batch_size*i, min(len(idx), batch_size*(i+1)))] for i in range(len(idx)//batch_size+1) if len(range(batch_size*i, min(len(idx), batch_size*(i+1))))]

    while (True):
        for i in batches:
            idxs=[train_user_index[u] for u in i]
            uid=np.array([])
            iid=np.array([])
            uneiemb=user_neighbor_emb[:0]
            y=np.array([])
            for idss in idxs:
                uid=np.concatenate([uid,trainu[idss]])
                iid=np.concatenate([iid,traini[idss]])
                y=np.concatenate([y,trainlabel[idss]])
                uneiemb=np.concatenate([uneiemb,user_neighbor_emb[trainu[idss]]],axis=0)
            uid=np.array(uid,dtype='int32')
            iid=np.array(iid,dtype='int32')
            ui=history[uid]
            uid=np.expand_dims(uid,axis=1)
            iid=np.expand_dims(iid,axis=1)
            
            
            yield ([uid,iid,ui,uneiemb], [y])


def generate_batch_data(batch_size,testu,testi,history,testlabel,user_neighbor_emb):
    idx = np.arange(len(testlabel))
    np.random.shuffle(idx)
    y=testlabel
    batches = [idx[range(batch_size*i, min(len(y), batch_size*(i+1)))] for i in range(len(y)//batch_size+1)]

    while (True):
        for i in batches:
            uid=np.expand_dims(testu[i],axis=1)
            iid=np.expand_dims(testi[i],axis=1)
            ui=history[testu[i]]
            uneiemb=user_neighbor_emb[testu[i]]

            yield ([uid,iid,ui,uneiemb], [y])
