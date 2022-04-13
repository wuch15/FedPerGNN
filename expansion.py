from  encrypt import *
from const import *
import random
import numpy as np

def graph_embedding_expansion(Otraining,usernei,alluserembs):
    #local encryption
    local_ciphertext = []
    for i in tqdm(usernei):
        messages = []
        for j in i:
            if j!= Otraining.shape[1]+2:
                messages.append(base64.b64encode(sign(str(j))).decode('utf-8'))
        local_ciphertext.append(messages)
        
    #local id-ciphertext mapping
    local_mapping_dict = {base64.b64encode(sign(str(j))).decode('utf-8'):j for j in range(Otraining.shape[1]+3)}
    
    #assume the local_ciphertext has been sent to the third-party server

    cipher2userid = {}
    for userid,i in enumerate(local_ciphertext):
        for j in i:
            if j not in cipher2userid:
                cipher2userid[j] = [userid]
            else:
                cipher2userid[j].append(userid)

    #third-party server prepares data
                
    send_data = []
    for userid,i in tqdm(enumerate(local_ciphertext)):
        neighbor_info={}
        for j in i:
            neighbor_id = [alluserembs[uid] for uid in cipher2userid[j]]
            if len(neighbor_id):
                neighbor_info[j] = neighbor_id
        send_data.append(neighbor_info)
        
    #third-party server distributes send_data   
    
    
    #local clients expand graphs
    user_neighbor_emb = []
    for userid,user_items in tqdm(enumerate(usernei)):
        receive_data = send_data[userid]
        decrypted_data = {local_mapping_dict[item_key]:receive_data[item_key] for item_key in receive_data}
        all_neighbor_embs=[]
        for item  in user_items:
            if item in decrypted_data:
                neighbor_embs = decrypted_data[item]
                random.shuffle(neighbor_embs)
                neighbor_embs = neighbor_embs[:NEIGHBOR_LEN] 
                neighbor_embs += [[0.]*HIDDEN]*(NEIGHBOR_LEN-len(neighbor_embs))
            else:
                neighbor_embs = [[0.]*HIDDEN]*NEIGHBOR_LEN
            all_neighbor_embs.append(neighbor_embs)
        all_neighbor_embs = all_neighbor_embs[:HIS_LEN]
        all_neighbor_embs += [[[0.]*HIDDEN]*HIS_LEN]*(HIS_LEN-len(all_neighbor_embs))
        user_neighbor_emb.append(all_neighbor_embs)
    
    user_neighbor_emb = np.array(user_neighbor_emb,dtype='float32')
    return user_neighbor_emb
    
