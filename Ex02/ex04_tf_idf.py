
#loading libraries
from mpi4py import MPI
import numpy as np 
import pandas as pd
import pickle
import os
from collections import defaultdict, Counter

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()



#using saved data
with open ('saved/final_list_1.ob','rb') as fp:
    data = pickle.load(fp)
    total_docs = []
    for i in range(len(data)-1):
        for j in range(len(data[i][0])):
            total_docs.append(data[i][0][j])



def calculate_tfidf(data):
    chunk = []
    word_folder = dict()
    for i in range(len(data)):
        word = defaultdict(int)
        unique_tf = np.unique(data[i])
        counter = len(np.unique(unique_tf))
        for i in data[i]:
            word[i] +=1
        dictionary = {k: v / counter for k, v in word.items()}
        chunk.append(dictionary)
        for unique in unique_tf:
            if unique in word_folder:
                word_folder[unique] += 1
            else:
                word_folder[unique] = 1
    batch = {k: (len(data) / v) for k, v in word_folder.items()}

    return chunk, batch




p_ranks = round(len(total_docs)/(size-1))
final_idf = dict()
final = []
dictcalculate_tfidf = dict()
#master process
if rank == 0:
    t0 = MPI.Wtime()
    total_docs2 = total_docs[0:p_ranks]
    for i in range(1, size):
        total_docs1 = total_docs[(i*p_ranks):(p_ranks*(i+1))]
        comm.send(total_docs1, dest=i)
    TF1, IDF1 = calculate_tfidf(total_docs2)

    global_dict = None
    for i in range(1, size):
        chunka = []
        TF2, IDF2 = comm.recv()
        TF1 = TF1 + TF2
        IDF1 = (Counter(IDF2) + Counter(IDF1))
    final = TF1 + final
    final_idf = {k: np.log(v / (size-1)) for k, v in IDF1.items()}
    for i in range(len(final)):
        n = final[i]
        for key, value in n.items():
            if key in final_idf:
                dictcalculate_tfidf[key] = (n[key]) * (final_idf[key])
            else:
                None 
                
    print('Time:',MPI.Wtime() - t0)

else:
    data = comm.recv()
    TF, IDF = calculate_tfidf(data)
    output = (TF, IDF)
    comm.send(output, dest=0)


with open("/home/mansoor/Desktop/DDA/ex02/saved"+f'/tfidf_res_{size}.ob', 'wb') as fp:
        pickle.dump(dictcalculate_tfidf, fp)
