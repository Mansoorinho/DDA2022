
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
name = MPI.Get_processor_name()

#using saved data
with open ('saved/final_list_1.ob','rb') as fp:
    data = pickle.load(fp)
    total_docs = []
    for i in range(len(data)-1):
        for j in range(len(data[i][0])):
            total_docs.append(data[i][0][j])


def word_frequency_in_doc(data):
    word_folder = dict()
    for i in range(len(data)):
        # Get unique words per document
        words = np.unique(data[i])
        # Counting the times a word has been mentioned in a document
        for word in words:
            if word in word_folder:
                word_folder[word] += 1
            else:
                word_folder[word] = 1
    batch = {k: (len(data) / v) for k, v in word_folder.items()}
    return batch


p_ranks = round(len(total_docs)/(size-1))


total = dict()

#master process
if rank == 0:
    t0 = MPI.Wtime()
    total_docs2 = total_docs[0:p_ranks]
    for i in range(1, size):
        total_docs1 = total_docs[(i*p_ranks):(p_ranks*(i+1))]
        comm.send(total_docs1, dest=i)
    output1 = word_frequency_in_doc(total_docs2)

    global_dict = None
    for i in range(1, size):
        idf = comm.recv()
        output1 = (Counter(output1) + Counter(idf))
    total = {k: np.log(v / (size)) for k, v in output1.items()}
   
    print('Time:',MPI.Wtime() - t0)
    with open("/home/mansoor/Desktop/DDA/ex02/saved"+f'/idf_res_{size}.ob', 'wb') as fp:
        pickle.dump(total, fp)
        
        

else:
    data = comm.recv()
    output = word_frequency_in_doc(data)
    comm.send(output, dest=0)


        


# FINALIZATION
