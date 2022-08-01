

from mpi4py import MPI
import numpy as np 
import os
import pickle

#using saved data
with open ('saved/final_list_1.ob','rb') as fp:
    data = pickle.load(fp)

#initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#name = MPI.Get_processor_name()

#batch = rank(len(data-1)/size)

#tf calculator function
def calculate_tf(tokenized_data):
    tf_list = []
    for document in tokenized_data:
        sentence_dict = dict()
        for word in document:
            sentence_dict[word] = sentence_dict.get(word,0)+1
        len_docu = len(document)
        
        for word in sentence_dict:
            sentence_dict[word] = sentence_dict[word]/len_docu
        tf_list.append(sentence_dict)
    return tf_list





t0 = MPI.Wtime()
def main_function():
    """
    This function is the main function to run the parallel program on.
    """
    #time
    
    
    final_list = []
    #going through each folders and files inside of them.
    for i in range(len(data)-1):
        line_inner = []

        #determining the share for each process
        share = round(len(data[i][0])/size)
        #the data is selected
        selected = data[i][0]
    

        #only slaves
        if rank != 0:

            #receiving names of the files
            filens = comm.recv(source=0, tag = 0)
            #assigning files to ditexterent workers except worker 0
            
            #calculating tf
            tf_res = calculate_tf(filens)

            #sending to master node
            comm.send(tf_res, dest=0, tag=1)
            
        
        #master process
        else:
            
            #distributing the data
            for i in range(1, size):

                comm.send(selected[(share*i):(share*(1+i))], dest = i, tag=0)

            #master node's share of file
            filens = selected[(share*rank):(share*(1+rank))]
            #tf
            tf_res = calculate_tf(filens)
            line_inner.append(tf_res)

            #receiving results
            for i in range(1, size):
            
                tff = comm.recv(source = i, tag = 1)
                line_inner.append(tff)
                


        #appending final result
        final_list.append(line_inner)
        
   
    t1 = MPI.Wtime() - t0
    print("Time: ", t1)

    with open("/home/mansoor/Desktop/DDA/ex02/saved"+f'/tf_res_{size}.ob', 'wb') as fp:
        pickle.dump(final_list, fp)
    return final_list
    
main_func = main_function()
    
