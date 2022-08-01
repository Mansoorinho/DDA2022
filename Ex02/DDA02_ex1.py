
#importing libraries
import os
import numpy as np
import re
import pickle
#import nltk 
from mpi4py import MPI


#loading manual stopwords
from stop_words_man import stopw
stopword_manually = stopw() #created stopwords manually


#initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def clean_tokenized(file_n):
    """
    This function receives a list of file names and goes through each file and removes\
    unnecessary words and symbols, it also checks every single word not to be of stopwords\
    at the end tokenize them by tputting them in a list.
    args:
        - file_n: file names
    output:
        - tokenized words (list)
    """
    #empty lists will be used for final addtion and also filtering words.
    tokenized_words = []
    filtered_line = []
    for file in (file_n):
        with open(file, 'r', encoding = 'Latin-1') as f:
            text = f.read()
            #not word replaced by space
        text = [re.sub("[^a-zA-Z]", " ", f) for f in text.split()]
        text = " ".join(text)#to remove white spaces
        text = text.lower()
        text = [f for f in text.split() if len(f)>1] #single words deleted
        filtered_line = [w for w in text if not w in stopword_manually]
        tokenized_words += [filtered_line] #tokenization
        
    #with open("/home/mansoor/Desktop/DDA/ex02/saved"+f'/tokenized_rank_{rank}.ob', 'wb') as fp:
    #    pickle.dump(tokenized_words, fp)
    
    return tokenized_words
            
            
            
    
def main_function():
    """
    This function is the main function to run the parallel program on.
    """
    #time
    t0 = MPI.Wtime()
    
    path = "/home/mansoor/Desktop/DDA/ex02/20_newsgroups/"
    #folders in the path
    folders = os.listdir(path)
    #change the current working directory to path
    os.chdir(path)

    final_list = []
    #going through each folders and files inside of them.
    for i in range(len(folders)):
        line_inner = []
        filenames = []
        inner_path = (path+str(folders[i]))
        os.chdir(inner_path)
        filenames += [f for f in os.listdir(inner_path) if os.path.isfile(os.path.join(inner_path,f))]
        #determining the share for each process
        share = round(len(filenames)/size)
    

        #only slaves
        if rank != 0:

            #receiving names of the files
            filens = comm.recv(source=0, tag = 0)
            #assigning files to ditexterent workers except worker 0

            #cleaning and tokenization
            tokenized = clean_tokenized(filens)


            #sending to master node
            comm.send(tokenized, dest=0, tag=1)
            
            
        #master process
        else:
            
            #distributing the data
            for i in range(1, size):

                comm.send(filenames[(share*i):(share*(1+i))], dest = i, tag=0)

            #master node's share of file
            filenames = filenames[(share*rank):(share*(1+rank))]
            #cleaning and tokenization 
            tokenized = clean_tokenized(filenames)
            #saving the file      
            #with open("/home/mansoor/Desktop/DDA/ex02/saved"+f'/tokenized_rank_{rank}.ob', 'wb') as fp:
            #    pickle.dump(tokenized, fp)
            #appending
            line_inner.append(tokenized)

            #receiving results
            for i in range(1, size):
            
                tknzd = comm.recv(source = i, tag = 1)

                line_inner.append(tknzd)
                


        #appending final result
        final_list.append(line_inner)
        
   
    t1 = MPI.Wtime() - t0
    print("Time: ", t1)
    #print(len(final_list))
    #save_csv(final_list, rank=10)
    with open("/home/mansoor/Desktop/DDA/ex02/saved"+f'/final_list_{size}.ob', 'wb') as fp:
        pickle.dump(final_list, fp)
    return final_list
    
main_func = main_function()
