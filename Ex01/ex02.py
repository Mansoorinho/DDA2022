
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

stat = MPI.Status()


#size of array
N=10**4
#print(size)
#print(rank)

Batch = round(N/size)

def multiplication(a,b):
  z = np.dot(a,b)
  return z

t1 = MPI.Wtime()
# master process
if rank == 0:

    data1 = np.random.random(size = (N,N))
    data2 = np.random.random(size = (N,N))
    #print(np.dot(data1[0,:],data2[:,0]))
    #vec_c = np.empty(N,1)
    
    #data = np.hstack([data1, data1])
    # master process sends data to worker processes by
    # going through the ranks of all worker processes
    for i in range(1, size):
        
        #sending the matrices
        comm.Send(data1, dest=i, tag=i)
        comm.Send(data2, dest=i, tag=i*2)
        #print(f'From Process {rank} to -> process {i}\n Data: \n{subs[i]}')

        
        
    d1 = data1[(Batch*rank):(Batch*(1+rank)),:]
    d2 = data2[(Batch*rank):(Batch*(1+rank)),:]
    
    #print(d1.shape)
    #print(d2.shape)
    #Calculating product
    vector_s = np.zeros(shape=(d1.shape[0],d1.shape[1]))
    for row in range(0, d1.shape[0]):
        for colm in range(0, d2.shape[0]):
            vector_s[row,0] += (d1[row,colm] * d2[colm,0])
    #Sending back segment of output vector C
    
    print(f"shape in rank {rank}, is: {vector_s.shape}")
    #res = sum_vec(data[(Batch*rank):(Batch*(1+rank)),:5], data[(Batch*rank):(Batch*(1+rank)),0])
    #res = multiplication(data1[(Batch*rank):(Batch*(1+rank))], data2[(Batch*rank):(Batch*(1+rank)), :])

    #receiving and doing final summation
    final_dt = [vector_s]
    for i in range(1,size):
        shp_ = np.empty(2, dtype=int)
        comm.Recv(shp_, source=i, tag=i+size*2)
         
        data_f = np.empty((shp_[0],shp_[1]))
        comm.Recv(data_f, source=i, tag=i+size*3)
        final_dt.append(data_f)
    
    final_res = np.concatenate(final_dt, axis=0)
    #print("final result: ",final_res[0])
    print("finallyyyyy....",final_res.shape)

    print("Time in rank 0 is:", MPI.Wtime()-t1)


# worker processes

else:

    data_1 = np.empty((N,N))
    comm.Recv(data_1, source=0, tag=rank)
    
    data_1 = data_1[(Batch*rank):(Batch*(1+rank))]
    #print(data_1.shape)
    
    data_2 = np.empty((N,N))
    comm.Recv(data_2, source=0, tag=rank*2)
    data_2 = data_2[(Batch*rank):(Batch*(1+rank))]
    #print(data_2.shape)
    
    
    
    #Calculating product
    vector_s = np.zeros(shape=(data_1.shape[0],data_2.shape[1]))
    for row in range(0, data_1.shape[0]):
        for colm in range(0, data_2.shape[0]):
            vector_s[row,0] += (data_1[row,colm] * data_2[colm,0])
    
    
    print(f"shape in rank {rank}, is: {vector_s.shape}")
    #for i in range(100):
    #res = multiplication(data_1, data_2)
    #data_r = sum_vec(data_[:,0], data_[:,1])
    
    shp = np.array(vector_s.shape, dtype=int)
    comm.Send(shp, dest=0, tag=size*2+rank)
    comm.Send(vector_s, dest=0, tag=rank+size*3)
    
    #print("Time in rank {} is:".format(rank), MPI.Wtime()-t1_)
    #print('Process {} received data:'.format(rank), data_.shape)
    #print(f'\nResult sent \n From Process {rank} to -> process {0}\n Data shape: \n{res.shape}')
