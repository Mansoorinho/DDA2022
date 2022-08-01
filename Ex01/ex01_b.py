
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

stat = MPI.Status()


#size of array
N=10**8
#print(size)
#print(rank)

Batch = round(N/size)

def find_avg(a):
    avg = np.sum(a)/len(a)
    return avg

t1 = MPI.Wtime()
# master process
if rank == 0:

    data = np.random.random(size = (N,1))
    #data2 = np.random.random(size = (N,1))
    
    #data = np.hstack([data1, data1])
    # master process sends data to worker processes by
    # going through the ranks of all worker processes
    for i in range(1, size):
        
        #sending the matrices
        comm.Send(data, dest=i, tag=i)
        #print(f'From Process {rank} to -> process {i}\n Data: \n{subs[i]}')

    res = find_avg(data[(Batch*rank):(Batch*(1+rank)),0])
    
    #receiving and doing final summation
    final_dt = [res]
    for i in range(1,size):
        #shp_ = np.empty(1, dtype=int)
        #comm.Recv(shp_, source=i, tag=i+size*2)
         
        data_f = np.empty((1,))
        comm.Recv(data_f, source=i, tag=i+size*3)
        final_dt.append(data_f)
    
    #final_res = np.concatenate(final_dt, axis=0)
    print("\naverage of all processes: ",sum(final_dt)/len(final_dt))
    print("\n Average in rank 0 (complete data): ",find_avg(data))

    print("Time in rank 0 is:", MPI.Wtime()-t1)


# worker processes

else:

    data_ = np.empty((N,1))
    comm.Recv(data_, source=0, tag=rank)
    
    #for i in range(100):
    res = find_avg(data_[(Batch*rank):(Batch*(1+rank)),0])
    #data_r = sum_vec(data_[:,0], data_[:,1])
    
    #shp = np.array(res.shape[0], dtype=int)
    #comm.Send(shp, dest=0, tag=size*2+rank)
    comm.Send(res, dest=0, tag=rank+size*3)
    
    #print("Time in rank {} is:".format(rank), MPI.Wtime()-t1_)
    #print('Process {} received data:'.format(rank), data_.shape)
    print(f'\nResult sent \n From Process {rank} to -> process {0} -> average is: {res}')
