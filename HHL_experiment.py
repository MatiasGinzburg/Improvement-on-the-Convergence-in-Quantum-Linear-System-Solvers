# import Qiskit
from qiskit import Aer, execute, transpile
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.extensions import HamiltonianGate, UnitaryGate
from qiskit.circuit.library import RYGate
from qiskit.quantum_info import Statevector, Operator, partial_trace, state_fidelity


import numpy as np
pi = np.pi
from scipy.stats import unitary_group

import matplotlib.pyplot as plt
from IPython.display import display
import time
import os
import datetime

import pickle


#########################################

##  Parameters of the experiment
t = 8*pi/5

number_of_samples = 50

nc_range = [i for i in range(3,12)]

##########################################



def qc_HHL(A_h,b_h,nc,t):
    
    #Parameters
    n_h = int(np.log2(len(b_h))) # number of qubits in the main register
    d = 2**n_h   # 
    T = 2**nc
    # Base circuit 
    br = QuantumRegister(n_h, name="b") # Main register
    cr = QuantumRegister(nc, name="c") # Clock register
    ar = QuantumRegister(1, name="a") # Ancillary register
    
    qc = QuantumCircuit(br,cr,ar,name = "HHL")
    
    # Initialize b

    ### Example of the gate when dim of b is 2
    #Init_b = np.eye(2)
    #Init_b[0,0] = b_h[0]
    #Init_b[0,1] = -b_h[1]
    #Init_b[1,0] = b_h[1]
    #Init_b[1,1] = b_h[0]
    #Init = UnitaryGate(Init_b,label='Init: b') 
    #qc.append(Init,br)
    ###
    qc.initialize(b_h,br)

    qc.barrier()
    
    # Quantum phase estimation

    # # set the clock
    qc.h(cr)

    # # Conditional hamiltonian evolution
    for ctrl in range(nc):
        # crtl is the qubit on the c register that is going to control the unitary evolution
        eA = HamiltonianGate(A_h,-(2**ctrl)*t,label ="exp(i2^"+str(ctrl)+"At)")
        # HamiltonianGate(H,t) implements e^{-itH}
        ctrl_eA = eA.control(1)
        # Add 1 control qubit to the Unitary evolution
        qbits = [cr[ctrl]] + [br[i] for i in range(n_h)]
        qc.append(ctrl_eA,qbits)
    
    qc.barrier()

    # # Inverse QFT
    
    for tar in range(nc//2):
        qc.swap(cr[tar],cr[nc-tar-1])
        
    
    for tar in range(nc):
        qc.h(cr[tar])
        for ctrl in range(tar+1,nc):
            qc.cp( -pi/(2**(ctrl-tar)), cr[ctrl], cr[tar] )
    
    qc.barrier()
    
    # Conditional rotation
    for k in range(1,2**nc):
        theta = 2*np.arcsin(1/k)
        cry = RYGate(theta).control(nc,ctrl_state=k)
        qbits = [cr[i] for i in range(nc)] + [ar[0]]
        qc.append(cry, qbits)
        
    qc.barrier()
    
    # Inverse phase estimation
   
    # # QFT 
    for tar in range(nc-1,-1,-1):
        for ctrl in range(tar+1,nc):
            qc.cp( pi/(2**(ctrl-tar)), cr[ctrl], cr[tar] )
        qc.h(cr[tar])
        
    for tar in range(nc//2):
        qc.swap(cr[tar],cr[nc-tar-1])

    # # inverse conditional unitaries
    
    for ctrl in range(nc-1,-1,-1):
        eA = HamiltonianGate(A_h,(2**ctrl)*t,label ="exp(-i2^"+str(ctrl)+"At)")
        # HamiltonianGate(H,t) implements e^{-itH}
        ctrl_eA = eA.control(1)
        qbits = [cr[ctrl]] + [br[i] for i in range(n_h)]
        qc.append(ctrl_eA,qbits)
    
    # # inverse clock seting
    
    qc.h(cr)
    return qc


FullStart = time.time()

dim = 2



#I build a list of combinations of matrices A ( Hermitean with eigenvlues in (0,1) ) and vectors b (normalized)


As = []
bs = []
eigenvlues_list = []

for _ in range(number_of_samples):

    eigenvalues = np.random.rand(dim)
    eigenvlues_list.append(eigenvalues)

    A_diag = np.diag(eigenvalues)
    U = unitary_group.rvs(dim)
    As.append( (U.conj().T).dot(A_diag).dot(U) )    

    b = np.random.rand(dim)
    b = b/np.linalg.norm(b)
    bs.append( b )




statevectors = []
ncs = []
times = []

print('t = ', t)
for nc in nc_range:
    print('nc = ',nc)
    for i  in range(number_of_samples):
        A = As[i]
        b = bs[i]
        # print('A = ', A)
        # print('b = ', b)
        print('test Ab number ',i, ' of ',number_of_samples)
        
        initial=''.join(['0' for _ in range(1+nc+1)])
        sv_initial = Statevector.from_label(initial)
        
        qc = qc_HHL(A,b,nc,t)
        
        start = time.time()
        sv = sv_initial.evolve(qc)
        time_elapsed = time.time()-start
        
        statevectors.append(sv)
        ncs.append(nc)
        times.append(time_elapsed)
        print('time elapsed = ', time_elapsed)



P_a1s = []
sv_a1s = []
Fd_a1s = []
norm2_fid_a1s = [] 

P_a1_c0s = []
sv_a1_c0s = []
Fd_a1_c0s = []
norm2_fid_a1_c0s = []




for i in range(len(ncs)):
    nc = ncs[i]
    A = As[ i%number_of_samples ]
    b = bs[ i%number_of_samples ]

    # compare the results with the exact solution
    x_exact = np.linalg.inv(A).dot(b)
    x_exact_norm = np.linalg.norm(x_exact)
    x_exact_normalized = x_exact/x_exact_norm

    # Build the ideal final state from the exact solution
    rsv_a1_c0 = Statevector([0,1])
    for _ in range(nc):
        rsv_a1_c0 = rsv_a1_c0.tensor(Statevector([1,0]))
    #rsv_a1_c0 Reduced state with a=1, c=0
    x_exact_sv = rsv_a1_c0.tensor(Statevector(x_exact_normalized)) # x state written in 1+nc+1 qubits

    # Projection on subspace a=1
    Proj_a1 = Operator(np.kron(np.array([[0,0],[0,1]],dtype=int),np.eye(2**(1+nc),dtype=int)))
    sv_a1 = statevectors[i].evolve(Proj_a1)
    
    P_a1 = np.linalg.norm(sv_a1)**2 # probability of meassure a=1
    P_a1s.append(P_a1)

    norm2_a1 = (t*(2**nc)/(2*pi)) **2 *P_a1
    norm2_fid_a1 = norm2_a1/ (x_exact_norm**2)
    norm2_fid_a1s.append(norm2_fid_a1)
    
    sv_a1 = sv_a1 / np.sqrt(P_a1) # This state is equivalent to the post selection on a=1 
    sv_a1s.append(sv_a1)

    Fd_a1 = state_fidelity(x_exact_sv,sv_a1)
    Fd_a1s.append(Fd_a1)


    Pc0=np.zeros((2**nc,2**nc),dtype=int)
    Pc0[0,0]=1 # projector c=0
    
    Proj_a1_c0 = Operator(np.kron(np.array([[0,0],[0,1]],dtype=int),np.kron(Pc0,np.eye(2,dtype=int)) ))

    sv_a1_c0 = statevectors[i].evolve(Proj_a1_c0)
    
    P_a1_c0 = np.linalg.norm(sv_a1_c0)**2 # probability of meassure a=1
    P_a1_c0s.append(P_a1_c0)

    norm2_a1_c0 = (t*(2**nc)/(2*pi)) **2 *P_a1_c0
    norm2_fid_a1_c0 = norm2_a1_c0 / (x_exact_norm**2)
    norm2_fid_a1_c0s.append(norm2_fid_a1_c0)

    sv_a1_c0 = sv_a1_c0 / np.sqrt(P_a1_c0) # This state is equivalent to the post selection on a=1 
    sv_a1_c0s.append(sv_a1_c0)
    
    Fd_a1_c0 = state_fidelity(x_exact_sv,sv_a1_c0)
    Fd_a1_c0s.append(Fd_a1_c0)



#Saving results

# Define the base folder name
base_folder_name = "data"

# Get the current date and time
current_datetime = datetime.datetime.now()

# Format the current date and time as a string
date_time_string = current_datetime.strftime("%m-%d_%H-%M")

# Combine the base folder name with the formatted date and time
folder = f"{'data_experiment//'}{base_folder_name}_{date_time_string}"

# Create the folder if it doesn't exist
os.makedirs(folder, exist_ok=True)

print("Data will be saved in folder:", folder)

folder_pos = f"{folder}{'//'}"
# Raw statevectors

# Specify the file path where you want to save the statevectors
file_path = 'statevectors.pkl'
# Save the statevectors to a file
with open(folder_pos+file_path, 'wb') as file:
    pickle.dump(statevectors, file)

print("Statevectors saved successfully to:", file_path)

# Raw sv_a1s

# Specify the file path where you want to save the sv_a1s
file_path = 'sv_a1s.pkl'
# Save the sv_a1s to a file
with open(folder_pos+file_path, 'wb') as file:
    pickle.dump(sv_a1s, file)

print("sv_a1s saved successfully to:", file_path)


# Raw sv_a1_c0s

# Specify the file path where you want to save the sv_a1_c0s
file_path = 'sv_a1_c0s.pkl'
# Save the sv_a1_c0s to a file
with open(folder_pos+file_path, 'wb') as file:
    pickle.dump(sv_a1_c0s, file)

print("sv_a1_c0s saved successfully to:", file_path)


# Raw P_a1s

# Specify the file path where you want to save the P_a1s
file_path = 'P_a1s.pkl'
# Save the P_a1s to a file
with open(folder_pos+file_path, 'wb') as file:
    pickle.dump(P_a1s, file)

print("P_a1s saved successfully to:", file_path)

# Raw P_a1_c0s

# Specify the file path where you want to save the P_a1_c0s
file_path = 'P_a1_c0s.pkl'
# Save the P_a1_c0s to a file
with open(folder_pos+file_path, 'wb') as file:
    pickle.dump(P_a1_c0s, file)

print("P_a1_c0s saved successfully to:", file_path)

# Raw Fd_a1s

# Specify the file path where you want to save the Fd_a1s
file_path = 'Fd_a1s.pkl'
# Save the Fd_a1s to a file
with open(folder_pos+file_path, 'wb') as file:
    pickle.dump(Fd_a1s, file)

print("Fd_a1s saved successfully to:", file_path)

# Raw Fd_a1_c0s

# Specify the file path where you want to save the Fd_a1_c0s
file_path = 'Fd_a1_c0s.pkl'
# Save the Fd_a1_c0s to a file
with open(folder_pos+file_path, 'wb') as file:
    pickle.dump(Fd_a1_c0s, file)

print("Fd_a1_c0s saved successfully to:", file_path)


# Raw norm2_fid_a1s

# Specify the file path where you want to save the norm2_fid_a1s
file_path = 'norm2_fid_a1s.pkl'
# Save the norm2_fid_a1s to a file
with open(folder_pos+file_path, 'wb') as file:
    pickle.dump(norm2_fid_a1s, file)

print("norm2_fid_a1s saved successfully to:", file_path)


# Raw norm2_fid_a1_c0s

# Specify the file path where you want to save the norm2_fid_a1_c0s
file_path = 'norm2_fid_a1_c0s.pkl'
# Save the norm2_fid_a1_c0s to a file
with open(folder_pos+file_path, 'wb') as file:
    pickle.dump(norm2_fid_a1_c0s, file)

print("norm2_fid_a1_c0s saved successfully to:", file_path)


# Raw ncs
# Specify the file path where you want to save the ncs
file_path = 'ncs.pkl'
# Save the norm2_fid_a1_c0s to a file
with open(folder_pos+file_path, 'wb') as file:
    pickle.dump(ncs, file)

print("ncs saved successfully to:", file_path)


# Parameters and metadata

# Specify the file path where you want to save the MatricesAndVectors
file_path = 'MatricesAndVectors.pkl'
# Save the Parameters to a file
with open(folder_pos+file_path, 'wb') as file:
    pickle.dump(As, file)
    pickle.dump(bs,file)
    pickle.dump(eigenvlues_list,file)

print("Parameters saved successfully to:", file_path)

# Specify the file path where you want to save the statevectors
file_path = 'Parameters.pkl'
# Save the Parameters to a file
with open(folder_pos+file_path, 'wb') as file:
    pickle.dump(t,file)
    pickle.dump(number_of_samples,file)
    pickle.dump(nc_range,file)

print("Parameters saved successfully to:", file_path)


FullFinish = time.time()


print('Total time = ',FullFinish-FullStart)

print('-----------------\n\n')
print('Folder : ', folder+'//')

# plt.plot(ncs,Fd_a1_c0s, 'D-',color='blue', label=r'$a=1,c=0$')
# plt.plot(ncs,Fd_a1s,'v-', color='green', label = r'$a=1$')
# plt.plot(ncs,np.ones_like(ncs),'black')
# plt.legend()
# plt.xlabel(r'$n_c$')
# plt.ylabel(r'$\langle x | out \rangle^2$')

# plt.show()


# plt.plot(ncs,norm2_fid_a1_c0s,'D-',color='blue', label=r'$a=1,c=0$')
# plt.plot(ncs,norm2_fid_a1s,'v-', color='green', label = r'$a=1$')
# plt.plot(ncs,np.ones_like(ncs),'black')
# plt.legend()
# plt.xlabel(r'$n_c$')
# plt.ylabel(r'$||\tilde x||^2 / ||x||^2$')

# plt.show()