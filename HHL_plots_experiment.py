import pickle
import numpy as np
import matplotlib.pyplot as plt

######################################################
# Complete here the name of the folder:
# It is the last print on the output of HHL_experiment


folder_pos = ''



#######################################################

 #    Reading files   # 


# Specify the file path where you saved the Parameters
file_path = 'Parameters.pkl'

# Load the Parameters from the file
with open(folder_pos+file_path, 'rb') as file:
    t = pickle.load(file)
    number_of_samples = pickle.load(file)
    nc_range = pickle.load(file)

# Specify the file path where you saved the Parameters
file_path = 'MatricesAndVectors.pkl'

# Load the Parameters from the file
with open(folder_pos+file_path, 'rb') as file:
    As = pickle.load(file)
    bs = pickle.load(file)
    eigenvalues_list = pickle.load(file)

# Now 'Fd_a1s' contains the data loaded from the file
print('t = ',t)
print('Number of samples = ', number_of_samples)

# Specify the file path where you saved the Fd_a1s
file_path = 'Fd_a1s.pkl'

# Load the Fd_a1s from the file
with open(folder_pos+file_path, 'rb') as file:
    Fd_a1s = pickle.load(file)


# Specify the file path where you saved the Fd_a1_c0s
file_path = 'Fd_a1_c0s.pkl'

# Load the Fd_a1_c0s from the file
with open(folder_pos+file_path, 'rb') as file:
    Fd_a1_c0s = pickle.load(file)

# Specify the file path where you saved the ncs
file_path = 'ncs.pkl'

# Load the ncs from the file
with open(folder_pos+file_path, 'rb') as file:
    ncs = pickle.load(file)

# Specify the file path where you saved the norm2_fid_a1s
file_path = 'norm2_fid_a1s.pkl'

# Load the norm2_fid_a1s from the file
with open(folder_pos+file_path, 'rb') as file:
    norm2_fid_a1s = pickle.load(file)

# Specify the file path where you saved the norm2_fid_a1_c0s
file_path = 'norm2_fid_a1_c0s.pkl'

# Load the norm2_fid_a1_c0s from the file
with open(folder_pos+file_path, 'rb') as file:
    norm2_fid_a1_c0s = pickle.load(file)



### Plots

Err_a1_c0s = [1-F for F in Fd_a1_c0s]
Err_a1s = [1-F for F in Fd_a1s]

plt.plot(np.array(ncs)-0.05,Err_a1_c0s, '+',color='blue', label=r'Post-selection $|1\rangle_a|0\rangle_c$')
plt.plot(np.array(ncs)+0.05,Err_a1s,'.', color='green', label = r'Post-selection $|1\rangle_a$')


#Get averages for each eigenvalues
number_of_samples = len(As)  # Number of samples for each nc
length = len(ncs)//number_of_samples # Number of different values of nc

ncs_avge = np.zeros(length)
Err_a1s_avge = np.zeros(length)
Err_a1_c0s_avge = np.zeros(length)

for i in range(number_of_samples):
    ncs_avge += np.array([ ncs[j*number_of_samples+i] for j in range(length)])/number_of_samples
    Err_a1s_avge += np.array([ Err_a1s[j*number_of_samples+i] for j in range(length)])/number_of_samples
    Err_a1_c0s_avge += np.array([ Err_a1_c0s[j*number_of_samples+i] for j in range(length)])/number_of_samples

plt.plot(ncs_avge,Err_a1_c0s_avge,'-',color='blue')
plt.plot(ncs_avge,Err_a1s_avge,'--',color='green')


plt.yticks(ticks=[0,0.25,0.5,0.75,1],fontsize=15)
plt.xticks(fontsize=15)
plt.legend()
plt.xlabel(r'$n_c$',fontsize=18)
plt.title(r'1-$\langle x | \tilde x \rangle^2$',fontsize=18)
plt.yscale("log")

plt.show()


Err_norm2_fid_a1_c0s = [np.abs(1-N) for N in norm2_fid_a1_c0s]
Err_norm2_fid_a1s = [np.abs(1-N) for N in norm2_fid_a1s]

plt.plot(np.array(ncs)-0.05,Err_norm2_fid_a1_c0s, '+',color='blue', label=r'Post-selection $|1\rangle_a|0\rangle_c$')
plt.plot(np.array(ncs)+0.05,Err_norm2_fid_a1s,'.', color='green', label = r'Post-selection $|1\rangle_a$')


#Get averages for each eigenvalues
number_of_samples = len(As)  # Number of samples for each nc
length = len(ncs)//number_of_samples # Number of different values of nc

ncs_avge = np.zeros(length)
Err_norm2_fid_a1s_avge = np.zeros(length)
Err_norm2_fid_a1_c0s_avge = np.zeros(length)

for i in range(number_of_samples):
    ncs_avge += np.array([ ncs[j*number_of_samples+i] for j in range(length)])/number_of_samples
    Err_norm2_fid_a1s_avge += np.array([ Err_norm2_fid_a1s[j*number_of_samples+i] for j in range(length)])/number_of_samples
    Err_norm2_fid_a1_c0s_avge += np.array([ Err_norm2_fid_a1_c0s[j*number_of_samples+i] for j in range(length)])/number_of_samples

plt.plot(ncs_avge,Err_norm2_fid_a1_c0s_avge,'-',color='blue')
plt.plot(ncs_avge,Err_norm2_fid_a1s_avge,'--',color='green')

plt.xlabel(r'$n_c$',fontsize=18)
#plt.ylabel(r'$|1-\frac{||\tilde x||^2}{||x||^2} |$',fontsize=18)
plt.title(r'$\frac{| \;||x||- ||\tilde x|| \; |^2} {||x||^2} $',fontsize=18)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.legend()
plt.yscale("log")

plt.show()


plt.plot(np.array(ncs)-0.05,Err_a1_c0s, '+',color='blue', label=r'Post-selection $|1\rangle_a|0\rangle_c$')
plt.plot(np.array(ncs)+0.05,Err_a1s,'.', color='green', label = r'Post-selection $|1\rangle_a$')

plt.plot(ncs_avge,Err_a1_c0s_avge,'-',color='blue')
plt.plot(ncs_avge,Err_a1s_avge,'--',color='green')


plt.yticks(ticks=[0,0.25,0.5,0.75,1],fontsize=15)
plt.xticks(fontsize=15)
plt.legend()
plt.xlabel(r'$n_c$',fontsize=18)
plt.title(r'1-$\langle x | \tilde x \rangle^2$',fontsize=18)

plt.show()


plt.plot(np.array(ncs)-0.05,Err_norm2_fid_a1_c0s, '+',color='blue', label=r'Post-selection $|1\rangle_a|0\rangle_c$')
plt.plot(np.array(ncs)+0.05,Err_norm2_fid_a1s,'.', color='green', label = r'Post-selection $|1\rangle_a$')

plt.plot(ncs_avge,Err_norm2_fid_a1_c0s_avge,'-',color='blue')
plt.plot(ncs_avge,Err_norm2_fid_a1s_avge,'--',color='green')

plt.xlabel(r'$n_c$',fontsize=18)
# plt.title(r'$|1-\frac{||\tilde x||^2}{||x||^2} |$',fontsize=18)
plt.title(r'$\frac{| \;||x||- ||\tilde x|| \; |^2} {||x||^2} $',fontsize=18)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.legend()
plt.show()