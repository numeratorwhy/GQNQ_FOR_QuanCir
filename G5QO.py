import os
import torch
import numpy as np

# Set number of qubits to 5
num_qubits = 5

# Create the folder to store npy files if it doesn't exist
os.makedirs('5qubit', exist_ok=True)

def gen_all_observables(num_qubits=5):
    # Generate index combinations for Pauli matrices
    v_idx = [torch.tensor([0, 1, 2])] * num_qubits
    v_idx = torch.cartesian_prod(*v_idx)
    
    # Define Pauli matrices
    PAULI = torch.tensor([[[1, 0], [0, -1]], [[0, 1], [1, 0]], [[0, 1j], [-1j, 0]]])
    basis = PAULI[v_idx.flatten()].view([3 ** num_qubits, num_qubits, 2, 2])
    basis_list = basis.unbind(dim=-3)
    
    # Generate einsum expression
    exp = ','.join('...'+chr(i+ord('a')) + chr(i+ord('A')) for i in range(num_qubits))
    v = torch.einsum(exp, *basis_list).reshape([3 ** num_qubits, 4 ** num_qubits])
    
    # Concatenate real and imaginary parts along the last dimension
    v = torch.concat([v.real, v.imag], dim=-1)
    
    # Save each observable as an individual .npy file
    for i, observable in enumerate(v):
        file_path = f'5qubit/float_observable5_{i}.npy'
        np.save(file_path, observable.numpy())

gen_all_observables()
