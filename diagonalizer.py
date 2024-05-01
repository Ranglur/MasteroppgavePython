
import numpy as np
import itertools
from sympy.physics.quantum.cg import CG
from scipy.special import comb
from functools import lru_cache
from bisect import bisect_left
import concurrent.futures
import matplotlib.pyplot as plt

# Simple basis functions:
# ------------------------------------------------------------------------

def generate_simple_basis(n: int) -> np.ndarray:
    """
    Generate a simple basis for a system of n spins.

    The function creates a 2D numpy array representing all possible states of n spins.
    Each row in the returned array represents a different basis state, and each column
    represents a different spin in that state. A True value represents spin up (|1⟩),
    and a False value represents spin down (|0⟩).

    Parameters:
    n (int): The number of spins in the system.

    Returns:
    np.ndarray: A 2D numpy array with each row representing a different basis state.

    Example:
    >>> generate_simple_basis(2)
    array([[False, False],
           [False,  True],
           [ True, False],
           [ True,  True]])
    """

    # Generate all possible combinations of spin up and spin down for n spins
    simple_basis = list(itertools.product([0, 1], repeat=n))

    # Convert the list of tuples to a 2D numpy array of booleans
    simple_basis = np.array(simple_basis, dtype=bool)

    return simple_basis

def sort_basis(simple_basis: np.ndarray) -> np.ndarray:
    """
    Sorts the simple_basis in-place by total magnetization in the z direction and computes the size of each block.

    Parameters:
    simple_basis (np.ndarray): The simple basis to be sorted in-place.

    Returns:
    np.ndarray: An array where the ith element denotes the size of the block corresponding to the ith unique magnetization value.
    """
    # Calculate the total magnetization in the z direction for each state in the basis
    total_magnetization = np.sum(2 * simple_basis - 1, axis=1)

    # Sort the indices of the basis states based on their total magnetization
    sorted_indices = np.argsort(total_magnetization, kind="mergesort")

    # Sort the basis in-place using the sorted indices
    simple_basis[:] = simple_basis[sorted_indices]

    return simple_basis




# Binary search of the simple, Sz-sorted basis
#----------------------------------------------------------------------------------

def bool_array_to_int(state: np.ndarray) -> int:
    """Converts a boolean array to an int

    Args:
        state (np.ndarray): boolean array representing a spin state

    Returns:
        int: integer representation of the state
    """
    binary_repr = ''.join(map(str, state.astype(int)))
    return int(binary_repr, 2)

def prepare_sorted_basis(sorted_basis: np.ndarray) -> np.ndarray:
    """Converts every state in a sorted spin basis (acording to some quantum number) to their corresponding integer

    Args:
        sorted_basis (np.ndarray): An Sz sorted spin basis (array of boolean arrays)

    Returns:
        np.ndarray: Array of ints
    """
    
    return np.array([bool_array_to_int(state) for state in sorted_basis], dtype=int)

def binary_search_in_block(sorted_basis_int: list, Sz: float, target_state: np.ndarray):
    """
    Perform binary search for a target state within a specific Sz block of the sorted basis.

    This function searches for the target state in the specified Sz block of the sorted basis 
    using a binary search algorithm. The sorted basis is provided as a list of integers, and 
    the target state is represented as a boolean array. 

    Parameters:
    - sorted_basis_int (list): The sorted basis represented as a list of integers, where each integer 
                               represents a state in binary format.
    - Sz (float): The Sz value specifying the block within which to search for the target state.
    - target_state (np.ndarray): The target state represented as a boolean array, with True denoting up 
                                 spins and False denoting down spins.

    Returns:
    - int or None: The position of the target state within the specified Sz block of the sorted basis if found, 
                   otherwise returns None.

    Example:
    >>> sorted_basis = [0, 1, 2, 3, ...]
    >>> target = np.array([True, False, True])
    >>> position = binary_search_in_block(sorted_basis, 0.5, target)
    """
    
    N = len(target_state)   # Number of spins in the target state

    block_start = M(N, Sz)
    block_size = get_c_array_dim(N, Sz)

    # # Convert Sorted Basis Block to binary
    # binary_block = [format(num, '0' + str(N+1) + 'b') for num in sorted_basis_int[block_start:block_start + block_size]]

    target_int = bool_array_to_int(target_state)
    position = bisect_left(sorted_basis_int[block_start:block_start + block_size], target_int)

    
    if position == block_size:
        return None
    
    if sorted_basis_int[block_start + position] != target_int:
        
        return None  # target_state is not present in the block
    else:
        return position



# Helper functions:
# -------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def M(N: int, Sz: float) -> int:
    """Finds the starting index of the block with given Sz in a simple, Sz-sorted basis

    Args:
        N (int): Number of spins
        Sz (float): total Sz of the state

    Returns:
        int: Start of the block in simple basis with given Sz
    """
    
    
    total = 0
    # Loop over possible Sz_prime values with the correct step size.
    for Sz_prime in np.arange(-N/2, Sz, 1):
        # Ensure Sz_prime is within the valid range.
        total += get_c_array_dim(N, Sz_prime)
    return total

@lru_cache(maxsize=None)
def get_c_array_dim(N: int, Sz: float) -> int:
    """Calculates the number of states in the simple basis needed to represent a certain state, which is every state in the simple
    with the same Sz as the input

    Args:
        N (int): Number of spins
        Sz (float): Total spin in Sz direction

    Returns:
        int: number of states with same Sz as the input in the simple basis
    """
    
    # Calculate the number of up spins required to achieve total Sz
    k = int(N / 2 + Sz)

    # Check if k is a valid number of up spins
    if k < 0 or k > N:
        return 0

    # Calculate the binomial coefficient
    return int(comb(N, k))

def get_i_dim(n: int, S: float) -> int:
    """Calculates the number of states with the same S and Sz 

    Args:
        n (int): number of spins
        S (float): Total spin

    Returns:
        int: number of states with common S, Sz
    """
    if n <= 0:
        return 0
    elif n == 1 and S == 0.5:
        return 1

    # Initialize a dictionary to store N_s for different n and s
    N_s = {(1, 0.5): 1}

    # Iteratively compute N_s for larger n
    for spins in range(2, n + 1):
        # Track whether the current number of spins is even or odd
        is_even_spins = spins % 2 == 0

        # Calculate the possible values of s for the current number of spins
        possible_s_values = [s/2 for s in range(spins + 1)] if is_even_spins else [s/2 + 0.5 for s in range(spins)]

        # Initialize the new_N_s dictionary for the current number of spins
        new_N_s = {}

        for s in possible_s_values:
            new_N_s[(spins, s)] = N_s.get((spins - 1, s - 0.5), 0) + N_s.get((spins - 1, s + 0.5), 0)

        # Update N_s to include the new_N_s values
        N_s.update(new_N_s)

    # Return N_s(n, S)
    return N_s.get((n, S), 0)


def get_paths(N: int) -> np.ndarray:
    """
    Generate valid paths for a given N.

    A path describes the sequence of raising (+0.5) and lowering (-0.5) operators 
    applied to reach a particular value of S starting from S=0.5. The path can be 
    represented as a binary array where True indicates raising and False indicates lowering.

    Parameters:
    - N (int): The total number of spins.

    Returns:
    - np.ndarray: A sorted array of valid paths.
    """
    
    # Total possible paths for N spins is 2^(N-1)
    total_paths = 2**(N-1)
    paths = []

    # Iterate over all possible combinations of paths
    for i in range(total_paths):
        # Extract binary representation of path
        path = [(i >> bit) & 1 for bit in range(N-1)]
        
        # Initialize S from the starting point
        S = 0.5
        valid = True

        # Traverse the path to check its validity
        for step in path:
            S += 0.5 if step else -0.5

            # If S goes negative at any step, mark path as invalid
            if S < 0:
                valid = False
                break
        
        # If path is valid and results in an allowable S, store it
        if valid and 0 <= S <= N/2:
            paths.append(path)
    
    
    # Convert the list of paths into a numpy array
    paths = np.array(paths, dtype=bool)

    # Order paths by their binary representation (integer value)
    int_order = np.argsort(np.dot(paths, 2**np.arange(N-1)[::-1]),kind="mergesort")  # Note the [::-1] here
    paths = paths[int_order]
    
    
    # Order paths by their sum (value of S)
    s_order = np.argsort(np.sum(paths, axis=1),kind="mergesort")
    paths = paths[s_order]

    
    return paths
    

# TODO: fix initialize function such that all i-s are initialized simulatiously. At the moment this runs N-times instead of just once

def initialize_spin_system(N: int) -> dict:
    """Initializes dict wich will contain the coefficients of a N-spin system. Every coefficient array is initialized as zero array.
    Also puts every valid path into the dict 

    Args:
        N (int): Number of spins

    Returns:
        dict: Empty S,Sz-basis with correct structure
    """
    
    spin_system = {}

    # Determine the possible values of S
    S_values = np.arange(N % 2 / 2, N / 2 + 0.5, 1)
    
    # Get a sorted list of all the possible paths to each S
    paths = get_paths(N)
    path_start = 0
    
    # Iterate through the block sizes to populate the spin_system dictionary
    for S in S_values:
        spin_system[S] = {}
        
        Sz_values = np.arange(-S, S + 1, 1)

        path_dim = get_i_dim(N, S)
        spin_system[S]["paths"] = paths[path_start:(path_start + path_dim)]

        path_start += path_dim
        for Sz in Sz_values:
            spin_system[S][Sz] = {}
            i_dim = get_i_dim(N, S)
            c_array_dim = get_c_array_dim(N, Sz)

            for i in range(i_dim):
                spin_system[S][Sz][i] = np.zeros(c_array_dim)

    
    
    return spin_system

def print_spin_system(spin_system: dict):
    """Prints a S,Sz basis

    Args:
        spin_system (dict): spinsystem in S,Sz basis
    """
    for S, Sz_dict in spin_system.items():
        for Sz, value in Sz_dict.items():
            if Sz != "paths":  # Ignore the "paths" key
                for i, c_array in value.items():
                    print(f"|S={S}, Sz={Sz}; i={i}>", c_array)

def initialize_one_spin_system():
    # Initialize the spin system with N=1
    one_spin_system = initialize_spin_system(1)

    # For a single spin system, there are two states: |1/2, 1/2> and |1/2, -1/2>
    # Set the coefficient arrays for these states to [1]
    one_spin_system[0.5][0.5][0] = np.array([1])
    one_spin_system[0.5][-0.5][0] = np.array([1])

    return one_spin_system

def add_spin_to_system(spin_system):
    """
    Add a spin to a given spin system and compute the resulting system's states.
    
    Parameters:
    spin_system (dict): A nested dictionary representing the current spin system,
                        structured as {S: {Sz: {i: [coefficients]}}}.
                        
    Returns:
    new_spin_system (dict): A nested dictionary representing the spin system after adding a spin,
                            structured as {S: {Sz: {i: [coefficients]}}}.
    
    The function takes a spin system represented by a nested dictionary. It computes the fock states
    for the current system size (N-1) and the enlarged system size (N). Then, it iterates over the coupled states
    of the current system and calculates the contributions of these states to the enlarged system using Clebsch-Gordan coefficients.
    The resulting new_spin_system represents the states of the system after a spin has been added.
    """
    
    # Determining the system size based on the maximum total spin S in the current system
    max_S = max(spin_system.keys())
    N = int(np.round(2 * max_S)) + 1

    # Generating and sorting the Fock states for the current (N-1) and enlarged (N) spin system
    smaller_simple_basis = generate_simple_basis(N-1)
    smaller_simple_basis = sort_basis(smaller_simple_basis)
    larger_simple_basis = generate_simple_basis(N)
    larger_simple_basis = sort_basis(larger_simple_basis)
    larger_simple_basis_int_representation = prepare_sorted_basis(larger_simple_basis)


    # Initializing the data structure for the output and helper variables
    new_spin_system = initialize_spin_system(N)
    helper_state = np.zeros(N, dtype=bool)
    helper_path = np.zeros(N-1, dtype=bool)

    # Iterating over coupled states in the current spin system
    for S, Sz_dict in spin_system.items():
        # Looping over Sz values excluding "paths"
        for Sz in [key for key in Sz_dict.keys() if key != "paths"]:
            for i, coefficients in Sz_dict[Sz].items():
                for new_S in [S - 0.5, S + 0.5]:  # Possible new total spin values
                    if new_S < 0:
                        continue
                    
                    # If it's the first spin being added, then set new_i to 0
                    if N == 2:
                        new_i = 0
                    else:
                        # Update the helper_path based on the previous path and the last step
                        helper_path[:-1] = spin_system[S]["paths"][i]
                        helper_path[-1] = (S < new_S)
                        
                        # Using the provided function to convert the paths to integers
                        helper_path_int = bool_array_to_int(helper_path)
                        sorted_paths_int = prepare_sorted_basis(new_spin_system[new_S]["paths"])

                        # Now using bisect_left on the integer representations
                        new_i = bisect_left(sorted_paths_int, helper_path_int)
                        if new_i == len(sorted_paths_int) or sorted_paths_int[new_i] != helper_path_int:
                            # This should never happen because all possible paths should be included in new_spin_system
                            continue
                        

                    for delta_Sz in [-0.5, 0.5]:  # Possible new Sz values
                        new_Sz = Sz + delta_Sz
                        if new_Sz > new_S:
                            continue
                        if new_Sz < -new_S:
                            continue
                        
                        # Calculating Clebsch-Gordan coefficients for the coupling
                        cg_coeff = CG(S, Sz, 0.5, new_Sz - Sz, new_S, new_Sz).doit().evalf()
                             
                        if cg_coeff == 0:
                            print(f"cg_coeff = {cg_coeff}, S = {S}, Sz = {Sz}, added spin = {new_Sz - Sz}, new_S = {new_S}, new_Sz = {new_Sz}")
                            continue
                        

                        for old_c_array_index, coeff in enumerate(coefficients):
                            
                            
                            # Constructing the state in the larger system based on the state in the smaller system and the added spin
                            helper_state[:-1] = smaller_simple_basis[M(N-1,Sz) + old_c_array_index]
                            helper_state[-1] = (new_Sz > Sz)
                            
                            # Finding the index of the constructed state in the sorted basis of the larger system
                            new_c_array_index = binary_search_in_block(larger_simple_basis_int_representation, new_Sz, helper_state)
                            
                            
                            if new_c_array_index == None:
                                raise ValueError("Generated state not found")

                            # Calculating the contribution to the state in the new system and updating it
                            contribution = float(cg_coeff) * coeff
                            new_spin_system[new_S][new_Sz][new_i][new_c_array_index] += contribution
            
    return new_spin_system

def get_spin_system(N: int) -> dict:
    """Uses add_spin_to_system to add one and one spin until system has N spins

    Args:
        N (int): Desired number of spins

    Returns:
        dict: A S,Sz basis
    """
    
    system = initialize_one_spin_system()
    
    for n in range(N-1):
        system = add_spin_to_system(system)
    
    
    return system


# Operators:
def apply_ladder_operator(plusminus: bool, i: int, input_state: np.ndarray, simple_basis: np.ndarray) -> int:
    """
    Apply the ladder operator S+_i or S-_i on a given input state.

    If the ith spin in the input state is opposite to the ladder operator, it acts as a NOT operator,
    flipping the spin; otherwise, it returns None, representing that the result is 0.

    Parameters:
    plusminus (bool): True denotes S+ operator, and False denotes S- operator.
    i (int): The index of the spin to be flipped (0-indexed).
    input_state (np.ndarray): The input state represented as a boolean array.
    simple_basis (np.ndarray): The simple basis used to find the index of the resulting state.

    Returns:
    int: The index of the resulting state in the simple basis if the operation is valid, else None.

    Example:
    >>> input_state = np.array([True, False, True], dtype=bool)
    >>> simple_basis = generate_simple_basis(3)
    >>> apply_ladder_operator(True, 1, input_state, simple_basis)
    7
    """
    # Copy the input state to avoid modifying the original state
    resulting_state = input_state.copy()

    # Check whether applying the ladder operator is valid
    if plusminus == resulting_state[i]:
        return None

    # Flip the ith spin
    resulting_state[i] = not resulting_state[i]

    # Find the index of the resulting state in the simple basis
    for idx, state in enumerate(simple_basis):
        if np.array_equal(state, resulting_state):
            return idx

    # Raise an exception if the resulting state is not found in the simple basis
    raise ValueError("Resulting state not found in the simple basis.")

def apply_double_ladder_operator(i: int, j: int, Sz: float, input_state: np.ndarray, sorted_simple_basis_int: list) -> int:
    """
    Apply the combined ladder operators S+_i S-_j and S-_i S+_j simultaneously on a given input state.

    Parameters:
    i (int): The index of the first spin to be flipped (0-indexed).
    j (int): The index of the second spin to be flipped (0-indexed).
    Sz (float): The Sz of the input state determines the block which is binary searched
    input_state (np.ndarray): The input state represented as a boolean array.
    simple_basis (np.ndarray): The simple basis used to find the index of the resulting state.

    Returns:
    int: The index of the resulting state in the simple basis if the operation is valid, else None.
    """
    
    if input_state[i] == input_state[j]:
        
        return None

    # Copy the input state to avoid modifying the original state
    resulting_state = input_state.copy()

    # Flip both the i-th and j-th spins
    resulting_state[i] = not resulting_state[i]
    resulting_state[j] = not resulting_state[j]

    
    # Use binary search to find the index of the resulting state in the simple basis
    index = binary_search_in_block(sorted_simple_basis_int, Sz, resulting_state)
    
    return index

def apply_Sz_operator(i: int, input_state: np.ndarray) -> float:
    """returns 1/2 or -1/2 depending on if the i-th spin in the input_state is spin up or spin down

    Args:
        i (int): index of the spin the operator acts on
        input_state (np.ndarray): a spin state 

    Returns:
        float: 1/2 or -1/2
    """
    return (1/2)*input_state[i] - (1/2)*(not input_state[i])


# Hammiltonian matrix functions:
def initialize_hammiltonian_matrix(N: int) -> dict:
    """Initializes the data structrure of the hammiltonian matrix as a nested dict containing hammiltonian matrices for each subspace
    given by S and Sz. Matrices are initialized as zero matrices

    Args:
        N (int): Number of spins

    Returns:
        dict: nested dict containing hammiltonian matrices
    """
    # Determine the possible values of S
    S_values = np.arange(N % 2 / 2, N / 2 + 0.5, 1)
    
    # Set up hammiltonian dict
    H = {}
    
    # Iterate through the block sizes to populate the spin_system dictionary
    for S in S_values:
        H[S] = {}
        
        Sz_values = np.arange(-S, S + 1, 1)

        for Sz in Sz_values:
            i_dim = get_i_dim(N, S)
            H[S][Sz] = np.zeros(shape = (i_dim, i_dim))
            
    return H


# def generate_hammiltonian_matrix(N: int, S_Sz_basis, H_operator):
#     """
#     Generate the Hamiltonian matrix representation in the given S, Sz basis.

#     This function constructs the Hamiltonian matrix for a quantum spin system using 
#     the given S, Sz basis. The Hamiltonian operator is provided as an argument and 
#     is applied to each state to compute matrix elements. The matrix is constructed 
#     in the S, Sz basis, making it block-diagonal.

#     Parameters:
#     - N (int): The number of spins in the system.
#     - S_Sz_basis (dict): A dictionary where keys are total spin S values and values 
#                           are dictionaries with keys being Sz values and values being 
#                           the associated states.
#     - H_operator (function): A function representing the Hamiltonian operator. This 
#                              function takes in a state and returns the state after 
#                              the action of the Hamiltonian.

#     Returns:
#     - dict: A block-diagonal matrix representation of the Hamiltonian in the S, Sz basis. 
#             The keys are total spin S values, and the values are dictionaries with keys 
#             being Sz values and values being the associated matrix blocks.

#     Example:
#     >>> def H_op(state):
#     ...     # Dummy Hamiltonian operator example
#     ...     return state
#     >>> basis = {0.5: {0.5: {...}, -0.5: {...}}, 1.5: {...}}
#     >>> H_matrix = generate_hammiltonian_matrix(2, basis, H_op)
#     """
    
#     # Initialize the hammiltonian matrix and basis
#     H = initialize_hammiltonian_matrix(N)
#     simple_basis = generate_simple_basis(N)
#     simple_basis = sort_basis(simple_basis)
#     simple_basis_int = prepare_sorted_basis(simple_basis)
#     print("generation started")
    
#     # Itterate over the states
#     for S, Sz_dict in S_Sz_basis.items():
#         print(f"generating matrices for S = {S}")
#         for Sz, value in Sz_dict.items():
#             if Sz != "paths":  # Ignore the "paths" key
#                 for j, state_j in value.items():
                    
#                     # calculate H|SSz:j>
#                     transformed_state = H_operator(state_j, S, Sz, N, simple_basis, simple_basis_int)
#                     for i, state_i in value.items():
#                         # calculate H_{ij}^{S,Sz} = <SSz;i|H|SSz;j>
#                         H[S][Sz][i][j] = np.dot(state_i, transformed_state)
                    
#     return H

def compute_H_for_Sz(N, S, Sz, value, H_operator, simple_basis, simple_basis_int, H):
    """
    Compute the Hamiltonian matrix block for a specific S and Sz.

    Parameters:
    - N, S, Sz: Quantum numbers for the system.
    - value: The states associated with this S, Sz.
    - H_operator: The Hamiltonian operator function.
    - simple_basis, simple_basis_int: Basis information.
    - H: The Hamiltonian matrix being populated.

    Returns:
    - Tuple (S, Sz, matrix_block): The S, Sz values and the computed matrix block.
    """
    matrix_block = np.zeros_like(H[S][Sz])
    for j, state_j in value.items():
        transformed_state = H_operator(state_j, S, Sz, N, simple_basis, simple_basis_int)
        for i, state_i in value.items():
            matrix_block[i][j] = np.dot(state_i, transformed_state)
    return S, Sz, matrix_block

def generate_hammiltonian_matrix(N, S_Sz_basis, H_operator):
    # Initialize the hammiltonian matrix and basis
    H = initialize_hammiltonian_matrix(N)
    simple_basis = generate_simple_basis(N)
    simple_basis = sort_basis(simple_basis)
    simple_basis_int = prepare_sorted_basis(simple_basis)
    print("Generation started")

    # ThreadPoolExecutor for parallel computation
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for S, Sz_dict in S_Sz_basis.items():
            print(f"Generating matrices for S = {S}")
            for Sz, value in Sz_dict.items():
                if Sz != "paths":  # Ignore the "paths" key
                    # Submit a job to the thread pool for each Sz
                    futures.append(executor.submit(compute_H_for_Sz, N, S, Sz, value, H_operator, simple_basis, simple_basis_int, H))

        # Retrieve results and populate H
        for future in futures:
            S, Sz, matrix_block = future.result()
            H[S][Sz] = matrix_block

    return H
    
    
def print_hammiltonian_matrix(M: dict):
    """Prints the matrices in a hammiltonian_matrix dictionary

    Args:
        M (dict): nested dict containing the hammiltonian matrices
    """
    for S, Sz_dict in M.items():
        for Sz, matrix in Sz_dict.items():
            print(f"Printing block S = {S}, Sz = {Sz}")
            print(matrix)
            
    
# Hammiltonian operators:

def H_heisenberg_chain(state: np.ndarray, S: float, Sz: float, N: int, simple_basis: np.ndarray, simple_basis_int: np.ndarray, J = 1) -> np.ndarray:
    """Hammiltonian operator for heisenberg N-chain acting on a state in the S, Sz basis and returning the transformed state. 

    Args:
        state (np.ndarray): input state
        S (float): total spin for the state
        Sz (float): total spin in z-direction for the state
        N (int): number of spins
        simple_basis (np.ndarray): the simple basis
        simple_basis_int (np.ndarray): the integer version of the simple basis
        J (int, optional): coupling constant. Defaults to 1.

    Returns:
        np.ndarray: the transformed state
    """
    transformed_state = np.zeros_like(state)
    calculated_M = M(N, Sz)
    for k, c in enumerate(state):
        for i in range(N):
            
            # Sz_iS_zj, modN to enforce periodic boundary:
            transformed_state[k] += c*apply_Sz_operator(i, simple_basis[k + calculated_M])*apply_Sz_operator((i + 1)%N, simple_basis[k + calculated_M])
            
                        
            idx = apply_double_ladder_operator(i, (i+1)%N, Sz, simple_basis[k + calculated_M], simple_basis_int)
            if idx == None:
                continue
            
            transformed_state[idx] += c/2
            

    
    return transformed_state*J



def H_heisenberg_Majumdar_Gosh(state: np.ndarray, S: float, Sz: float, N: int, simple_basis: np.ndarray, simple_basis_int: np.ndarray, J = 1, ratio = 0.5) -> np.ndarray:
    """Mujamdar-Gosh hammiltonian operator acting on a state in the S, Sz basis and returning the transformed state.

    Args:
        state (np.ndarray): input state
        S (float): total spin for the state
        Sz (float): total spin in z-direction for the state
        N (int): number of spins
        simple_basis (np.ndarray): the simple basis
        simple_basis_int (np.ndarray): the integer version of the simple basis
        J (int, optional): coupling constant for NN interaction. Defaults to 1.        
        ratio (float, optional): ratio of coupling constant for second NN interaction wrt NN interaction. Defaults to 0.5.

    Returns:
        np.ndarray: the transformed state
    """
    transformed_state = np.zeros_like(state)
    calculated_M = M(N, Sz)
    for k, c in enumerate(state):
        for i in range(N):
            
            # Sz_iS_zi+1:
            transformed_state[k] += c*apply_Sz_operator(i, simple_basis[k + calculated_M])*apply_Sz_operator((i + 1)%N, simple_basis[k + calculated_M])
            
            # Sz_iS_zi+2:
            transformed_state[k] += ratio*c*apply_Sz_operator(i, simple_basis[k + calculated_M])*apply_Sz_operator((i + 2)%N, simple_basis[k + calculated_M])
                        
            idx1 = apply_double_ladder_operator(i, (i+1)%N, Sz, simple_basis[k + calculated_M], simple_basis_int)
            idx2 = apply_double_ladder_operator(i, (i+2)%N, Sz, simple_basis[k + calculated_M], simple_basis_int)
            
            if idx1 != None:
                transformed_state[idx1] += c/2
            
            if idx2 != None:
                transformed_state[idx2] += ratio*c/2 
            
            

    
    return transformed_state*J




# Diagonalization:

# Helper function used for paralellization
def diagonalize_block(args):
    S, Sz, block_matrix = args
    eigvals, eigvecs = np.linalg.eigh(block_matrix)  # diagonalize the block
    sorted_indices = np.argsort(eigvals)  # get sorted indices

    # Sort eigenvalues and eigenvectors
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]

    return S, Sz, eigvals, eigvecs


def fulldiag(H: dict, N: int, basis: dict , return_eigenbasis: bool = False):
    """Takes in a hammiltonian dict and returns the spectrum and eigenbasis (if return_eigenbasis is true) for the system. 

    Args:
        H (dict): Dict containing the hammiltonian matrices
        N (int): Number of spins
        return_eigenbasis (bool, optional): Bool which controls if eigenbasis is returned or not. Defaults to False.

    Returns:
        dict: spectrum
        dict: eigenbasis (optional)
    """
    # Initialize result structures
    spectrum = {}
    eigenbasis = initialize_spin_system(N) if return_eigenbasis else None

    # Determine the possible values of S
    S_values = np.arange(N % 2 / 2, N / 2 + 0.5, 1)

    # # Set up eigenbasis dict
    eigenbasis = {}
    
    # Iterate through the block sizes to populate the spin_system dictionary
    for S in S_values:
        eigenbasis[S] = {}
        
        Sz_values = np.arange(-S, S + 1, 1)

        for Sz in Sz_values:
            i_dim = get_i_dim(N, S)
            c_array_dim = get_c_array_dim(N, Sz)
            eigenbasis[S][Sz] = {}
            for i in range(i_dim):
                eigenbasis[S][Sz][i] = np.zeros(c_array_dim)
            
    

    # Prepare list of blocks for diagonalization
    blocks_to_diag = [(S, Sz, H[S][Sz]) for S in H for Sz in H[S]]

    # Using ThreadPool to speed up the diagonalization process
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for S, Sz, eigvals, eigvecs in executor.map(diagonalize_block, blocks_to_diag):
            if S not in spectrum:
                spectrum[S] = {}
            spectrum[S][Sz] = eigvals

            if return_eigenbasis:
                for i, vec in enumerate(eigvecs.T):
                    for j, vec_elem in enumerate(vec):
                    
                        eigenbasis[S][Sz][i] += vec_elem*basis[S][Sz][j]

    return spectrum if not return_eigenbasis else (spectrum, eigenbasis)

def plot_spectrum(spectrum: dict, filename = None):
    """Plots the spectrum of a system in a scatterplot, with total spin on the x-axis and energies in the y-axis

    Args:
        spectrum (dict): Dict containing spectrum data
        filename (_type_, optional): Filename, which if given will be the filename the figure is saved to. Defaults to None.
    """
    
    x_vals = []
    y_vals = []

    for S, energies in spectrum.items():
        for Sz, energy_levels in energies.items():
            for energy in energy_levels:
                x_vals.append(S)
                y_vals.append(energy)

    plt.scatter(x_vals, y_vals, c='blue', marker='o', edgecolors='black')
    plt.xlabel('S value')
    plt.ylabel('Energy [J]')
    plt.title('Spectrum Scatter Plot') 
    plt.grid(True)
    
    if filename != None:
        plt.savefig(filename, format="pdf")
    plt.show()


def get_fullspectrum_sorted(spectrum: dict) -> np.ndarray:
    """Takes in a spectrum dictionary and puts all the entries into a np array, and sorts it

    Args:
        spectrum (dict): spectrum data from diagonalization

    Returns:
        np.ndarray: full, sorted spectrum 
    """
    fullspectrum = []
    for S, energies in spectrum.items():
        for Sz, energy_levels in energies.items():
            for energy in energy_levels:
                fullspectrum.append(energy)
    
    fullspectrum = np.array(fullspectrum)
    return fullspectrum[np.argsort(fullspectrum, kind="mergesort")]

    
# Tests of functions, and other functions to be ran in main
# --------------------------------------------------------------

def test_get_i_dim():
    for N in range(1, 5):  # Test for N = 1, 2, 3
        print(f"N = {N}")
        is_even_N = N % 2 == 0

        # Calculate the possible values of S for the current N
        possible_S_values = [s for s in range((N//2) + 1)] if is_even_N else [s + 0.5 for s in range(N//2 + 1)]

        for S in possible_S_values:
            result = get_i_dim(N, S)
            print(f"S = {S}, get_i_dim({N}, {S}) = {result}")
        print("-" * 30)  # Print separator line between different N values


def test_binary_search():
    # Generate the simple basis for four spins
    n = 4
    simple_basis = generate_simple_basis(n)

    # Sort the basis 
    simple_basis = sort_basis(simple_basis)

    # Convert the sorted basis to integer representation
    sorted_basis_int_representation = prepare_sorted_basis(simple_basis)

    # Define a few sample states to test
    sample_states = [
        np.array([True, True, True, True], dtype=bool),  # |1111⟩
        np.array([True, True, True, False], dtype=bool), # |1110⟩
        np.array([False, True, False, True], dtype=bool), # |0101⟩
    ]

    # Perform binary search for each sample state
    for state in sample_states:
        Sz = 0.5*np.sum(2 * state - 1)

        index = binary_search_in_block(sorted_basis_int_representation, Sz, state)
        print(f"State: {state}, Index in block: {index}")

def normtest(spin_system, tolerance=1e-10):
    """
    Test if the L2 norm of each coefficient array in a given spin system is approximately 1.

    Parameters:
    - spin_system (dict): A nested dictionary representing the spin system, structured as {S: {Sz: {i: [coefficients]}}}.
    - tolerance (float, optional): A small positive value indicating the acceptable difference from 1 for the L2 norm. Default is 1e-10.

    Raises:
    - ValueError: If the L2 norm of any coefficient array deviates from 1 by more than the tolerance.

    Returns:
    - None: Function completes successfully if all norms are within the specified tolerance of 1.
    """
    
    for S, Sz_dict in spin_system.items():
        for Sz, value in Sz_dict.items():
            if Sz != "paths":  # Ignore the "paths" key
                for i, c_array in value.items():
                    
                    norm = np.linalg.norm(c_array)
                    if np.abs(norm - 1) > tolerance:
                        raise ValueError(f"The state |S={S}, Sz={Sz}; i={i}> has an L2 norm of {norm}, which deviates from 1 by more than the tolerance.")
        
    print("All states have an L2 norm within the specified tolerance of 1.")

def one_two_three_spin_test():
    one_spin_system = initialize_one_spin_system()
    normtest(one_spin_system)
    print_spin_system(one_spin_system)
    
    print("\n-----------------------------------")
    two_spin_system = add_spin_to_system(one_spin_system)
    normtest(two_spin_system)
    print_spin_system(two_spin_system)
    
    print("\n-----------------------------------")
    three_spin_system = add_spin_to_system(two_spin_system)
    normtest(three_spin_system)
    print_spin_system(three_spin_system)

def test_operators():
    sb = generate_simple_basis(2)
    sb = sort_basis(sb)
    sb_int = prepare_sorted_basis(sb)
    
    
    print(apply_Sz_operator(0,sb[0]))
    
    print("--------------------------------------------")
    print(f"sb[1] = {sb[1]}")
    idx = apply_double_ladder_operator(0, 1, 0, sb[1], sb_int)
    print(f"idx = {idx}")
    print(f"transformed state = {sb[M(2,0) + idx]}")

def hammiltonian_test(N: int, print_matrices = False):
    basis = get_spin_system(N)
    H = generate_hammiltonian_matrix(N, basis, H_heisenberg_chain)
    if print_matrices:
        print_hammiltonian_matrix(H)

def diagonalization_test(N: int, fname = None):
    basis = get_spin_system(N)
    H = generate_hammiltonian_matrix(N, basis, H_heisenberg_chain)
    spectrum, eigenbasis = fulldiag(H, N, basis, return_eigenbasis=True)    
    plot_spectrum(spectrum, filename = fname)
    print(eigenbasis[0][0][0])
    
    
def Majumdar_Gosh_test(N: int, fname = None):
    basis = get_spin_system(N)
    H = generate_hammiltonian_matrix(N, basis, H_heisenberg_Majumdar_Gosh)
    spectrum, eigenbasis = fulldiag(H, N, basis, return_eigenbasis=True)    
    plot_spectrum(spectrum, filename = fname)

def Mujamdar_Gosh_splitting(N: int, M:int, fname = None):
    
    basis = get_spin_system(N)
    ratios = np.linspace(0,1,num=M)
    splitting = np.zeros(M)
    for i, ratio in enumerate(ratios):
        def H_heisenberg_Helper(state: np.ndarray, S: float, Sz: float, N: int, simple_basis, simple_basis_int, J = 1):
            return H_heisenberg_Majumdar_Gosh(state, S, Sz, N, simple_basis, simple_basis_int, J , ratio = ratio)        

        H = generate_hammiltonian_matrix(N, basis, H_heisenberg_Helper)
        spectrum, eigenbasis = fulldiag(H, N, basis, return_eigenbasis=True)
        fullspectrum = get_fullspectrum_sorted(spectrum)
        splitting[i] = fullspectrum[1] - fullspectrum[0] 
        #splitting[i] = spectrum[0][0][1] - spectrum[0][0][0]
        
    print(splitting)
    plt.plot(ratios, splitting)
    plt.title("Splitting of two lowest ground state energies wrt the ratio J'/J")
    plt.xlabel("ratio")
    plt.ylabel("splitting")
    if fname != None:    
        plt.savefig(fname)
    plt.show()


def Mujamdar_Gosh_Symmetry_investigation(N: int, fname = None):
    basis = get_spin_system(N)
    J_primes = [0.5, 0.52]

    for J_prime in J_primes:
        def H_heisenberg(state: np.ndarray, S: float, Sz: float, N: int, simple_basis: np.ndarray, simple_basis_int: np.ndarray, J = 1):
            return H_heisenberg_Majumdar_Gosh(state, S, Sz, N, simple_basis, simple_basis_int, J = 1, ratio = J_prime)
        
        H = generate_hammiltonian_matrix(N, basis, H_heisenberg)
        spectrum, eigenbasis = fulldiag(H, N, basis, return_eigenbasis=True)
        fullspectrum = get_fullspectrum_sorted(spectrum)
        
        count = 1
        groupings = []
        


        for i in range(1, len(fullspectrum)):
            if not np.isclose(fullspectrum[i-1],fullspectrum[i]):
                groupings.append(count)
                count = 1
                continue

            count += 1

        print(groupings)





def MJ_splitting(N: int, n: int, pointsize: float , fname: str):
    # Sets up the |S,Sz,i> spinbasis
    basis = get_spin_system(N)
    ratios = np.linspace(0.5,0.6,num=n)

    # Dictionary to store spectra for each S and full spectrum
    spectra_for_S = {}
    full_spectrum_data = []

    # Loops through the different values for the second nearest neigbour interaction strength
    for i, ratio in enumerate(ratios):
        # Sets up the helper function
        def H_heisenberg_Helper(state: np.ndarray, S: float, Sz: float, N: int, simple_basis, simple_basis_int, J = 1):
            return H_heisenberg_Majumdar_Gosh(state, S, Sz, N, simple_basis, simple_basis_int, J , ratio = ratio)        

        # Generates the block diagonal hammiltonian matrix using the hammiltonian operator above
        H = generate_hammiltonian_matrix(N, basis, H_heisenberg_Helper)
        #Diagonalizes the spectrum
        spectrum, eigenbasis = fulldiag(H, N, basis, return_eigenbasis=True)
        

        # Store spectra for each S
        for S in spectrum.keys():
            if S not in spectra_for_S:
                spectra_for_S[S] = []
            spectra_for_each_S = [spectrum[S][Sz][i] for Sz in spectrum[S] for i in range(len(spectrum[S][Sz]))]
            spectra_for_S[S].extend(spectra_for_each_S)

        # Store the full spectrum
        full_spectrum = get_fullspectrum_sorted(spectrum)  # Assuming this returns a flat list of all eigenvalues
        full_spectrum_data.extend(full_spectrum)

    
    # Calculate M as the number of unique S values
    l = len(spectra_for_S)

    # Plotting
    fig, axs = plt.subplots(l + 1, 1, figsize=(10, 10 * (l + 1)))

    # Plot for each S value
    for i, S in enumerate(spectra_for_S.keys()):
        num_eigenvalues = len(spectra_for_S[S])
        expanded_ratios = np.repeat(ratios, num_eigenvalues // n)
        axs[i].scatter(expanded_ratios, spectra_for_S[S], s=pointsize)
        axs[i].set_title(f"Spectrum for S = {S}")
        axs[i].set_xlabel("Perturbation Ratio")
        axs[i].set_ylabel("Eigenenergy")

    # Plot for the full spectrum
    expanded_full_ratios = np.repeat(ratios, len(full_spectrum_data) // n)
    axs[-1].scatter(expanded_full_ratios, full_spectrum_data, s=pointsize)
    axs[-1].set_title("Full Spectrum")
    axs[-1].set_xlabel("Perturbation Ratio")
    axs[-1].set_ylabel("Eigenenergy")

    plt.tight_layout()
    plt.savefig(fname)
    plt.show()
   


def Eigenbasis(N: int):
    basis = get_spin_system(N)
    H = generate_hammiltonian_matrix(N, basis, H_heisenberg_Majumdar_Gosh)
    spectrum, eigenbasis = fulldiag(H, N, basis, return_eigenbasis=True)
    
    
    
    
    
# main
def main():

    basis = generate_simple_basis(4)
    sort_basis(basis)
    print(prepare_sorted_basis(basis))
    #Initialize and print the spin system for demonstration purposes
    #one_two_three_spin_test()
    
    #Create a larger spinbasis as a test
    #spin_system = get_spin_system(10)
    #normtest(spin_system)

    #test_operators()
    
    #hammiltonian_test(10, print_matrices=False)
    #diagonalization_test(8, "Heisenberg_N8_spectrum.pdf")
    #Majumdar_Gosh_test(8, "Majumdar_Gosh_N8_spectrum.pdf")
    #Mujamdar_Gosh_splitting(8, 41, fname="Second_nn_splitting.pdf")
    #Mujamdar_Gosh_Symmetry_investigation(8)
    #Eigenbasis(8)

    #MJ_splitting(8,120,4,"spectrum_splitting.pdf")


    #Diagonalizing the system
    # N = 8
    # simple_basis = generate_simple_basis(N)
    # simple_basis = sort_basis(simple_basis)
    # simple_basis_int = prepare_sorted_basis(simple_basis)
    # simple_state = np.array([False,True,False,True,False,True,False,True],dtype=bool)
    # idx = binary_search_in_block(simple_basis_int, 0, simple_state)
    # print(idx)


    return 0


# Run main if file not imported
if __name__ == "__main__":
    main()
    
