from __future__ import annotations
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import diagonalizer as diag
import UV_group as UV


# Sorting function for spinbasis
def sort_and_count_degeneracy(data, sorting_numbers, number_of_rungs):
    """
    Sorts the input data based on sorting numbers, counts the degeneracy of each sorting number,
    and calculates the starting index for each degeneracy block in the sorted data.
    
    Args:
    - data (np.ndarray): The data to be sorted, typically representing eigenstates or other quantities associated with a toric code Hamiltonian.
    - sorting_numbers (np.ndarray): An array of numbers used to sort the data. These might represent quantum numbers, energy levels, or other relevant quantities.
    - number_of_rungs (int): The number of rungs in the ladder system, which determines the size of the degeneracy array.
    
    Returns:
    - tuple: A tuple containing:
        - sorted_data (np.ndarray): The data sorted according to the sorting numbers.
        - degeneracy (np.ndarray): An array counting the degeneracy of each unique sorting number.
        - block_start (np.ndarray): An array indicating the starting index of each block of degenerate states in the sorted data.
    """    
    # Stable sort the data based on sorting numbers
    sorted_indices = np.argsort(sorting_numbers, kind='stable')
    sorted_data = data[sorted_indices]
    sorted_qn = sorting_numbers[sorted_indices]

    # Count the degeneracy of each sorting number
    degeneracy = np.zeros(2**(2*number_of_rungs), dtype=int)
    for num in sorted_qn:
        degeneracy[int(num)] += 1

    block_start = np.zeros_like(degeneracy,dtype=int)
    for i, elem in enumerate(degeneracy[0:-1]):
        block_start[i+1] = block_start[i] + elem

    return sorted_data, degeneracy, block_start

# Binary search for state in sorted spinbasis
def get_index_for_state_in_block(sorted_basis_int: np.ndarray, As_quantum_number: int, target_state: np.ndarray, block_size, block_start):
    """
    Perform binary search for a target state within a specific As block of the sorted basis.

    This function searches for the target state in the specified As block of the sorted basis 
    using a binary search algorithm. The sorted basis is provided as a array of integers, and 
    the target state is represented as a boolean array. 

    Parameters:
    - sorted_basis_int (list): The sorted basis represented as a list of integers, where each integer 
                               represents a state in binary format.
    - As (float): The As value specifying the block within which to search for the target state.
    - target_state (np.ndarray): The target state represented as a boolean array, with True denoting up 
                                 spins and False denoting down spins.

    Returns:
    - int or None: The position of the target state within the specified As block of the sorted basis if found, 
                   otherwise returns None.

    Example:
    >>> sorted_basis = [0, 1, 2, 3, ...]
    >>> target = np.array([True, False, True])
    >>> position = binary_search_in_block(sorted_basis, 0.5, target)
    """
    
    N = len(target_state)


    target_int = diag.bool_array_to_int(target_state) # type: ignore
    position = diag.bisect_left(sorted_basis_int[block_start:block_start + block_size], target_int) # type: ignore

    
    
    if position == block_size:
        print("not in block")
        return None
        
    if sorted_basis_int[block_start + position] != target_int:
        print("not in block")
        return None  # target_state is not present in the block
    else:
        return position


# Conversion functions
# --------------------------------------------------------------------------
def point_2_index_star(point: np.ndarray, number_of_rungs: int) -> int:
    """
    Convert a point coordinate to a star index.
    
    Args:
    - point (np.ndarray): The point coordinate as an array [x, y].
    - number_of_rungs (int): The number of rungs in the lattice.
    
    Returns:
    - int: The index of the star.
    """
    return (point[0]//2) + (point[1]//2)*number_of_rungs

def index_2_point_star(index: int, number_of_rungs: int) -> np.ndarray:
    """
    Convert a star index to a point coordinate.
    
    Args:
    - index (int): The index of the star.
    - number_of_rungs (int): The number of rungs in the lattice.
    
    Returns:
    - np.ndarray: The point coordinate as an array [x, y].
    """
    y = 2*(index // number_of_rungs)
    x = 2*(index % number_of_rungs)
    return np.array([x, y])

def point_2_index_spin(point: np.ndarray, number_of_rungs: int) -> int:
    """
    Convert a point coordinate to a spin index.
    
    Args:
    - point (np.ndarray): The point coordinate as an array [x, y].
    - number_of_rungs (int): The number of rungs in the lattice.
    
    Returns:
    - int: The index of the spin.
    """
    return point[1]*number_of_rungs + (point[0] - point[0]%2)//2

def index_2_point_spin(index: int, number_of_rungs: int) -> np.ndarray:
    """
    Convert a spin index to a point coordinate.
    
    Args:
    - index (int): The index of the spin.
    - number_of_rungs (int): The number of rungs in the lattice.
    
    Returns:
    - np.ndarray: The point coordinate as an array [x, y].
    """
    y = index // number_of_rungs
    x = (y + 1)%2 + 2*(index % number_of_rungs)
    return np.array([x, y])

def index_2_point_placetes(index: int, number_of_rungs: int) -> np.ndarray:
    """
    Convert a plaquette index to a point coordinate.
    
    Args:
    - index (int): The index of the plaquette.
    - number_of_rungs (int): The number of rungs in the lattice.
    
    Returns:
    - np.ndarray: The point coordinate as an array [x, y].
    """
    y = 1 + (index // number_of_rungs)*2
    x = 1 + (index % number_of_rungs)*2
    return np.array([x, y])

def point_2_index_placetes(point: np.ndarray, number_of_rungs: int) -> int:
    """
    Convert a point coordinate to a plaquette index.
    
    Args:
    - point (np.ndarray): The point coordinate as an array [x, y].
    - number_of_rungs (int): The number of rungs in the lattice.
    
    Returns:
    - int: The index of the plaquette.
    """
    index = (point[0]-1)//2 + ((point[1]-1)//2)*number_of_rungs
    return index


# Periodicity functions: 
#------------------------------------------------------------------------------
def periodicity_model_1(point: np.ndarray, number_of_rungs: int) -> np.ndarray:
    """
    Apply periodic boundary conditions for a ladder version of the toric code.
    
    This function applies periodicity along the x-axis within a lattice of width '2*number_of_rungs'. 
    The y-axis is bounded, allowing only values 0, 1, or 2. Points outside these bounds are mapped to [-1, -1].
    
    Args:
    - point (np.ndarray): The point coordinates as an array [x, y].
    - number_of_rungs (int): The number of rungs, defining the periodicity along the x-axis.
    
    Returns:
    - np.ndarray: The adjusted point coordinates after applying periodicity, or [-1, -1] if outside bounds.
    """
    point[0] = point[0] % (2*number_of_rungs)
    if (point[1] < 0) or (point[1] > 2):
        return np.array([-1, -1])
    return point

def periodicity_model_2(point: np.ndarray, number_of_rungs: int) -> np.ndarray:
    """
    Apply periodic boundary conditions for a 2D version of the toric code.
    
    This function enforces periodicity along both the x and y axes within a lattice. The x-axis is wrapped 
    within '2*number_of_rungs' width, and the y-axis is wrapped within a height of 4, effectively modeling 
    a toroidal topology.
    
    Args:
    - point (np.ndarray): The point coordinates as an array [x, y].
    - number_of_rungs (int): The number of rungs, defining the periodicity along the x-axis.
    
    Returns:
    - np.ndarray: The adjusted point coordinates after applying periodicity on both axes.
    """
    point[0] = (point[0] + 2*number_of_rungs) % (2*number_of_rungs)
    point[1] = (point[1] + 4) % 4
    return point




# Operators:
#--------------------------------------------------------------------------------
def A_s(simple_state, point, periodicity_func, number_of_rungs):
    """
    Apply the A_s operator to a given point in the lattice.
    
    Args:
    - simple_state (np.array): The current state of the system, where each spin is represented as either 0 or 1.
    - point (np.array): The coordinates [x, y] of the star in the lattice.
    - periodicity_func (function): The periodicity function to apply, which handles the boundary conditions.
    - number_of_rungs (int): The number of rungs in the lattice.
    
    Returns:
    - int: The product of the values (-1 or 1) associated with the spins around the star.
    """
    shifts = np.array([[-1,0],[1,0],[0,-1],[0,1]])  # Define shifts to access neighboring spins.
    val: int = 1  # Initialize the product value.
    for shift in shifts:  # Loop through each neighboring position.
        new_point = point + shift  # Calculate the new point's position.
        new_point = periodicity_func(new_point, number_of_rungs)  # Apply periodicity to handle boundaries.
        if np.all(new_point == np.array([-1,-1])):  # Check if the new point is outside the valid range.
            continue  # Skip invalid points.

        idx = point_2_index_spin(new_point, number_of_rungs)  # Convert the point to a spin index.
        val *= 2*simple_state[idx] - 1  # Update the product value based on the spin state.

    return val

def B_p(simple_state, point, periodicity_func, number_of_rungs):
    """
    Apply the B_p operator to a given point in the lattice, flipping the spins around a plaquette.
    
    Args:
    - simple_state (np.array): The current state of the system, where each spin is represented as either 0 or 1.
    - point (np.array): The coordinates [x, y] of the plaquette in the lattice.
    - periodicity_func (function): The periodicity function to apply, which handles the boundary conditions.
    - number_of_rungs (int): The number of rungs in the lattice.
    
    Returns:
    - np.array: The new state of the system after applying the B_p operator.
    """
    new_state = simple_state.copy()  # Copy the current state to modify.
    shifts = np.array([[-1,0],[1,0],[0,-1],[0,1]])  # Define shifts to access neighboring spins.
    for shift in shifts:  # Loop through each neighboring position.
        new_point = point + shift  # Calculate the new point's position.
        new_point = periodicity_func(new_point, number_of_rungs)  # Apply periodicity to handle boundaries.
        if np.all(new_point == np.array([-1,-1])):  # Check if the new point is outside the valid range.
            continue  # Skip invalid points.

        idx = point_2_index_spin(new_point, number_of_rungs)  # Convert the point to a spin index.
        new_state[idx] = not new_state[idx]  # Flip the spin at the calculated index.

    return new_state

def B_p_1_projector(state_as_lincombo: np.ndarray, as_qn: int, number_of_rungs: int, degeneracy: int, block_start: int, sorted_basis: np.ndarray, sorted_basis_int: np.ndarray, periodicity_func):
    """
    Apply the B_p operator as a projector to a state represented as a linear combination of basis states.
    
    This function flips the spins around each plaquette for the entire lattice and updates the state's representation 
    as a linear combination accordingly. It is designed for use with quantum states of the toric code Hamiltonian.
    
    Args:
    - state_as_lincombo (np.ndarray): The initial quantum state as a linear combination of basis states.
    - as_qn (int): Quantum number associated with the state.
    - number_of_rungs (int): The number of rungs in the lattice.
    - degeneracy (int): The degeneracy of the quantum number.
    - block_start (int): The starting index of the block in the sorted basis.
    - sorted_basis (np.ndarray): The sorted basis of quantum states.
    - sorted_basis_int (np.ndarray): The integer representation of the sorted basis.
    - periodicity_func (function): The periodicity function to apply, handling the boundary conditions.
    
    Returns:
    - np.ndarray: The new quantum state as a normalized linear combination of basis states after applying the B_p operator.
    """
    # Copy the initial state to avoid modifying it directly.
    new_state_as_lincombo = np.copy(state_as_lincombo)
    
    # Iterate over all plaquettes.
    for j in range(2*number_of_rungs):     
        point = index_2_point_placetes(j, number_of_rungs)  # Convert index to point coordinate for the plaquette.
        point = periodicity_func(point, number_of_rungs)  # Apply periodicity function to handle boundary conditions.
        if np.all(point == np.array([-1,-1])):  # Skip if the point is invalid after applying periodicity.
            continue
        
        # Initialize a transformed state array with zeros.
        transformed_state_as_lincombo = np.zeros_like(new_state_as_lincombo)
        
        # Iterate over each coefficient in the linear combination.
        for i, c in enumerate(new_state_as_lincombo):
            simple_state = sorted_basis[i + block_start]  # Retrieve the basis state from the sorted basis.
            # Apply B_p operator to the basis state at the current plaquette.
            transformed_simple_state = B_p(simple_state, point, periodicity_func, number_of_rungs)
            # Find the index of the transformed state in the block.
            k = get_index_for_state_in_block(sorted_basis_int, as_qn, transformed_simple_state, degeneracy, block_start)
            
            # Update the coefficients of the transformed state.
            transformed_state_as_lincombo[i] += c
            transformed_state_as_lincombo[k] += c
        
        # Update the new state as the transformed state.
        new_state_as_lincombo = transformed_state_as_lincombo
    
    # Normalize the new state and return it.
    return new_state_as_lincombo/la.norm(new_state_as_lincombo)

def get_qn(state, number_of_rungs, periodicity_func):
    """
    Calculate the quantum number for a given state in a toric code lattice.
    
    The quantum number is calculated by applying the A_s operator to each star in the lattice and 
    encoding the results as bits in an integer. This quantum number serves as a unique identifier for the 
    state's configuration with respect to the A_s operators.
    
    Args:
    - state (np.array): The state of the system, where each element represents the state of a spin.
    - number_of_rungs (int): The number of rungs in the lattice, which determines the size of the lattice.
    - periodicity_func (function): The periodicity function to apply, which handles boundary conditions of the lattice.
    
    Returns:
    - int: The quantum number representing the configuration of the state with respect to the A_s operators.
    """
    qn: int = 0  # Initialize the quantum number as 0.
    # Iterate over each star in the lattice.
    for i in range(0, number_of_rungs*2):
        # Calculate the point coordinate of the star from its index.
        point = index_2_point_star(i, number_of_rungs)
        # Apply the A_s operator to the star, and adjust the result to be either 0 or 1.
        result = (A_s(state, point, periodicity_func, number_of_rungs) + 1) // 2
        # Update the quantum number by encoding the result as a bit in the corresponding position.
        qn += 2**i * result
    return int(qn)  # Return the quantum number as an integer.




# Generation of A_s basis (not eigenbasis as theese are not eigenstates of B_p)
#-------------------------------------------------------------------------------------
def generate_toric_basis_model_1(number_of_rungs: int):
    """
    Generate the basis states for the toric code model 1 with specified boundary conditions.
    
    Args:
    - number_of_rungs (int): The number of rungs in the lattice, which determines the size of the system.
    
    Returns:
    - tuple: Sorted basis states, their degeneracy, and block start indices, all considering the A_s operator quantum numbers.
    """
    # Generate a simple basis with 3 times the number of rungs. This implies a specific lattice configuration.
    simple_basis = diag.generate_simple_basis(3*number_of_rungs)
    # Initialize an array to store the quantum numbers for each basis state.
    quantum_numbers = np.zeros(len(simple_basis), dtype=int)

    # Calculate the quantum number for each state in the simple basis.
    for j, state in enumerate(simple_basis):
        quantum_numbers[j] = get_qn(state, number_of_rungs, periodicity_model_1)

    # Sort the basis states by quantum numbers and count the degeneracy of these quantum numbers.
    return sort_and_count_degeneracy(simple_basis, quantum_numbers, number_of_rungs)


def generate_toric_basis_model_2(number_of_rungs: int):
    """
    Generate the basis states for the toric code model 2 with specified boundary conditions.
    
    Args:
    - number_of_rungs (int): The number of rungs in the lattice, which determines the size of the system.
    
    Returns:
    - tuple: Sorted basis states, their degeneracy, and block start indices, all considering the A_s operator quantum numbers.
    """
    # Generate a simple basis with 4 times the number of rungs. This configuration accommodates different boundary conditions.
    simple_basis = diag.generate_simple_basis(4*number_of_rungs)
    # Initialize an array to store the quantum numbers for each basis state.
    quantum_numbers = np.zeros(len(simple_basis), dtype=int)

    # Calculate the quantum number for each state in the simple basis.
    for j, state in enumerate(simple_basis):
        quantum_numbers[j] = get_qn(state, number_of_rungs, periodicity_model_2)

    # Sort the basis states by quantum numbers and count the degeneracy of these quantum numbers.
    return sort_and_count_degeneracy(simple_basis, quantum_numbers, number_of_rungs)


def generate_toric_ground_states(sorted_basis: np.ndarray, degeneracy: int, bloc_start: int, number_of_rungs: int, periodicity_func):
    """
    Generate ground states of the toric code model by projecting each basis state within a degeneracy block.
    
    Args:
    - sorted_basis (np.ndarray): The sorted basis of quantum states, arranged by their quantum numbers.
    - degeneracy (int): The degeneracy of the block of states with the same quantum number.
    - bloc_start (int): The starting index of the block in the sorted basis.
    - number_of_rungs (int): The number of rungs in the lattice, defining the size of the system.
    - periodicity_func (function): The periodicity function to apply, handling boundary conditions.
    
    Returns:
    - np.ndarray: An array containing unique ground state configurations as determined by the B_p_1_projector.
    """
    # Calculate the quantum number for the last state in the sorted basis to define the block.
    as_qn = get_qn(sorted_basis[-1], number_of_rungs, periodicity_func)
    
    # Initialize an array to hold the results of projecting each basis state.
    results = np.zeros((degeneracy, degeneracy))
    
    # Prepare the integer representation of the sorted basis for processing.
    sorted_basis_int = diag.prepare_sorted_basis(sorted_basis)
    
    # Iterate over each state in the degeneracy block.
    for i in range(0, degeneracy):
        # Initialize a state as a linear combination with a single basis state.
        state_as_lincombo = np.zeros(degeneracy)
        state_as_lincombo[i] = 1

        # Apply the B_p_1_projector to project the state across all plaquettes.
        projected_state_as_lincombo = B_p_1_projector(state_as_lincombo, as_qn, number_of_rungs, degeneracy, bloc_start, sorted_basis, sorted_basis_int, periodicity_func)

        # Store the projected state in the results array.
        results[i] = projected_state_as_lincombo
        print(f"Finished projecting {i}th state")
        
    # Find and return unique quantum states from the results.
    return find_unique_quantum_states(results)

def efficient_generate_toric_ground_states(sorted_basis: np.ndarray, degeneracy: int, bloc_start: int, number_of_rungs: int, model_type=2):
    """
    Efficiently generate the ground state manifold (GSM) of a toric code model using U and V operators.
    
    Args:
    - sorted_basis (np.ndarray): The sorted basis of quantum states.
    - degeneracy (int): The degeneracy of the ground state manifold.
    - bloc_start (int): The starting index of the block in the sorted basis.
    - number_of_rungs (int): The number of rungs in the lattice.
    - model_type (int, optional): The type of the model (1 or 2), which affects the choice of periodicity function and operators.
    
    Returns:
    - np.ndarray: An array representing the ground state manifold after applying symmetry and projection operations.
    """
    # Choose the periodicity function and define the GSM dimensions based on the model type.
    if model_type == 2:
        periodicity_func = periodicity_model_2
        gsm_dim = 4  # Dimension of the GSM for model type 2.
        UV_operator = UV_group_operator  # Use the full set of U and V operators.
        # Define group elements for applying the UV operators.
        gsm_group_elems = np.zeros(shape=(4, 5), dtype=bool)
        gsm_group_elems[1][1] = True
        gsm_group_elems[2][2] = True
        gsm_group_elems[3][1] = True
        gsm_group_elems[3][2] = True
    else:
        periodicity_func = periodicity_model_1
        gsm_dim = 2  # Dimension of the GSM for model type 1.
        UV_operator = small_UV_group_operator  # Use a smaller set of U and V operators.
        # Define group elements for applying the UV operators.
        gsm_group_elems = np.zeros(shape=(2, 3), dtype=bool)
        gsm_group_elems[1][1] = True

    # Prepare the integer representation of the sorted basis.
    sorted_basis_int = diag.prepare_sorted_basis(sorted_basis)
    # Calculate the quantum number for the last state in the sorted basis.
    as_qn = get_qn(sorted_basis[-1], number_of_rungs, periodicity_func)
    # Initialize an array to hold the results of the GSM projection.
    results = np.zeros((gsm_dim, degeneracy))

    # Apply UV operators and B_p projector to generate the GSM.
    for i in range(0, gsm_dim):
        helper_lincombo = np.zeros(degeneracy)
        helper_lincombo[-1] = 1  # Prepare a helper linear combination state.
        # Apply the corresponding UV group operator to the helper state.
        state_as_lincombo = UV_operator(gsm_group_elems[i], helper_lincombo, as_qn, number_of_rungs, degeneracy, bloc_start, sorted_basis, sorted_basis_int)
        # Project the state using the B_p_1 projector to align with the toric code ground states.
        projected_state_as_lincombo = B_p_1_projector(state_as_lincombo, as_qn, number_of_rungs, degeneracy, bloc_start, sorted_basis, sorted_basis_int, periodicity_func)
        # Store the projected state in the results.
        results[i] = projected_state_as_lincombo
        print(f"Finished projecting {i}th state")

    return results

def find_unique_quantum_states(states):
    """
    Identify unique quantum states from a set of states based on their inner products.
    
    Args:
    - states (np.ndarray): An array of quantum states, where each state is represented as a vector.
    
    Returns:
    - np.ndarray: An array of unique quantum states filtered from the input.
    """
    tol = 1e-5  # Tolerance for considering two states parallel (essentially the same).
    unique_indices = []  # To keep track of indices of unique vectors.

    # Iterate over each state to determine if it is unique.
    for i, state_i in enumerate(states):
        is_unique = True  # Assume the state is unique until proven otherwise.
        # Compare with previously identified unique states.
        for j in unique_indices:
            state_j = states[j]
            # Compute the absolute value of the inner product between the two states.
            inner_product = np.abs(np.dot(state_i, state_j))
            # If the inner product is close to 1, the states are parallel (not unique).
            if np.abs(inner_product - 1) < tol:
                is_unique = False  # The state is not unique.
                break  # No need to check against other unique states.
        # If the state is unique, add its index to the list of unique indices.
        if is_unique:
            unique_indices.append(i)

    # Extract the unique vectors using the indices identified above.
    unique_vectors = states[unique_indices]
    return unique_vectors


# Symmetry operators:
#------------------------------------------------------------------------------------------------------------------------
def U_y(state_as_lincombo: np.ndarray, as_qn: int, number_of_rungs: int, degeneracy: int, block_start: int, sorted_basis: np.ndarray, sorted_basis_int: np.ndarray):
    """
    Apply the U_y logical operator to a quantum state represented as a linear combination of basis states.
    
    This operator flips the spins along a loop around the y-axis of the lattice. In the context of the toric code, 
    it corresponds to creating or moving anyonic excitations around the system in a way that is topologically 
    protected, thus not altering the energy of the state.
    
    Args:
    - state_as_lincombo (np.ndarray): The initial quantum state as a linear combination of basis states.
    - as_qn (int): The quantum number associated with the state.
    - number_of_rungs (int): The number of rungs in the lattice, defining the size of the system.
    - degeneracy (int): The degeneracy of the block of states with the same quantum number.
    - block_start (int): The starting index of the block in the sorted basis.
    - sorted_basis (np.ndarray): The sorted basis of quantum states.
    - sorted_basis_int (np.ndarray): The integer representation of the sorted basis.
    
    Returns:
    - np.ndarray: The new state as a linear combination of basis states after applying the U_y operator.
    """
    # Copy the initial state to avoid modifying it directly.
    new_state_as_lincombo = np.copy(state_as_lincombo)
    # Initialize a transformed state array with zeros.
    transformed_state_as_lincombo = np.zeros_like(new_state_as_lincombo)
    
    # Iterate over each coefficient in the linear combination.
    for i, c in enumerate(new_state_as_lincombo):
        # Retrieve the basis state from the sorted basis.
        simple_state = sorted_basis[i + block_start]
        # Copy the simple state to perform transformations.
        transformed_simple_state = np.copy(simple_state)
        
        # Flip the spins along the loop around the y-axis.
        for n in range(number_of_rungs, 4*number_of_rungs, 2*number_of_rungs):
            transformed_simple_state[n] = not transformed_simple_state[n]
        
        # Find the index of the transformed state in the block.
        k = get_index_for_state_in_block(sorted_basis_int, as_qn, transformed_simple_state, degeneracy, block_start)
        # Update the coefficients of the transformed state.
        transformed_state_as_lincombo[k] += c
    
    # Update the new state as the transformed state.
    new_state_as_lincombo = transformed_state_as_lincombo
    
    return new_state_as_lincombo
    
def U_x(state_as_lincombo: np.ndarray, as_qn: int, number_of_rungs: int, degeneracy: int, block_start: int, sorted_basis: np.ndarray, sorted_basis_int: np.ndarray):
    """
    Apply the U_x logical operator to a quantum state represented as a linear combination of basis states.
    
    This operator flips the spins along a loop around the x-axis of the lattice. It is a key operation in the toric code,
    allowing for the manipulation of anyonic excitations in a topologically protected way, thereby not altering the state's energy.
    
    Args:
    - state_as_lincombo (np.ndarray): The initial quantum state as a linear combination of basis states.
    - as_qn (int): The quantum number associated with the state.
    - number_of_rungs (int): The number of rungs in the lattice, defining the size of the system.
    - degeneracy (int): The degeneracy of the block of states with the same quantum number.
    - block_start (int): The starting index of the block in the sorted basis.
    - sorted_basis (np.ndarray): The sorted basis of quantum states.
    - sorted_basis_int (np.ndarray): The integer representation of the sorted basis.
    
    Returns:
    - np.ndarray: The new state as a linear combination of basis states after applying the U_x operator.
    """
    # Copy the initial state to avoid direct modifications.
    new_state_as_lincombo = np.copy(state_as_lincombo)
    # Initialize a transformed state array with zeros.
    transformed_state_as_lincombo = np.zeros_like(new_state_as_lincombo)
    
    # Iterate over each coefficient in the linear combination.
    for i, c in enumerate(new_state_as_lincombo):
        # Retrieve the corresponding basis state from the sorted basis.
        simple_state = sorted_basis[i + block_start]
        # Copy the simple state to apply transformations.
        transformed_simple_state = np.copy(simple_state)
        # Flip the spins along the loop around the x-axis.
        for n in range(0, number_of_rungs):
            transformed_simple_state[n] = not transformed_simple_state[n]
        
        # Find the index of the transformed state in the block.
        k = get_index_for_state_in_block(sorted_basis_int, as_qn, transformed_simple_state, degeneracy, block_start)
        # Update the coefficients of the transformed state.
        transformed_state_as_lincombo[k] += c
    
    # Update the new state as the transformed state.
    new_state_as_lincombo = transformed_state_as_lincombo
    
    return new_state_as_lincombo

def V_x(state_as_lincombo: np.ndarray, as_qn: int, number_of_rungs: int, degeneracy: int, block_start: int, sorted_basis: np.ndarray, sorted_basis_int: np.ndarray):
    """
    Apply the V_x logical operator to a quantum state represented as a linear combination of basis states on a sublattice.
    
    This operator conceptually performs a loop around the x-axis on the sublattice, affecting the state based on the configuration
    of spins it encounters. In the toric code, this is another way of manipulating anyonic excitations in a manner that 
    explores the system's topological aspects without altering the overall energy.
    
    Args:
    - state_as_lincombo (np.ndarray): The initial quantum state as a linear combination of basis states.
    - as_qn (int): The quantum number associated with the state.
    - number_of_rungs (int): The number of rungs in the lattice, defining the size of the system.
    - degeneracy (int): The degeneracy of the block of states with the same quantum number.
    - block_start (int): The starting index of the block in the sorted basis.
    - sorted_basis (np.ndarray): The sorted basis of quantum states.
    - sorted_basis_int (np.ndarray): The integer representation of the sorted basis.
    
    Returns:
    - np.ndarray: The new state as a linear combination of basis states after applying the V_x operator.
    """
    # Copy the initial state to avoid direct modifications.
    new_state_as_lincombo = np.copy(state_as_lincombo)

    # Iterate over the range specific to the sublattice.
    for n in range(number_of_rungs, 2*number_of_rungs):
        # Initialize a transformed state array with zeros.
        transformed_state_as_lincombo = np.zeros_like(new_state_as_lincombo)
        
        # Iterate over each coefficient in the linear combination.
        for i, c in enumerate(new_state_as_lincombo):
            # Retrieve the corresponding basis state from the sorted basis.
            simple_state = sorted_basis[i + block_start]
            # Determine the factor based on the spin at the current position; -1 for spin down, +1 for spin up.
            factor = (int(simple_state[n])*2) - 1  

            # Update the coefficient of the state, adjusting by the factor.
            transformed_state_as_lincombo[i] += c * factor
        
        # Update the new state as the transformed state.
        new_state_as_lincombo = transformed_state_as_lincombo
    
    return new_state_as_lincombo

def V_y(state_as_lincombo: np.ndarray, as_qn: int, number_of_rungs: int, degeneracy: int, block_start: int, sorted_basis: np.ndarray, sorted_basis_int: np.ndarray):
    """
    Apply the V_y logical operator to a quantum state represented as a linear combination of basis states on a sublattice.
    
    This operator performs a conceptual loop around the y-axis on the sublattice, modifying the state based on the 
    configuration of spins it encounters. It's another method within the toric code for exploring topological properties 
    and manipulating anyonic excitations without changing the state's energy.
    
    Args:
    - state_as_lincombo (np.ndarray): The initial quantum state as a linear combination of basis states.
    - as_qn (int): The quantum number associated with the state.
    - number_of_rungs (int): The number of rungs in the lattice, defining the size of the system.
    - degeneracy (int): The degeneracy of the block of states with the same quantum number.
    - block_start (int): The starting index of the block in the sorted basis.
    - sorted_basis (np.ndarray): The sorted basis of quantum states.
    - sorted_basis_int (np.ndarray): The integer representation of the sorted basis.
    
    Returns:
    - np.ndarray: The new state as a linear combination of basis states after applying the V_y operator.
    """
    # Copy the initial state to avoid direct modifications.
    new_state_as_lincombo = np.copy(state_as_lincombo)

    # Iterate over specific positions in the sublattice to enact the V_y operation.
    for n in range(0, 4*number_of_rungs, 2*number_of_rungs):
        # Initialize a transformed state array with zeros.
        transformed_state_as_lincombo = np.zeros_like(new_state_as_lincombo)
        
        # Iterate over each coefficient in the linear combination.
        for i, c in enumerate(new_state_as_lincombo):
            # Retrieve the corresponding basis state from the sorted basis.
            simple_state = sorted_basis[i + block_start]
            # Determine the factor based on the spin at the current position; -1 for spin down, +1 for spin up.
            factor = (int(simple_state[n])*2) - 1  

            # Update the coefficient of the state, adjusting by the factor.
            transformed_state_as_lincombo[i] += c * factor
        
        # Update the new state as the transformed state.
        new_state_as_lincombo = transformed_state_as_lincombo
    
    return new_state_as_lincombo


def UV_group_operator(group_elem: np.ndarray, state_as_lincombo: np.ndarray, as_qn: int, number_of_rungs: int, degeneracy: int, block_start: int, sorted_basis: np.ndarray, sorted_basis_int: np.ndarray):
    """
    Apply a combination of U and V operators to a state based on a given group element for the second model of the toric code.
    
    Args:
    - group_elem (np.ndarray): A boolean array representing the group element specifying which operators to apply.
    - state_as_lincombo (np.ndarray): The initial quantum state as a linear combination of basis states.
    - as_qn, number_of_rungs, degeneracy, block_start: Parameters defining the state and system configuration.
    - sorted_basis (np.ndarray): The sorted basis of quantum states.
    - sorted_basis_int (np.ndarray): The integer representation of the sorted basis.
    
    Returns:
    - np.ndarray: The state after applying the specified combination of operators.
    """
    funcs = [U_x, U_y, V_x, V_y]  # Define the set of operators to be applied.
    new_state_as_lincombo = np.copy(state_as_lincombo)  # Copy the initial state to avoid direct modifications.

    if group_elem[0]:  # If the first element is True, multiply the state by -1 (a phase flip).
        new_state_as_lincombo *= -1
    
    # Apply each operator based on the group element specification.
    for i, func in enumerate(funcs):
        if group_elem[i+1]:  # Check if the operator should be applied.
            new_state_as_lincombo = func(new_state_as_lincombo, as_qn, number_of_rungs, degeneracy, block_start, sorted_basis, sorted_basis_int)

    return new_state_as_lincombo


def small_UV_group_operator(group_elem: np.ndarray, state_as_lincombo: np.ndarray, as_qn: int, number_of_rungs: int, degeneracy: int, block_start: int, sorted_basis: np.ndarray, sorted_basis_int: np.ndarray):
    """
    Apply a combination of U and V operators to a state based on a given group element for the first model of the toric code.
    
    Args:
    - group_elem (np.ndarray): A boolean array representing the group element specifying which operators to apply.
    - state_as_lincombo (np.ndarray): The initial quantum state as a linear combination of basis states.
    - as_qn, number_of_rungs, degeneracy, block_start: Parameters defining the state and system configuration.
    - sorted_basis (np.ndarray): The sorted basis of quantum states.
    - sorted_basis_int (np.ndarray): The integer representation of the sorted basis.
    
    Returns:
    - np.ndarray: The state after applying the specified combination of operators.
    """
    funcs = [U_x, V_y]  # Define the set of operators for this model.
    new_state_as_lincombo = np.copy(state_as_lincombo)  # Copy the initial state.

    if group_elem[0]:  # Apply a phase flip if the first element is True.
        new_state_as_lincombo *= -1
    
    # Apply each operator based on the group element specification.
    for i, func in enumerate(funcs):
        if group_elem[i+1]:  # Check if the operator should be applied.
            new_state_as_lincombo = func(new_state_as_lincombo, as_qn, number_of_rungs, degeneracy, block_start, sorted_basis, sorted_basis_int)

    return new_state_as_lincombo


# Test functions
# ----------------------------------------------
def first_system_test():
    # Dermine the degeneracy for ladder model (result is 2)
    num_rungs = 3
    sb, dgen, block_start = generate_toric_basis_model_1(num_rungs)
    as_qn = get_qn(sb[-1], num_rungs, periodicity_model_1)
    ground_state_manifold = generate_toric_ground_states(sb, dgen[as_qn], block_start[as_qn], num_rungs, periodicity_model_1)
    print(len(ground_state_manifold))

    # Get the characters for the representation
    sorted_basis_int = diag.prepare_sorted_basis(sb) # type: ignore
    UV_cover = UV.generate_group_elements(3)
    conjugacy_classes = UV.generate_conjugacy_classes(UV_cover, UV.small_covering_multiplication)
    representation = np.zeros(shape=(len(conjugacy_classes),len(ground_state_manifold),len(ground_state_manifold)))

    for i, conjugacy_class in enumerate(conjugacy_classes):
        for j, state_j in enumerate(ground_state_manifold):
            for k, state_k in enumerate(ground_state_manifold):          
                group_elem = conjugacy_class[0]
                transformed_state_k = small_UV_group_operator(group_elem,state_k, as_qn, num_rungs, dgen[as_qn], block_start[as_qn], sb, sorted_basis_int)
                representation[i][j][k] = np.dot(state_j, transformed_state_k)

    print(np.trace(representation, axis1=1, axis2=2))    
    
    for matrix in representation:
        print(matrix)

def second_system_test():
    # Generate the toric ground states, verifying ground state has degeneracy four. 
    num_rungs = 3
    sb, dgen, block_start = generate_toric_basis_model_2(num_rungs)
    as_qn = get_qn(sb[-1], num_rungs, periodicity_model_2)
    ground_state_manifold = efficient_generate_toric_ground_states(sb, dgen[as_qn], block_start[as_qn], num_rungs)
    print(len(ground_state_manifold))

    

    # Get the characters for the representation
    sorted_basis_int = diag.prepare_sorted_basis(sb) # type: ignore
    UV_cover = UV.generate_group_elements(5)
    conjugacy_classes = UV.generate_conjugacy_classes(UV_cover, UV.covering_multiplication)
    representation = np.zeros(shape=(len(conjugacy_classes),len(ground_state_manifold),len(ground_state_manifold)))

    for i, conjugacy_class in enumerate(conjugacy_classes):
        group_elem = conjugacy_class[0]
        for j, state_j in enumerate(ground_state_manifold):
            for k, state_k in enumerate(ground_state_manifold):          
                
                transformed_state_k = UV_group_operator(group_elem,state_k, as_qn, num_rungs, dgen[as_qn], block_start[as_qn], sb, sorted_basis_int)
                representation[i][j][k] = np.dot(state_j, transformed_state_k)

    print(np.trace(representation, axis1=1, axis2=2))    
    
    for i, matrix in enumerate(representation):
        print(conjugacy_classes[i])
        print(matrix)



def main():
    first_system_test()
    second_system_test()

    return 0



if __name__ == '__main__':
    main()

