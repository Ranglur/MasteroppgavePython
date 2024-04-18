from __future__ import annotations
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from itertools import product
import numpy as np
import seaborn as sns




def generate_group_elements(N):
    """
    Generate all possible group elements of size N using binary combinations.
    
    Args:
    - N (int): The size of the group elements to generate.
    
    Returns:
    - list: A list of numpy arrays representing the group elements.
    """
    # Generate all binary combinations for the given size N.
    binary_combinations = list(product([False, True], repeat=N))
    
    # Convert each binary combination into a numpy array of type bool.
    group_elements = [np.array(combination, dtype=bool) for combination in binary_combinations]
    
    return group_elements


def simple_multiplication(g, h):
    """
    Perform a simple multiplication (XOR operation) between two group elements.
    
    Args:
    - g (np.array): The first group element.
    - h (np.array): The second group element.
    
    Returns:
    - np.array: The result of the XOR operation between g and h.
    """
    # Perform XOR operation between the two elements.
    return g ^ h

def covering_multiplication(g, h):
    """
    Perform a covering multiplication operation between two group elements,
    including a specific interaction between certain positions.
    
    Args:
    - g (np.array): The first group element.
    - h (np.array): The second group element.
    
    Returns:
    - np.array: The result of the covering multiplication.
    """
    # Start with simple XOR operation.
    result = g ^ h
    
    # Apply specific rule: modify the first bit based on the interaction of certain bits from g and h.
    result[0] ^= (g[4] and h[1]) ^ (g[3] and h[2])
    
    return result

def small_covering_multiplication(g, h):
    """
    Perform a simplified version of the covering multiplication operation between two group elements,
    with an interaction involving fewer positions.
    
    Args:
    - g (np.array): The first group element.
    - h (np.array): The second group element.
    
    Returns:
    - np.array: The result of the simplified covering multiplication.
    """
    # Start with simple XOR operation.
    result = g ^ h
    
    # Apply simplified rule: modify the first bit based on the interaction of certain bits from g and h.
    result[0] ^= g[2] and h[1]
    
    return result




def generate_mul_table(group_elements, mul_func):
    """
    Generate a multiplication table for a group given its elements and a multiplication function.
    
    Args:
    - group_elements (list): A list of the group's elements, represented as numpy arrays.
    - mul_func (function): A function that defines the multiplication operation between two elements of the group.
    
    Returns:
    - np.ndarray: A square numpy array representing the group's multiplication table. Each cell [i, j] contains the result of mul_func(group_elements[i], group_elements[j]).
    """
    # Determine the number of elements in the group.
    num_elements = len(group_elements)
    
    # Initialize an empty numpy array to store the multiplication table.
    mul_table = np.empty((num_elements, num_elements), dtype=object)

    # Populate the multiplication table by applying the multiplication function to each pair of elements.
    for i, a in enumerate(group_elements):
        for j, b in enumerate(group_elements):
            result = mul_func(a, b)  # Perform multiplication operation.
            mul_table[i, j] = result  # Store the result in the table.

    return mul_table

def bool_array_to_int(array, is_signed=False):
    """
    Convert a boolean array into an integer. If the array represents a signed number,
    the first element is treated as the sign bit.
    
    Args:
    - array (np.array): A boolean array representing a binary number.
    - is_signed (bool): Indicates whether the number is signed (True) or unsigned (False).
    
    Returns:
    - int: The integer representation of the input boolean array.
    """
    if is_signed:
        # Treat the first element as the sign bit: 1 for negative, 0 for positive.
        sign_bit = 1 if array[0] else 0
        # Convert the remaining elements to a binary string, then to an integer.
        value = int(''.join(['1' if x else '0' for x in array[1:]]), 2)
        # Apply the sign based on the sign bit.
        return (-1) ** sign_bit * value
    else:
        # Convert the entire array to a binary string, then to an unsigned integer.
        return int(''.join(['1' if x else '0' for x in array]), 2)


def print_mul_table(mul_table, is_signed=False):
    """
    Print the multiplication table of a group in a formatted manner.
    
    Args:
    - mul_table (np.ndarray): A numpy array representing the group's multiplication table.
    - is_signed (bool): Indicates whether the numbers in the multiplication table are signed.
    
    This function formats the multiplication table for easy reading, aligning numbers within columns for signed and unsigned values.
    """
    num_elements = mul_table.shape[0]  # Get the number of elements in the group.
    
    if is_signed:
        # Determine the maximum length of the string representation of the elements for formatting.
        max_len = len(str(num_elements - 1))
        for i in range(num_elements):
            row = []  # Initialize an empty list to store the row elements.
            for j in range(num_elements):
                # Convert each element to an integer, then to a string, and right-justify it within the column.
                row.append(str(bool_array_to_int(mul_table[i, j], is_signed)).rjust(max_len))
            print(" ".join(row))  # Print the formatted row.
    else:
        for i in range(num_elements):
            for j in range(num_elements):
                # For unsigned numbers, simply print the integer conversion of each element.
                print(bool_array_to_int(mul_table[i, j], is_signed))

def calculate_omega(group_elements):
    """
    Calculate the omega matrix for a set of group elements, based on specific conditions.
    
    Args:
    - group_elements (list): A list of numpy arrays representing the group elements.
    
    Returns:
    - np.ndarray: The omega matrix, where each element omega_matrix[i, j] represents the phase (-1 or 1) calculated based on the conditions applied to elements g and h.
    
    The calculation of the omega matrix involves determining the phase for each pair of group elements based on specific bitwise conditions.
    """
    num_elements = len(group_elements)  # Get the number of elements in the group.
    # Initialize the omega matrix with 1s.
    omega_matrix = np.ones((num_elements, num_elements), dtype=int)

    for i, g in enumerate(group_elements):
        for j, h in enumerate(group_elements):
            # Calculate the phase based on specific conditions involving bits of g and h.
            omega = -1 if (g[3] and h[0]) ^ (g[2] and h[1]) else 1
            omega_matrix[i, j] = omega  # Set the phase in the omega matrix.

    return omega_matrix


def plot_omega_matrix(omega_matrix, figname = None):
    """
    Plot the omega matrix using a heatmap to visualize the phase relationships between group elements.
    
    Args:
    - omega_matrix (np.ndarray): The omega matrix to be plotted, where each element indicates the phase (-1 or 1) associated with a pair of group elements.
    
    This function creates a heatmap visualization of the omega matrix, providing a visual representation of the phase factors (-1 or 1) across group element interactions. The heatmap uses the 'coolwarm' color map to distinguish between the two phases, with annotations for clarity.
    """
    sns.set()
    plt.figure(figsize=(8, 6))
    sns.heatmap(omega_matrix, cmap="coolwarm", annot=True, fmt="d", linewidths=.5, square=True, cbar=False)
    plt.xlabel("Group Elements (g)")
    plt.ylabel("Group Elements (h)")
    plt.title("Omega Matrix")
    if figname != None:
        plt.savefig(figname)
    plt.show()


def plot_mul_table(mul_table, figname = None):
    """
    Visualize the multiplication table of a group with boolean array elements.
    
    The function plots each element of the multiplication table, adjusting the color based on the sign bit of each element. Elements with a sign bit of 1 are plotted in blue, while those with a sign bit of 0 are in red. The numerical values (ignoring the sign bit) are displayed in the corresponding cells.
    
    Args:
    - mul_table (np.ndarray): A numpy array representing the group's multiplication table, where each element is a boolean array.
    
    This visualization helps in understanding the structure and operation of the group, especially highlighting the distribution of positive and negative elements if interpreted in a signed context.
    """
    
    num_elements = mul_table.shape[0]
    sns.set()
    fig, ax = plt.subplots(figsize=(10, 8))  # Adjust the figure size as needed
    ax.imshow(np.zeros((num_elements, num_elements)), cmap="coolwarm", vmin=-1, vmax=1, extent=[0, num_elements, num_elements, 0])

    for i in range(num_elements):
        for j in range(num_elements):
            value = mul_table[i, j][1:]
            sign_bit = mul_table[i, j][0]
            label = bool_array_to_int(value, is_signed=False)
            color = 'blue' if sign_bit == 1 else 'red'  # Adjust color mapping
            ax.text(j + 0.5, num_elements - i - 0.5, str(label), ha='center', va='center', color=color, fontsize=8)  # Adjust fontsize

    ax.set_xticks(np.arange(0, num_elements, 1))
    ax.set_yticks(np.arange(0, num_elements, 1))
    ax.set_yticklabels(list(map(lambda x: str(num_elements-1-x), range(num_elements))), fontsize=8)  # Reverse y-axis labels
    ax.invert_yaxis()  # Invert the y-axis
    ax.grid(color='black', linewidth=0.5)
    plt.xlabel("Group Elements (g)")
    plt.ylabel("Group Elements (h)")
    plt.title("Multiplication Table")
    if figname != None:
        plt.savefig(figname)

    plt.show()

def generate_conjugacy_classes(group_elements, mul_func):
    """
    Generate the conjugacy classes of a group given its elements and a multiplication function.
    
    A conjugacy class of a group element is the set of elements that are conjugate to it, i.e., for any element 'g' 
    in the group, its conjugate by 'h' (another group element) is 'h^-1 * g * h', where '*' denotes the group multiplication,
    and 'h^-1' is the inverse of 'h'. This function computes these classes for all elements in the group.
    
    Args:
    - group_elements (list): A list of numpy arrays representing the group's elements.
    - mul_func (function): The multiplication function defining the group operation.
    
    Returns:
    - list: A list of lists, where each inner list represents a conjugacy class of the group.
    
    Each element of the group is considered, and its conjugates are computed with respect to all other elements. Unique
    conjugacy classes are then identified and returned.
    """
    have_generated = []  # Track elements that have already been considered for conjugacy classes
    conjugacy_classes = []  # Store the resulting conjugacy classes

    # Iterate through each group element to determine its conjugacy class
    for group_elem in group_elements:
        # Skip elements that have already been processed
        if any(np.array_equal(group_elem, elem) for elem in have_generated):
            continue
        have_generated.append(group_elem)  # Mark the current element as processed
        conjugacy_class = [group_elem]  # Initialize the conjugacy class with the current element
        
        # Iterate through the group elements to find conjugates
        for second_group_elem in group_elements:
            # Calculate the inverse of the second group element. Note: This works only for this group as every element has
            # Order 2 or 4, so raising them to the third power allways generates the inverse. 
            inv_group_elem = mul_func(mul_func(second_group_elem, second_group_elem), second_group_elem)

            # Compute the conjugate of 'group_elem' by 'second_group_elem' and its inverse
            conjugate_elem = mul_func(mul_func(inv_group_elem, group_elem), second_group_elem)

            # If the conjugate is already in the class, skip it
            if any(np.array_equal(conjugate_elem, elem) for elem in conjugacy_class):
                continue

            # Otherwise, add it to the class and mark it as processed
            conjugacy_class.append(conjugate_elem)
            have_generated.append(conjugate_elem)

        # Add the fully determined conjugacy class to the list of classes
        conjugacy_classes.append(conjugacy_class)

    return conjugacy_classes

def main():
    group_elements = generate_group_elements(4)
    mul_table = generate_mul_table(group_elements, simple_multiplication)
    omega_matrix = calculate_omega(group_elements)
    plot_omega_matrix(omega_matrix)

    covering_group = generate_group_elements(5)
    covering_mul_table = generate_mul_table(covering_group, covering_multiplication)

    plot_mul_table(covering_mul_table)
    conjugacy_classes = generate_conjugacy_classes(covering_group, covering_multiplication)

    for cc in conjugacy_classes:
        cc_as_ints = np.zeros(len(cc))
        for i, elem in enumerate(cc):
            cc_as_ints[i] = bool_array_to_int(elem)

        print(cc_as_ints)

    print("Number of CC's = ",len(conjugacy_classes))


    small_covering_group = generate_group_elements(3)
    covering_mul_table = generate_mul_table(small_covering_group, small_covering_multiplication)

    plot_mul_table(covering_mul_table)
    conjugacy_classes = generate_conjugacy_classes(small_covering_group, small_covering_multiplication)

    for cc in conjugacy_classes:
        cc_as_ints = np.zeros(len(cc))
        for i, elem in enumerate(cc):
            cc_as_ints[i] = bool_array_to_int(elem)

        print(cc_as_ints)

    print("Number of CC's = ",len(conjugacy_classes))


if __name__ == '__main__':
    main()