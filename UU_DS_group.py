from __future__ import annotations
import numpy as np
from itertools import combinations
import numpy as np
from tqdm import tqdm 
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def generate_UU_group():
    group = np.zeros((64,3))
    index = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                group[index] = np.array([i,j,k], dtype = int)
                index += 1

    return group

def generate_UU_reduced_Group():
    group = np.zeros((16,3))
    index = 0
    for i in range(2):
        for j in range(4):
            for k in range(2):
                group[index] = np.array([i,j,k], dtype = int)
                index += 1

    return group

def UU_multiplication(elem1, elem2):
    elem3 = elem1 + elem2
    elem3[0] += elem1[2]*elem2[1]
    return elem3 % 4

def UU_reduced_multiplication(elem1, elem2):
    elem3 = np.zeros_like(elem1)
    elem3[0] = (elem1[0] + elem2[0] + elem1[2]*elem2[1])%2    
    elem3[1] = (elem1[1] + elem2[1])%4
    elem3[2] = (elem1[2] + elem2[2])%2
    return elem3

def UU_inverse(elem, mulfunction):
    elem2 = elem
    for i in range(6):
        elem2 = mulfunction(elem,elem2)

    return elem2

def UU_reduced_inverse(elem, mulfunction):
    elem2 = elem
    for i in range(2):
        elem2 = mulfunction(elem,elem2)

    return elem2

def generate_conjugacy_classes(group_elements, mul_func, inv_func):
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
            inv_group_elem = inv_func(second_group_elem, mul_func)

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

def get_commutator(elem1, elem2, mul_func, inv_func):
    """
    Compute the commutator of two elements.
    """
    inv_elem1 = inv_func(elem1, mul_func)
    inv_elem2 = inv_func(elem2, mul_func)
    return mul_func(mul_func(inv_elem1, mul_func(inv_elem2, mul_func(elem1, elem2))), np.array([0,0,0]))

def generate_derived_subgroup(group_elements, mul_func, inv_func):
    """
    Generate the derived subgroup of a group given its elements and a multiplication function.
    """
    commutators = []

    for elem1 in group_elements:
        for elem2 in group_elements:
            commutator = get_commutator(elem1, elem2, mul_func,inv_func)
            # Add the commutator if it's not already in the list
            if not any(np.array_equal(commutator, com) for com in commutators):
                commutators.append(commutator)

    return np.array(commutators)

def group_elem_2_idx(elem):
    idx = 0
    for i, part in enumerate(elem):
        idx += part * 4**i
    
    return idx

def exponents_forall_elements(group, mul_func):
    exponents = np.ones(len(group))
    for i, elem in enumerate(group):
        transformed_elem = elem
        #print(elem)
        while not np.all(transformed_elem == np.zeros(3)):
            transformed_elem = mul_func(elem , transformed_elem)
            exponents[i] += 1
    
    return exponents



def generate_subgroups(group, group_2_idx_func, mul_func):
    exponents = []
    for elem in group:
        exponents_of_elem = [group_2_idx_func(elem)]
        elem_transformed = elem
        while np.all(elem_transformed != np.zeros(3)):
            elem_transformed = mul_func(elem_transformed, elem)
            exponents_of_elem.append(elem) 



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



def plot_mul_table_DS(mul_table, figname=None):
    """
    Visualize the multiplication table of a group with boolean array elements.
    
    Adjusts the color based on the values of each element:
    - Different values are plotted with different colors.
    The numerical values are displayed in the corresponding cells.
    
    Args:
    - mul_table (np.ndarray): A numpy array representing the group's multiplication table.
    """
    num_elements = 16
    display_matrix = np.zeros((num_elements, num_elements), dtype=int)
    color_matrix = np.zeros((num_elements, num_elements), dtype=int)  # Use int for color indices

    # Prepare display matrix and color matrix
    for i in range(num_elements):
        for j in range(num_elements):
            display_matrix[i, j] = abs(4 * mul_table[i, j][1] + mul_table[i, j][2])
            color_matrix[i, j] = mul_table[i, j][0]  # 0, 1, 2, or 3

    # Create a custom colormap with white, light gray, dark gray, and black
    cmap = ListedColormap(['white', 'lightgray', 'darkgray', 'black'])
    norm = BoundaryNorm([0, 1, 2, 3, 4], cmap.N)

    sns.set()
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(color_matrix, annot=display_matrix, fmt="d", cmap=cmap, norm=norm, cbar=False,
                     linewidths=0.5, linecolor='black', square=True)

    # Create a custom legend
    legend_elements = [
        mpatches.Patch(facecolor='white', edgecolor='black', label='0 (i^0)'),
        mpatches.Patch(facecolor='lightgray', edgecolor='black', label='1 (i^1)'),
        mpatches.Patch(facecolor='dimgray', edgecolor='black', label='2 (i^2)'),
        mpatches.Patch(facecolor='black', edgecolor='black', label='3 (i^3)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.20, 1))

    plt.xlabel("Group Elements (g)")
    plt.ylabel("Group Elements (h)")
    plt.title("Multiplication Table")

    if figname:
        plt.savefig(figname)
    plt.show()


def plot_mul_table_DSL(mul_table, figname=None):
    """
    Visualize the multiplication table of a group with boolean array elements.
    
    Adjusts the color based on the sign bit of each element:
    - Elements with a sign bit of 1 are plotted in blue.
    - Elements with a sign bit of 0 are in red.
    The numerical values (ignoring the sign bit) are displayed in the corresponding cells.
    
    Args:
    - mul_table (np.ndarray): A numpy array representing the group's multiplication table.
    """
    num_elements = mul_table.shape[0]
    display_matrix = np.zeros((num_elements, num_elements), dtype=int)
    color_matrix = np.zeros((num_elements, num_elements), dtype=float)  # Use float for color scaling

    # Prepare display matrix and color matrix
    for i in range(num_elements):
        for j in range(num_elements):
            display_matrix[i, j] = 2*mul_table[i,j][1] + mul_table[i,j][2]
            color_matrix[i, j] = mul_table[i,j][0]  # -1 for blue, 1 for red


    # Create a custom colormap
    cmap = ListedColormap(['white','gray'])

    sns.set()
    plt.figure(figsize=(10, 8))
    sns.heatmap(color_matrix, annot=display_matrix, fmt="d", cmap=cmap, cbar=False,
                linewidths=0.5, linecolor='black', square=True)
    plt.xlabel("Group Elements (g)")
    plt.ylabel("Group Elements (h)")
    plt.title("Multiplication Table")

    if figname:
        plt.savefig(figname)
    plt.show()



def main():
    print("UU-group:\n----------------------------------")
    group = generate_UU_group()
    derived_subgroup = generate_derived_subgroup(group, UU_multiplication, UU_inverse)

    print("Derived Subgroup Size:", len(derived_subgroup))
    print(derived_subgroup)

    CCs = generate_conjugacy_classes(group, UU_multiplication, UU_inverse)
    print(len(CCs))


    table = generate_mul_table(group, UU_multiplication)
    plot_mul_table_DS(table,"UUmultable.pdf")

    orders = exponents_forall_elements(group, UU_multiplication)
    uneque_orders, counts = np.unique(orders, return_counts = True)
    order_counts = dict(zip(uneque_orders, counts))
    print(order_counts)

    print("UU-reduced group:\n----------------------------------")
    group = generate_UU_reduced_Group()
    table = generate_mul_table(group, UU_reduced_multiplication)
    plot_mul_table_DSL(table,"UUReducedmultable.pdf")
    CCs = generate_conjugacy_classes(group, UU_reduced_multiplication, UU_reduced_inverse)
    print(len(CCs))
    orders = exponents_forall_elements(group, UU_reduced_multiplication)
    uneque_orders, counts = np.unique(orders, return_counts = True)
    order_counts = dict(zip(uneque_orders, counts))
    print(order_counts)



if __name__ == '__main__':
    main()