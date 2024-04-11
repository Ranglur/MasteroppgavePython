from __future__ import annotations
import numpy as np
from itertools import combinations
import numpy as np
from tqdm import tqdm 


def generate_UU_group():
    group = np.zeros((64,3))
    index = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                group[index] = np.array([i,j,k], dtype = int)
                index += 1

    return group

group = generate_UU_group()



def UU_multiplication(elem1, elem2):
    elem3 = elem1 + elem2
    elem3[0] += elem1[2]*elem2[1]
    return elem3 % 4

def UU_inverse(elem, mulfunction):
    elem2 = elem
    for i in range(6):
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

def UU_commutator(elem1, elem2, mul_func):
    """
    Compute the commutator of two elements.
    """
    inv_elem1 = UU_inverse(elem1, mul_func)
    inv_elem2 = UU_inverse(elem2, mul_func)
    return mul_func(mul_func(inv_elem1, mul_func(inv_elem2, mul_func(elem1, elem2))), np.array([0,0,0]))

def generate_derived_subgroup(group_elements, mul_func):
    """
    Generate the derived subgroup of a group given its elements and a multiplication function.
    """
    commutators = []

    for elem1 in group_elements:
        for elem2 in group_elements:
            commutator = UU_commutator(elem1, elem2, mul_func)
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
    exponents = np.zeros(len(group))
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






derived_subgroup = generate_derived_subgroup(group, UU_multiplication)
print("Derived Subgroup Size:", len(derived_subgroup))
print(derived_subgroup)

elem1 = np.array([0,1,2])
elem2 = np.array([2, 3 ,2])

print(UU_multiplication(elem1, elem2))
print(UU_inverse(elem1, UU_multiplication))

group = generate_UU_group()

CCs = generate_conjugacy_classes(group, UU_multiplication, UU_inverse)
print(len(CCs))

print(exponents_forall_elements(group, UU_multiplication))



