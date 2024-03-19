from __future__ import annotations
import numpy as np
import sympy as sp
import numpy.linalg as la




class LatticeSite:
    def __init__(self, N_array: np.ndarray, basis: np.ndarray) -> None:
        # Initializer for the LatticeSite class.
        # N_array is a numpy array specifying the number of lattice points in each dimension.
        # basis is a numpy array representing the basis vectors of the Bravais lattice.

        # Check for valid dimensions (1 to 3 dimensions are allowed)
        self.d = len(N_array)
        if 1 > self.d or self.d > 3:
            raise ValueError("Illegal dimension")

        # Ensure all elements of N_array are positive natural numbers
        if np.any(N_array <= 0):
            raise ValueError("N not a natural number")

        # Check if the basis is a square matrix with dimensions matching the lattice dimensions
        if np.shape(basis) != (self.d, self.d):
            print(np.shape(basis))
            print((self.d, self.d))
            raise ValueError("Not a valid basis")

        # Store the input parameters and initialize the lattice
        self.N_array = N_array
        self.N = np.product(N_array)  # Total number of lattice points
        self.lattice = np.zeros(shape=(np.product(N_array), self.d))  # Create the lattice array
        self.basis = basis  # Store the basis vectors

        # Populate the lattice with points
        for idx in range(self.N):
            for i in range(self.d):
                prod = np.product(N_array[0:i])
                self.lattice[idx, i] = (idx // prod) % N_array[i]

def point_2_index(point: np.ndarray, N_array: np.ndarray) -> int:
    """
    Convert a point in lattice coordinates to its linear index.

    Args:
    point (np.ndarray): The coordinates of the point in the lattice.
    N_array (np.ndarray): An array representing the number of lattice points in each dimension.

    Returns:
    int: The linear index corresponding to the given lattice point.

    The function iterates through each dimension of the lattice, calculating the contribution of each dimension to the linear index based on the point's coordinates and the size of the lattice in each dimension.
    """
    idx = 0  # Initialize the linear index
    for i, N_i in enumerate(N_array):
        # Calculate the product of the sizes of previous dimensions to determine positional value
        prod = np.product(N_array[0:i])
        # Update the linear index by adding the contribution from the current dimension
        idx += point[i] * prod
    return idx

def index_2_point(idx: int, N_array: np.ndarray) -> np.ndarray:
    """
    Convert a linear index to its corresponding point in lattice coordinates.

    Args:
    idx (int): The linear index in the lattice.
    N_array (np.ndarray): An array representing the number of lattice points in each dimension.

    Returns:
    np.ndarray: The coordinates of the point in the lattice corresponding to the given index.

    The function calculates the coordinates of the point in each dimension by iterating through the dimensions and using modular arithmetic and integer division.
    """
    d = len(N_array)  # The number of dimensions in the lattice
    point = np.zeros(d)  # Initialize the point array
    for i, N_i in enumerate(N_array):
        # Calculate the product of the sizes of previous dimensions
        prod = np.product(N_array[0:i])
        # Calculate the coordinate in the current dimension
        point[i] = (idx // prod) % N_i
    return point
    
def shift_point(point: np.ndarray, N_array: np.ndarray) -> None:
    """
    Shift a point within a lattice, applying periodic boundary conditions.

    This function modifies the point in place. It ensures that each coordinate of the point is within the lattice boundaries defined by N_array. If a coordinate goes out of bounds, it is wrapped around using periodic boundary conditions.

    Args:
    point (np.ndarray): The coordinates of the point in the lattice to be shifted.
    N_array (np.ndarray): An array representing the number of lattice points in each dimension.

    The function iterates through each dimension of the point, adjusting coordinates that are out of bounds.
    """
    d = len(N_array)  # Number of dimensions in the lattice
    for i, N_i in enumerate(N_array):
        # Check if the coordinate in the current dimension is out of bounds (negative)
        if point[i] < 0:
            # Calculate the number of times the point wraps around the lattice
            helper = point[i] * (-1)
            point[i] += N_i * (helper // N_i + 1)

        # Apply periodic boundary conditions to ensure the point is within the lattice
        point[i] = point[i] % N_i

def get_reciprocal_lattice(lattice: LatticeSite) -> LatticeSite:
    # Temperary function, shoud be replaced with general function for 2d and 3d
    if lattice.d == 1:
        new_basis = 2*np.pi*lattice.basis/(lattice.N_array[0]*np.dot(lattice.basis[0],lattice.basis[0]))
        return LatticeSite(lattice.N_array, new_basis)
    else:
        raise ValueError

def star(k_vec: np.ndarray, PointGroup, N_array: np.ndarray):
    """
    Compute the star of a wave vector under the point group transformations.

    The 'star' of a wave vector consists of all vectors that can be obtained by applying
    the point group transformations to the original wave vector. This function also applies
    periodic boundary conditions to ensure the transformed wave vectors are within the 
    first Brillouin zone.

    Args:
    k_vec (np.ndarray): The original wave vector.
    PointGroup (iterable): A collection of matrices representing point group transformations.
    N_array (np.ndarray): An array representing the reciprocal lattice vectors' magnitudes for periodic boundary conditions.

    Returns:
    tuple: A tuple containing two lists. The first list is the star of k_vec, and the second list is the corresponding unique point group transformation matrices.

    The function iterates through the point group transformations, applies them to k_vec, 
    and then uses shift_point to apply periodic boundary conditions. It ensures uniqueness 
    of the vectors in the star.
    """
    star = []  # Initialize list to store the star of k_vec
    unique_array = []  # Initialize list to store unique transformation matrices

    for A in PointGroup:
        # Apply point group transformation to k_vec
        k_new = A @ k_vec
        # Apply periodic boundary conditions to ensure k_new is within the first Brillouin zone
        shift_point(k_new, N_array)

        # Check for uniqueness of the transformed vector
        if np.any([np.allclose(k_new, k) for k in star]):
            continue

        # Append unique transformed vector and its corresponding transformation matrix
        star.append(k_new)
        unique_array.append(A)

    return star, unique_array

def get_K_group(point_group: list, star: list, N_array: np.ndarray):
    """
    Determine the subgroup of the point group that leaves the star of a wave vector invariant.

    This function checks which elements of the point group leave all vectors in the star of a given wave vector unchanged. This subgroup is known as the 'little group' or 'k-group'.

    Args:
    point_group (list): A list of matrices representing the point group transformations.
    star (list): A list of vectors representing the star of a wave vector.
    N_array (np.ndarray): An array representing the reciprocal lattice vectors' magnitudes for periodic boundary conditions.

    Returns:
    list: A list of point group elements (matrices) that form the K-group for the given star.

    The function iterates through each element of the point group. For each element, it tests whether applying this transformation to each vector in the star leaves it invariant, considering periodic boundary conditions.
    """
    K_group = []  # Initialize list to store elements of the K-group

    for point_group_element in point_group:
        is_in_group = True  # Initialize flag to check if the element is in the K-group

        for k_vec in star:
            # Apply the point group transformation to k_vec
            new_k = point_group_element @ k_vec
            # Apply periodic boundary conditions
            shift_point(new_k, N_array)

            # Check if the transformed vector is different from the original
            if not np.allclose(new_k, k_vec):
                is_in_group = False
                break

        # If the point group element leaves all vectors in the star invariant, add it to K_group
        if is_in_group:
            K_group.append(point_group_element)

    return K_group

def get_irreps(point_group: list) -> IrrepData:
    # Bad temporary solution whilst i think of ways to do this propperly
    # Should take in a point group and return its irreps
    E = np.array([[1]])
    Pi = np.array([[-1]])
    C1_cc = [[E]]
    C2_cc = [[E], [Pi]]
    
    if len(point_group) == 1:
        result = IrrepData(C1_cc)
        result[0,E] = np.eye(1)
        
        return result
    if len(point_group) == 2:
        result = IrrepData(C2_cc)
        result[0,E] = np.eye(1)
        result[0,Pi] = np.eye(1)
        result[1,E] = np.eye(1)
        result[1,Pi] = -np.eye(1)

        return result
    
    raise ValueError("Only irreps for C1 and C2 have been implemented")

def generate_conjugacy_classes_pg(group: list) -> list:
    """
    Generate the conjugacy classes of a group.

    In group theory, a conjugacy class of a group is the set of elements that are conjugate to one another. This function computes the conjugacy classes for a given group represented as a list of matrices (group elements).

    Args:
    group (list): A list of matrices representing the elements of the group.

    Returns:
    list: A list of lists, where each inner list is a conjugacy class of the group.

    The function iterates over elements of the group, generating the conjugacy class for each element that hasn't been processed yet.
    """
    have_generated = []  # List to keep track of elements for which conjugacy classes have been generated
    conjugacy_classes = []  # List to store the conjugacy classes

    for group_elem in group:
        # Check if the conjugacy class for this element has already been generated
        if group_elem in have_generated:
            continue

        have_generated.append(group_elem)
        conjugacy_class = []  # Initialize the current conjugacy class

        for second_group_elem in group:
            # Generate the conjugate of group_elem by second_group_elem
            conjugate_group_elem = la.inv(second_group_elem) @ group_elem @ second_group_elem

            # Check if the conjugate is already in the conjugacy class
            if conjugate_group_elem in conjugacy_class:
                continue

            # Add the conjugate to the conjugacy class and the list of processed elements
            conjugacy_class.append(conjugate_group_elem)
            have_generated.append(conjugate_group_elem)

        conjugacy_classes.append(conjugacy_class)

    return conjugacy_classes


class SGTransformation:
    """
    A class representing space group transformations in crystallography.

    This class handles transformations that include both rotational and translational components, which are essential for modeling the symmetry operations in a crystal lattice.

    Attributes:
    A (np.ndarray): The rotational part of the transformation.
    t (np.ndarray): The translational part of the transformation.
    N_array (np.ndarray): An array representing the size of the lattice in each dimension.

    Methods:
    apply_transformation(point): Apply the space group transformation to a lattice point.
    determine_permutation(): Determine the permutation of lattice points induced by the transformation.
    operate_on_simple_state(state): Apply the transformation to a simple state represented as an array.
    inverse(): Compute the inverse of the transformation.
    __mul__(other): Implement the multiplication of two SGTransformation instances.
    __eq__(other): Check equality of two SGTransformation instances.
    __pow__(exponent): Raise the transformation to a power.
    __hash__(): Provide a hash representation of the transformation.
    __str__(): Provide a string representation of the transformation.
    __repr__(): Provide a detailed string representation of the transformation.
    """

    def __init__(self, A: np.ndarray, t: np.ndarray, N_array: np.ndarray):
        """
        Initialize the SGTransformation instance.
        
        Args:
        A (np.ndarray): The rotational matrix component of the transformation.
        t (np.ndarray): The translational vector component of the transformation.
        N_array (np.ndarray): The dimensions of the lattice.
        """
        self.A = A  # The rotational component
        self.t = t  # The translational component
        self.N_array = N_array  # Lattice dimensions
        self.determine_permutation()  # Determine the permutation of lattice points

    def apply_transformation(self, point):
        """
        Apply the space group transformation to a given lattice point.

        Args:
        point (np.ndarray): The lattice point to be transformed.

        Returns:
        np.ndarray: The transformed lattice point.
        """
        # Apply the rotational and translational components
        transformed_point = np.dot(self.A, point) + self.t
        
        # Rounding to mitigate floating-point errors and applying periodic boundary conditions
        transformed_point = np.round(transformed_point).astype(int)
        shift_point(transformed_point, self.N_array)
        return transformed_point
    
    def determine_permutation(self):
        """
        Determine the permutation of lattice points induced by the transformation.
        
        This method computes how lattice points are rearranged (permuted) by the transformation.
        """
        N = np.product(self.N_array)  # Total number of points in the lattice
        self.permutation = np.zeros(N, dtype=int)  # Initialize the permutation array
        for i in range(N):
            point = index_2_point(i, self.N_array)  # Convert index to lattice point
            transformed_point = self.apply_transformation(point)  # Apply transformation
            index = point_2_index(transformed_point, self.N_array)  # Convert back to index
            self.permutation[i] = index  # Store in permutation array

        

    def operate_on_simple_state(self, state: np.ndarray) -> np.ndarray:
        """
        Apply the space group transformation to a simple state.

        This method permutes the elements of a given state array based on the transformation's effect on the lattice points.

        Args:
        state (np.ndarray): An array representing the state before transformation.

        Returns:
        np.ndarray: The state after applying the transformation.
        """
        # Initialize the transformed state array
        transformed_state = np.zeros_like(state)
        for i in range(len(state)):
            # Apply the permutation to each element of the state
            transformed_state[self.permutation[i]] = state[i]

        return transformed_state

    def __mul__(self, other: SGTransformation):
        """
        Multiply two SGTransformation instances.

        This method implements the composition of two space group transformations.

        Args:
        other (SGTransformation): Another SGTransformation instance.

        Returns:
        SGTransformation: A new SGTransformation instance representing the composition of self and other.
        """
        # Multiply the rotational components and combine the translational components
        new_A = np.dot(self.A, other.A)
        new_t = np.dot(self.A, other.t) + self.t
        # Apply periodic boundary conditions
        shift_point(new_t, self.N_array)
        return SGTransformation(new_A, new_t, self.N_array)

    def __eq__(self, other: SGTransformation):
        """
        Check if two SGTransformation instances are equal.

        Args:
        other (SGTransformation): Another SGTransformation instance to compare with.

        Returns:
        bool: True if both instances are equal, False otherwise.
        """
        # Check equality of all components
        return np.array_equal(self.A, other.A) and np.array_equal(self.t, other.t) and np.array_equal(self.N_array, other.N_array)

    def inverse(self):
        """
        Compute the inverse of the space group transformation.

        Returns:
        SGTransformation: A new SGTransformation instance representing the inverse of the current transformation.
        """
        # Compute the inverse of the rotational component
        inverse_A = la.inv(self.A).astype(int)
        # Compute the inverse of the translational component
        inverse_t = -inverse_A @ self.t
        # Apply periodic boundary conditions
        shift_point(inverse_t, self.N_array)
        return SGTransformation(inverse_A, inverse_t, self.N_array)

    def __pow__(self, exponent: int):
        """
        Raise the transformation to a power.

        Args:
        exponent (int): The power to which the transformation is to be raised.

        Returns:
        SGTransformation: A new SGTransformation instance representing the transformation raised to the specified power.
        """
        if exponent == 0:
            # Return the identity transformation
            identity_matrix = np.eye(len(self.N_array), dtype=int)
            zero_translation = np.zeros_like(self.t)
            return SGTransformation(identity_matrix, zero_translation, self.N_array)
        
        elif exponent > 0:
            # Perform repeated multiplication for a positive exponent
            result = self
            for _ in range(exponent - 1):
                result = result * self
            return result
        
        else:
            # Calculate the inverse and perform repeated multiplication for a negative exponent
            inverse = self.inverse()
            result = inverse
            for _ in range(abs(exponent) - 1):
                result = result * inverse
            return result

    def __hash__(self):
        """
        Generate a hash representation of the SGTransformation.

        Returns:
        int: A hash value representing the transformation.
        """
        # Convert arrays into a hashable type (tuples of tuples)
        A_tuple = tuple(map(tuple, self.A))
        t_tuple = tuple(self.t)
        N_array_tuple = tuple(self.N_array)
        # Combine the hashes of the individual elements
        return hash((A_tuple, t_tuple, N_array_tuple))

    # Functions for printing and debugging
    def __str__(self):
        """
        Provide a string representation of the SGTransformation.

        Returns:
        str: A string representation of the transformation.
        """
        return f"A:\n{self.A}\nt:\n{self.t}"

    def __repr__(self):
        """
        Provide a detailed string representation of the SGTransformation.

        Returns:
        str: A detailed string representation of the transformation.
        """
        return f"SGTransformation(A={self.A}, t={self.t}, N_array={self.N_array})"

class SpaceGroup:
    """
    A class representing space groups in crystallography.

    Space groups describe the symmetries of a crystal lattice including rotations, reflections, translations, and glide reflections. This class provides methods to handle these transformations and analyze their implications in a crystal structure.

    Attributes:
    lattice (LatticeSite): The lattice on which the space group operations are defined.
    reciprocal_lattice (LatticeSite): The reciprocal lattice derived from the given lattice.
    point_group (iterable): The point group associated with the space group.
    SG_transformations (list): A list of space group transformations.

    Methods:
    generate_SG_transformations(): Generate space group transformations from the point group and lattice.
    generate_conjugacy_classes(): Compute the conjugacy classes of the space group transformations.
    convert_to_cartesian(k_vector, r_vector): Convert vectors from reciprocal and real space bases to Cartesian coordinates.
    partition_k_space(): Partition the reciprocal space into distinct stars.
    generate_irreps(rounding): Generate irreducible representations of the space group.
    """

    def __init__(self, lattice: LatticeSite, point_group):
        """
        Initialize the SpaceGroup instance.

        Args:
        lattice (LatticeSite): The lattice defining the space.
        point_group (iterable): The point group defining the symmetries of the lattice.
        """
        self.lattice = lattice
        self.reciprocal_lattice = get_reciprocal_lattice(lattice)
        self.point_group = point_group
        self.SG_transformations = self.generate_SG_transformations()

    def generate_SG_transformations(self) -> list:
        """
        Generate the space group transformations.

        This method combines each element of the point group with translations defined by the lattice to create the complete set of space group transformations.

        Returns:
        list: A list of SGTransformation instances representing the space group transformations.
        """
        transformations = []
        for PG_element in self.point_group:
            for idx in range(self.lattice.N):
                translation = self.lattice.lattice[idx]
                transformations.append(SGTransformation(PG_element, translation, self.lattice.N_array))
        return transformations

    def generate_conjugacy_classes(self) -> list:
        """
        Generate the conjugacy classes of the space group transformations.

        This method computes the conjugacy classes for the transformations in the space group, which are sets of elements that are conjugate to each other.

        Returns:
        list: A list of lists, each inner list representing a conjugacy class of the space group transformations.
        """
        have_generated = []
        conjugacy_classes = []

        for SGT in self.SG_transformations:
            if SGT in have_generated:
                continue
            have_generated.append(SGT)
            conjugacy_class = []

            for second_SGT in self.SG_transformations:
                # Compute the conjugate of SGT by second_SGT
                conjugate_SGT = second_SGT**(-1) * SGT * second_SGT
                if conjugate_SGT in conjugacy_class:
                    continue

                conjugacy_class.append(conjugate_SGT)
                have_generated.append(conjugate_SGT)

            conjugacy_classes.append(conjugacy_class)

        return conjugacy_classes

    def convert_to_cartesian(self, k_vector, r_vector):
        """
        Convert vectors from lattice bases to Cartesian coordinates.

        Args:
        k_vector (np.ndarray): A vector in the reciprocal lattice basis.
        r_vector (np.ndarray): A vector in the real space lattice basis.

        Returns:
        tuple: The vectors converted to Cartesian coordinates.
        """
        k_cartesian = np.dot(self.reciprocal_lattice.basis, k_vector)
        r_cartesian = np.dot(self.lattice.basis, r_vector)
        
        return k_cartesian, r_cartesian

    def partition_k_space(self):
        """
        Partition the reciprocal space into distinct 'stars'.

        A 'star' is a set of k-vectors related by the symmetry operations of the point group. This method identifies unique stars in the reciprocal lattice.

        Returns:
        tuple: Two lists, one of k-stars and the other of unique transformation arrays corresponding to each k-star.
        """
        k_stars = []
        unique_arrays = []
        k_generated = []

        for k in self.reciprocal_lattice.lattice:
            if k in k_generated:
                continue
            k_star, unique_array = star(k, self.point_group, self.lattice.N_array)
            
            for k_new in k_star:
                k_generated.append(k_new)
            k_stars.append(k_star)
            unique_arrays.append(unique_array)

        return k_stars, unique_arrays

    def generate_irreps(self, rounding=10e-15) -> IrrepData:
        """
        Generate irreducible representations (irreps) for each k-star in the reciprocal lattice.

        Irreps are fundamental in understanding the symmetries of electronic states in crystals. This method computes the irreps for each k-star considering the space group symmetries.

        Args:
        rounding (float): The precision for rounding the matrix elements of the irreps.

        Returns:
        IrrepData: A data structure containing the irreps for each k-star.

        This function computes the irreps for each k-vector in the k-stars, taking into account the symmetry operations of the space group and applying rounding to the matrix elements for numerical stability.
        """
        data = IrrepData(self.generate_conjugacy_classes())  # Initialize Irrep data structure
        k_stars, unique_arrays = self.partition_k_space()  # Partition the reciprocal space

        idx = 0  # Index for storing irreps in data structure
        zero_translation = np.zeros(len(self.lattice.N_array))  # Zero translation vector

        for k_star, unique_array in zip(k_stars, unique_arrays):
            k = k_star[0]  # Reference k-vector in the k-star

            K_group = get_K_group(self.point_group, k_star, self.lattice.N_array)  # Get little group for k
            K_Irreps = get_irreps(K_group)  # Compute irreps of the little group

            for i in range(K_Irreps.number_of_irreps):
                K_dim = np.shape(K_Irreps[i, K_group[0]])[0]  # Dimension of the i-th irrep
                Irrep_size = len(unique_array) * K_dim  # Size of the irrep matrix

                for sg_element in self.SG_transformations:
                    data[idx, sg_element] = np.zeros(shape=(Irrep_size, Irrep_size), dtype=complex)
                    A = sg_element.A  # Rotational part of the space group element

                    for l, A_l in enumerate(unique_array):
                        k_m = A @ k_star[l]  # Apply rotation to k-vector
                        shift_point(k_m, self.lattice.N_array)  # Apply periodic boundary conditions
                        m = k_star.index(k_m)  # Find index of rotated k-vector in the k-star
                        A_m = unique_array[m]

                        # Calculate the conjugate space group element
                        new_sg_elem = SGTransformation(A_m, zero_translation, self.lattice.N_array)**(-1) * sg_element * SGTransformation(A_l, zero_translation, self.lattice.N_array)
                        B = new_sg_elem.A  # Rotational part of the conjugate element
                        sigma = new_sg_elem.t  # Translational part of the conjugate element

                        # Calculate the phase factor and apply it to the irrep matrix
                        D = np.exp(1j * np.dot(self.reciprocal_lattice.basis @ k, self.lattice.basis @ sigma)) * K_Irreps[i, B]
                        
                        # Round the real and imaginary parts for numerical stability
                        D_real_rounded = np.round(D.real, decimals=int(-np.log10(rounding)))
                        D_imag_rounded = np.round(D.imag, decimals=int(-np.log10(rounding)))
                        D_rounded = D_real_rounded + 1j * D_imag_rounded

                        # Store the rounded irrep matrix in the data structure
                        data[idx, sg_element][l * K_dim:(l + 1) * K_dim, m * K_dim:(m + 1) * K_dim] = D_rounded
                    
                idx += 1 

        
        return data


class IrrepData:
    def __init__(self, conjugacy_classes: list):
        self.conjugacy_classes = conjugacy_classes
        self.all_group_elements = [element for cl in conjugacy_classes for element in cl]
        self.order_of_group = len(self.all_group_elements)
        self.number_of_irreps = len(conjugacy_classes)
        # Initialize a list with n_cc lists, each of size |G|
        self.irrep_matrices = [[np.zeros(shape = (0,0))] * self.order_of_group for _ in range(self.number_of_irreps)]

    def __getitem__(self, key):
        # Here, the key[0] is the irrep index and key[1] is the group element
        irrep_index, group_element = key
        element_index = self.all_group_elements.index(group_element)
        return self.irrep_matrices[irrep_index][element_index]

    def __setitem__(self, key, value: np.ndarray):
        # Here, the key[0] is the irrep index and key[1] is the group element
        irrep_index, group_element = key
        element_index = self.all_group_elements.index(group_element)
        # Make sure the matrix dimensions match expected dimensions if already set
        current_matrix = self.irrep_matrices[irrep_index][element_index]
        if current_matrix.size > 0:
            assert current_matrix.shape == value.shape, "Irrep matrix dimensions do not match existing dimensions."
        self.irrep_matrices[irrep_index][element_index] = value

    def sort_irreps_by_dimension(self):
        # Extract dimensions of each irrep using the first matrix of each
        dimensions = [np.shape(self.irrep_matrices[i][0])[0] for i in range(self.number_of_irreps)]

        # Combine dimensions with their corresponding indices
        dim_irrep_pairs = list(zip(dimensions, range(self.number_of_irreps)))

        # Sort pairs by dimension (first element of the tuple)
        sorted_dim_irrep_pairs = sorted(dim_irrep_pairs, key=lambda x: x[0])

        # Reorder irrep_matrices and conjugacy_classes based on the sorted indices
        self.irrep_matrices = [self.irrep_matrices[i] for _, i in sorted_dim_irrep_pairs]
        # If necessary, also sort conjugacy_classes
    
    def generate_character_table(self, print_table=False):
        # Initialize an empty 2D array for the character table with a complex data type
        character_table = np.zeros((self.number_of_irreps, len(self.conjugacy_classes)), dtype=np.complex_)

        # Sort the irreps by dimension before generating the table
        self.sort_irreps_by_dimension()

        # Iterate through each irrep to build the character table
        for i in range(self.number_of_irreps):
            # Since all matrices are filled, directly take the shape of the first matrix
            irrep_dimension = np.shape(self.irrep_matrices[i][0])[0]

            # Iterate through each conjugacy class to calculate characters
            for j, conjugacy_class in enumerate(self.conjugacy_classes):
                # Assuming the first element's matrix of each class represents the class
                representative_element = conjugacy_class[0]
                element_index = self.all_group_elements.index(representative_element)
                irrep_matrix = self.irrep_matrices[i][element_index]
                # The character is the trace of the matrix, preserved as complex
                character_table[i, j] = np.trace(irrep_matrix)

        # Optionally print the character table
        if print_table:
            # Creating a header for the table
            header = ["Irrep (dim)"] + [f"Class {i}" for i in range(len(self.conjugacy_classes))]
            print("\t".join(header))
            for i in range(self.number_of_irreps):
                # Print each row of characters, starting with Irrep 0 and its dimension
                irrep_dimension = np.shape(self.irrep_matrices[i][0])[0]
                row = [f"Irrep {i} ({irrep_dimension})"] + [str(character_table[i, j]) for j in range(character_table.shape[1])]
                print("\t".join(row))

        return character_table



def test_generate_cc():
    lattice = LatticeSite(N_array=np.array([4]), basis=np.eye(1))
    point_group = [np.array([[1]]), np.array([[-1]])]

    space_group = SpaceGroup(lattice, point_group)
    cc_list = space_group.generate_conjugacy_classes()
    for cc in cc_list:    
        print(cc)


def test_irrep_setting():
    # Define the conjugacy classes for the group {E, Pi}
    E = np.array([[1]])
    Pi = np.array([[-1]])
    pg_cc = [[E], [Pi]]

    # Initialize IrrepData with the conjugacy classes
    irrep_data = IrrepData(pg_cc)
    print(irrep_data.irrep_matrices)

    # Set the irreps using the [] operator
    irrep_data[0, E] = np.eye(1)
    irrep_data[0, Pi] = np.eye(1)
    irrep_data[1, E] = np.eye(1)
    irrep_data[1, Pi] = -np.eye(1)

    # Test the setting
    assert np.array_equal(irrep_data[0, E], np.eye(1)), "Irrep for (0,E) is incorrect"
    assert np.array_equal(irrep_data[0, Pi], np.eye(1)), "Irrep for (0,Pi) is incorrect"
    assert np.array_equal(irrep_data[1, E], np.eye(1)), "Irrep for (1,E) is incorrect"
    assert np.array_equal(irrep_data[1, Pi], -np.eye(1)), "Irrep for (1,Pi) is incorrect"

    print("All irreps set and retrieved correctly.")

    irrep_data.generate_character_table(True)

def test_generate_irreps(N):
    
    N_array=np.array([4])
    lattice = LatticeSite(N_array, basis=np.eye(1))
    E = np.eye(1)
    Pi = -np.eye(1)
    point_group = [E, Pi]  

    space_group = SpaceGroup(lattice, point_group)
    irreps = space_group.generate_irreps()
    
    # for i in range(irreps.number_of_irreps):
    #     for group_elem in irreps.all_group_elements:
    #         print(irreps[i,group_elem])

    irreps.generate_character_table(True)

def main():
# Example usage:
    #test_generate_cc()
    N_array=np.array([8])
    lattice = LatticeSite(N_array, basis=np.eye(1))
    E = np.eye(1)
    Pi = -np.eye(1)
    point_group = [E, Pi]
    
    space_group = SpaceGroup(lattice, point_group)
    irreps = space_group.generate_irreps()
 
    ct = irreps.generate_character_table(True)
    conjugacy_classes = irreps.conjugacy_classes
    simple_state = np.array([False,True,False,True,False,True,False,True],dtype=bool)

    for conjugacy_class in conjugacy_classes:
        transform: SGTransformation = conjugacy_class[0]
        print(transform.operate_on_simple_state(simple_state))
        

    #print(space_group.generate_conjugacy_classes())
    # stars, uneque_arrays = space_group.partition_k_space()
    # print(stars)

    #print(space_group.reciprocal_lattice.basis @ space_group.reciprocal_lattice.lattice[3])

    # test_irrep_setting()

    #test_generate_irreps(4)

    return 0


# Run main if file not imported
if __name__ == "__main__":
    main()


