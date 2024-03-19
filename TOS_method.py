import irrep_generator as irg
import diagonalizer as diag
import numpy as np

# # #Diagonalizing the system
N = 8
basis = diag.get_spin_system(N)

def Test_H(state: np.ndarray, S: float, Sz: float, N: int, simple_basis: np.ndarray, simple_basis_int: np.ndarray, J = 1):
    return diag.H_heisenberg_Majumdar_Gosh(state, S, Sz, N, simple_basis, simple_basis_int, J = 1, ratio = 0.4)

H = diag.generate_hammiltonian_matrix(N, basis, Test_H)
spectrum, eigenbasis = diag.fulldiag(H, N, basis, return_eigenbasis=True)
simple_basis = diag.generate_simple_basis(N)
simple_basis = diag.sort_basis(simple_basis)


simple_basis_int = diag.prepare_sorted_basis(simple_basis)



# Setting up the symmetry group of the system
N_array=np.array([8])
lattice = irg.LatticeSite(N_array, basis=np.eye(1))
E = np.eye(1)
Pi = -np.eye(1)
point_group = [E, Pi]

space_group = irg.SpaceGroup(lattice, point_group)
irreps = space_group.generate_irreps()
character_table = irreps.generate_character_table()
group_elems = irreps.all_group_elements

# # # Parameters for GSM:
M = diag.M(N, 0)

gsm = [eigenbasis[0][0][0] , eigenbasis[0][0][1]]




def get_fingerprint(gsm: list, irreps: irg.IrrepData, simple_basis_int, simple_basis):
    characters = np.zeros(irreps.number_of_irreps, dtype=complex)

    for i, conjugacy_class in enumerate(irreps.conjugacy_classes):
        transform : irg.SGTransformation = conjugacy_class[0]
        for state in gsm:
            transformed_state = np.zeros_like(state)
            for j, coeff in enumerate(state):
                simple_state = simple_basis[j + M]
                
                transformed_simple_state = transform.operate_on_simple_state(simple_state)
                
                idx = diag.binary_search_in_block(simple_basis_int, 0, transformed_simple_state)
                
                transformed_state[idx] += coeff
            
            characters[i] += np.dot(state, transformed_state)

    characters = np.round(characters, decimals=5)
    


    n_reps = np.zeros(irreps.number_of_irreps, dtype=complex)

    for i in range(irreps.number_of_irreps):
        for j, conjugacy_class in enumerate(irreps.conjugacy_classes):
            n_reps[i] += len(conjugacy_class)*np.conj(character_table[i][j])*characters[j]

    return n_reps * 1/len(irreps.all_group_elements) 
    


n_reps = get_fingerprint(gsm, irreps, simple_basis_int, simple_basis)
print((n_reps))







        

                
        


