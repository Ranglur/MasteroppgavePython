import irrep_generator as irg
import diagonalizer as diag
import numpy as np
import matplotlib.pyplot as plt


#Test:
N = 12


# Setting up the symmetry group of the system
N_array=np.array([N])
lattice = irg.LatticeSite(N_array, basis=np.eye(1))
E = np.eye(1)
Pi = -np.eye(1)
point_group = [E, Pi]

space_group = irg.SpaceGroup(lattice, point_group)
irreps = space_group.generate_irreps()
character_table = irreps.generate_character_table(True)

group_elems = irreps.all_group_elements








def get_fingerprint(gsm: list, irreps: irg.IrrepData,N, simple_basis_int, simple_basis):
    characters = np.zeros(irreps.number_of_irreps, dtype=complex)
    irrep = np.zeros(shape=(irreps.number_of_irreps,2,2), dtype=complex)
    M = diag.M(N, 0)

    for i, conjugacy_class in enumerate(irreps.conjugacy_classes):
        transform : irg.SGTransformation = conjugacy_class[0]
        for m, state_m in enumerate(gsm):
            transformed_state = np.zeros_like(state_m)
            for j, coeff in enumerate(state_m):
                simple_state = simple_basis[j + M]
                
                transformed_simple_state = transform.operate_on_simple_state(simple_state)
                
                idx = diag.binary_search_in_block(simple_basis_int, 0, transformed_simple_state)
                
                transformed_state[idx] += coeff
            
            characters[i] += np.dot(state_m, transformed_state)
            for n, state_n in enumerate(gsm):
                irrep[i][m][n] = np.dot(state_n, transformed_state)

    characters = np.round(characters, decimals=5)
    print(characters)
    for matrix in irrep:
        print(matrix)

    n_reps = np.zeros(irreps.number_of_irreps, dtype=complex)

    for i in range(irreps.number_of_irreps):
        for j, conjugacy_class in enumerate(irreps.conjugacy_classes):
            n_reps[i] += len(conjugacy_class)*np.conj(character_table[i][j])*characters[j]

    return n_reps * 1/len(irreps.all_group_elements) 



# # #Diagonalizing the system

basis = diag.load_spin_basis("12_spin_system.npz")
#basis = diag.get_spin_system(N)
H_1 = diag.generate_hammiltonian_matrix(N, basis, diag.H_heisenberg_chain)
H_2 = diag.generate_hammiltonian_matrix(N, basis, diag.H_second_nearest)
H = diag.add_H_matrices(H_1,diag.scalar_mul_H_matrices(0.60,H_2))
spectrum, eigenbasis = diag.fulldiag(H, N, basis, return_eigenbasis=True)
simple_basis = diag.generate_simple_basis(N)
simple_basis = diag.sort_basis(simple_basis)


simple_basis_int = diag.prepare_sorted_basis(simple_basis)


gsm = [eigenbasis[0][0][0],eigenbasis[0][0][1]]
print(gsm)
print(irreps.conjugacy_classes)
n_reps = get_fingerprint(gsm, irreps,N, simple_basis_int, simple_basis)
print((n_reps))



ratios = np.linspace(0, 1, 42)
trivial_irrep = np.full_like(ratios, np.nan)  # Initialize with NaN
fourth_irrep = np.full_like(ratios, np.nan)  # Initialize with NaN

for i, ratio in enumerate(ratios):
    print(f"finished ratio: {ratio}")
    # Adjust the Hamiltonian construction for J'/J
    H = diag.add_H_matrices(H_1, diag.scalar_mul_H_matrices(ratio, H_2))
    spectrum, eigenbasis = diag.fulldiag(H, N, basis, return_eigenbasis=True)
    n_reps = get_fingerprint([eigenbasis[0][0][0]], irreps, N, simple_basis_int, simple_basis)

    # Check for trivial irrep
    if np.isclose(n_reps[0], 1):
        trivial_irrep[i] = spectrum[0][0][0]
        fourth_irrep[i] = spectrum[0][0][1]
    # Check for fourth irrep
    elif np.isclose(n_reps[3], 1):
        trivial_irrep[i] = spectrum[0][0][1]
        fourth_irrep[i] = spectrum[0][0][0]

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(ratios, trivial_irrep, label='Trivial Irrep', marker='o')
plt.plot(ratios, fourth_irrep, label='Fourth Irrep', marker='x')
plt.title('Energy of Lowest Two States vs J\'/J')
plt.xlabel('J\'/J')
plt.ylabel('Energy')
plt.grid(True)
plt.legend()
plt.savefig('Eigenstates_wrt_J_prime.pdf')
plt.show()




        

                
        


