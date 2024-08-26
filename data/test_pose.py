
import numpy as np

def quaternion_to_rotation_matrix(q):
    q_w, q_x, q_y, q_z = q
    R = np.array([
        [1 - 2*(q_y**2 + q_z**2), 2*(q_x*q_y - q_z*q_w), 2*(q_x*q_z + q_y*q_w)],
        [2*(q_x*q_y + q_z*q_w), 1 - 2*(q_x**2 + q_z**2), 2*(q_y*q_z - q_x*q_w)],
        [2*(q_x*q_z - q_y*q_w), 2*(q_y*q_z + q_x*q_w), 1 - 2*(q_x**2 + q_y**2)]
    ])
    return R

def create_homogeneous_matrix(translation, quaternion):
    R = quaternion_to_rotation_matrix(quaternion)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation
    return T

# Vecteur donné
vector_left = [7.008041858673095703e+00, -3.013244628906250000e+01, -3.011430501937866211e+00,
          -1.910096853971481323e-01, 1.507585942745208740e-01, 8.008892536163330078e-01, 5.471411943435668945e-01]


vector_right = [6.774543285369873047e+00, -3.022140121459960938e+01, -3.003315448760986328e+00, -1.910096853971481323e-01, 1.507585942745208740e-01, 8.008892536163330078e-01, 5.471411943435668945e-01]

# Séparation des composantes
translation_left = vector_left[:3]
quaternion_left = vector_left[3:]

# Conversion
homogeneous_matrix_left = create_homogeneous_matrix(translation_left, quaternion_left)
print(homogeneous_matrix_left)

# Séparation des composantes
translation_right = vector_right[:3]
quaternion_right = vector_right[3:]

# Conversion
homogeneous_matrix_right = create_homogeneous_matrix(translation_right, quaternion_right)
print(homogeneous_matrix_right)
