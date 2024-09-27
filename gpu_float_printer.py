
import gdb.printing
import numpy as np

from PIL import Image

A = slice(None)

class TensorFloatPrinter:
    """Pretty printer for CUDA Tensor-like objects."""

    def __init__(self, val):
        self.val = val

    def to_string(self, indices_spec=None):

        if indices_spec is None:
            # Accéder au membre __b_N2at31GenericPackedTensorAccessorBaseIfLm3ENS_17RestrictPtrTraitsEiEE
            #base = self.val['__b_N2at31GenericPackedTensorAccessorBaseIfLm3ENS_17RestrictPtrTraitsEiEE']

            base = self.val['__b_N2at31GenericPackedTensorAccessorBaseIfLm4ENS_17RestrictPtrTraitsEiEE']
            
            # Récupération des informations de l'objet
            data_ptr = base['data_']
            sizes = base['sizes_']
            strides = base['strides_']

            print("data_ptr : ", data_ptr)
            print("sizes : ", sizes)
            print("strides : ", strides)

            # Extraire les dimensions du tenseur en convertissant les gdb.Value en entiers
            size_length = sizes.type.sizeof // sizes[0].type.sizeof
            sizes_list = [int(sizes[i].cast(gdb.lookup_type('int'))) for i in range(size_length)]
     
            print(f"sizes_list : {sizes_list}")

            data_address = int(data_ptr)
            print("data_address : ", data_address)

            # Calculer le nombre total d'éléments dans le tenseur
            num_elements = np.prod(sizes_list)
            print("num_elements : ", num_elements)

            # Initialiser un tableau NumPy vide avec les bonnes dimensions
            tensor = np.zeros(sizes_list, dtype=float)


            # Configurer GDB pour afficher tous les éléments
            gdb.execute('set print elements 0', to_string=True)


# Fonction récursive pour remplir le tenseur
            def fill_tensor(indices, offset):
                if len(indices) == len(sizes_list) - 1:
                    # Nous sommes au dernier niveau, il faut récupérer une ligne
                    size_at_last_dim = sizes_list[-1]
                    line_address = f'({data_address} + {offset})'
                    expr = f'*(@global float[{size_at_last_dim}]*) {line_address}'
                    result = gdb.execute(f'print {expr}', to_string=True)

                    # Extraire les valeurs entre les accolades
                    result = result[result.index('{')+1 : result.index('}')]
                    
                    # Gestion du format abrégé "valeur <repeats N times>"
                    elements = []
                    for x in result.split(','):
                        x = x.strip()  # Supprimer les espaces
                        if '<repeats' in x:
                            # Gérer les répétitions
                            value, repeat_info = x.split(' <repeats ')
                            repeat_count = int(repeat_info.split()[0])
                            elements.extend([float(value)] * repeat_count)
                        else:
                            elements.append(float(x))
                    
                    # Remplir la dernière dimension du tenseur
                    tensor[tuple(indices)] = elements
                else:
                    # Parcourir les indices récursivement pour chaque dimension
                    stride = strides[len(indices)] * 4  # Taille en octets du float (ajuster si nécessaire)
                    for i in range(sizes_list[len(indices)]):
                        fill_tensor(indices + [i], offset + i * stride)

            # Lancer le remplissage du tenseur avec une fonction récursive
            fill_tensor([], 0)

            # Afficher ou utiliser le tenseur
            print("Tensor \n", tensor)

            return "Voici un Tensor Kernel"
        else:
            #base = self.val['__b_N2at31GenericPackedTensorAccessorBaseIfLm3ENS_17RestrictPtrTraitsEiEE']

            base = self.val['__b_N2at31GenericPackedTensorAccessorBaseIfLm4ENS_17RestrictPtrTraitsEiEE']
            
            # Récupération des informations de l'objet
            data_ptr = base['data_']
            sizes = base['sizes_']
            strides = base['strides_']

            print("data_ptr : ", data_ptr)
            print("sizes : ", sizes)
            print("strides : ", strides)

            # Extraire les dimensions du tensor
            size_length = sizes.type.sizeof // sizes[0].type.sizeof
            sizes_list = [int(sizes[i].cast(gdb.lookup_type('int'))) for i in range(size_length)]

            # Si aucun indice n'est fourni, afficher tout le tenseur
            if indices_spec is None:
                indices_spec = [slice(None)] * size_length

            # Vérification des indices pour les dimensions
            if len(indices_spec) != size_length:
                raise ValueError(f"Le nombre de dimensions spécifiées ({len(indices_spec)}) ne correspond pas à la taille du tensor ({size_length})")

            # Obtenir l'adresse de début des données
            data_address = int(data_ptr)

            # Trouver la taille de la matrice 2D résultante
            result_shape = [sizes_list[i] for i, idx in enumerate(indices_spec) if isinstance(idx, slice)]
            if len(result_shape) != 2:
                raise ValueError(f"La sélection doit aboutir à une matrice 2D, mais elle donne une forme {result_shape}")

            # Initialiser une matrice vide
            matrix = np.zeros(result_shape, dtype=float)

            # GDB: Afficher tous les éléments
            gdb.execute('set print elements 0', to_string=True)

            # Fonction pour remplir la matrice 2D
            def fill_matrix(offset, row_stride, col_stride):
                for i in range(result_shape[0]):  # Parcours des lignes
                    for j in range(result_shape[1]):  # Parcours des colonnes
                        # Calculer l'adresse de chaque élément dans le tensor
                        element_address = f'({data_address} + {offset} + {i * row_stride} + {j * col_stride})'
                        expr = f'*(@global float*) {element_address}'
                        value = gdb.execute(f'print {expr}', to_string=True)
                        value = float(value.split('=')[-1].strip())
                        matrix[i, j] = value

            # Calcul de l'offset initial et des strides pour les dimensions sélectionnées
            offset = 0
            row_stride = col_stride = 0
            for dim, idx in enumerate(indices_spec):
                if isinstance(idx, slice):
                    if row_stride == 0:
                        row_stride = strides[dim] * 4  # Taille en octets d'un float
                    else:
                        col_stride = strides[dim] * 4
                else:
                    offset += strides[dim] * int(idx) * 4  # Mise à jour de l'offset pour les dimensions figées

            # Remplir la matrice 2D
            fill_matrix(offset, row_stride, col_stride)
            # Trouver les valeurs minimales et maximales de la matrice
            min_val = np.min(matrix)
            max_val = np.max(matrix)
            
            # Normaliser la matrice pour être dans la plage [0, 1]
            matrix_normalized = (matrix - min_val) / (max_val - min_val)
            
            # Mettre à l'échelle les valeurs à la plage [0, 255]
            matrix_scaled = matrix_normalized * 255
            matrix_scaled = matrix_scaled.astype(np.uint8)
            # Redimensionner l'image pour qu'elle soit plus grande

            image = Image.fromarray(matrix_scaled, mode='L')  # Mode 'L' pour une image en niveaux de gris
            scale_factor = 10
            new_size = (image.width * scale_factor, image.height * scale_factor)
            image_resized = image.resize(new_size, Image.NEAREST)  # Utiliser un mode de redimensionnement approprié


            image_resized.show()

            return matrix


# Fonction pour appeler la méthode avec les indices donnés ou sans
def print_args_float(tensor_name, indices_spec=None):
    obj = gdb.parse_and_eval(tensor_name)
    printer = TensorFloatPrinter(obj)
    matrix = printer.to_string(indices_spec)
    print("Tensor : \n", matrix)


def build_pretty_printer():
    pp = gdb.printing.RegexpCollectionPrettyPrinter("cuda_pretty_printers_float")
    # Adapter le pattern pour le type complet manglé
    pp.add_printer('TensorFloat', '^_ZN2at27GenericPackedTensorAccessorIfLm4ENS_17RestrictPtrTraitsEiEE', TensorFloatPrinter)
    return pp

# Enregistrer le pretty printer dans GDB
gdb.printing.register_pretty_printer(gdb.current_objfile(), build_pretty_printer())


