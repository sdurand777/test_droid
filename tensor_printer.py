
import gdb
import gdb.printing

import numpy as np

from PIL import Image

A = slice(None)

class AtTensorPrinter:
    """Pretty-printer for at::GenericPackedTensorAccessorBase objects."""

    def __init__(self, val):
        self.val = val

    def to_string(self, tensor_name = None):
        # Accéder à la structure des tailles
        self.tensor_name = tensor_name
        #print("tensor : ", self.tensor_name)

        expr = 'print '+str(self.tensor_name)+'.sizes().Length'
        #print("expr : ", expr)
        result = gdb.execute(expr, to_string=True)
        # Étape 1 : Diviser la chaîne pour obtenir la valeur après le "="
        value_str = result.split('=')[-1].strip()  # Supprime les espaces autour

        # Étape 2 : Convertir la valeur en entier
        value = int(value_str)

        # Affichage pour vérifier
        #print("value : ", value)  # Cela affichera : 2

        dimensions = np.zeros(value, dtype=int)
        for i in range(value):
            expr_data = f'print {tensor_name}.sizes().Data[{i}]'
            result = gdb.execute(expr_data, to_string=True)
            value_str = result.split('=')[-1].strip()  # Supprime les espaces autour
            value = int(value_str)
            dimensions[i] = value

        print("dimensions : ", dimensions)
 

def print_tensor(tensor_name):
    obj = gdb.parse_and_eval(tensor_name)
    printer = AtTensorPrinter(obj)
    printer.to_string(tensor_name)

def build_pretty_printer():
    print("build pretty for at::Tensor")
    pp = gdb.printing.RegexpCollectionPrettyPrinter("torchprinter")
    pp.add_printer('AtTensorPrinter', '^at::TensorBase*$', AtTensorPrinter)
    return pp

gdb.printing.register_pretty_printer(gdb.current_objfile(), build_pretty_printer())
