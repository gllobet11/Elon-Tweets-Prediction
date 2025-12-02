import sys
import os

# Agrega el directorio raíz del proyecto al sys.path para que pytest
# pueda encontrar los módulos en 'src' y 'config'.
# Esto es una práctica estándar para estructuras de proyecto como esta.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
