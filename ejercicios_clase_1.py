import numpy as np

# Ejercicio 1
p = 2 # 1,2,3,4,5

x = np.random.random((3,3))
print(x)

print("Ejercicio 1")
p_norms = np.sum(np.abs(x)**p, axis=1)**(1/p)
print(p_norms)
# usar mascaras para la norma 0
# usar np.max para la norma infinito

# Ejercicio 2

print("Ejercicio 2")
indexes = np.argsort(p_norms * -1) # El -1 hace que los ordene de mayor a menor
print(indexes)
print(x[indexes])

# Ejercicio 3

# masking, crear clases, hacer que todo sea vectorizado

# Ejercicio 4 (hacer)

# masking, metricas

# Ejercicio 5

# normas y masking

# Ejercicios 6, 7 y 8 para la clase que viene 

# Ejercicio 9

# Crear clase metrica base, template sobre la cual agregamos las otras e implementar metodo call en cada una. kwargs ?? (intentar hacer)

# Ejercicio 10 (hacer)

# No usar panda, usar numpy y pickle, cargar SCV usar singleton