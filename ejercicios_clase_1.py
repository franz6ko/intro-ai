import numpy as np

# Ejercicio 1
""" p = 2 # 1,2,3,4,5

x = np.random.random((3,3))
print(x)

print("Ejercicio 1")
p_norms = np.sum(np.abs(x)**p, axis=1)**(1/p)
print(p_norms)
# usar mascaras para la norma 0
# usar np.max para la norma infinito """

# Ejercicio 2

""" print("Ejercicio 2")
indexes = np.argsort(p_norms * -1) # El -1 hace que los ordene de mayor a menor
print(indexes)
print(x[indexes]) """

# Ejercicio 3

# masking, crear clases, hacer que todo sea vectorizado

""" class IdxManager:
    def __init__(self, name):
        self.users_id = [15, 12, 14, 10, 1, 2, 1]
    def get_users_id(self):
        # TODO
    def get_users_idx(self):
        # TODO """

# Ejercicio 4 (hacer)

""" truth       = np.array([1,1,0,1,1,1,0,0,0,1])
prediction  = np.array([1,1,1,1,0,0,1,1,0,0])

TP = np.sum(truth & prediction)
TN = np.sum((truth == False) & (prediction == False))
FN = np.sum((truth == True) & (prediction == False))
FP = np.sum((truth == False) & (prediction == True))

precision = TP / (TP + FP)
recall = TP / (TP + FN)
accuracy = (TP + TN) / (TP + TN + FP + FN)

print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("Accuracy: " + str(accuracy)) """

# Ejercicio 5

# normas y masking

T = True
F = False

q_id            = np.array([1,1,1,1,2,2,2,3,3,3,3,3,4,4,4,4])
predicted_rank  = np.array([0,1,2,3,0,1,2,0,1,2,3,4,0,1,2,3]) # Para qué me sirve ?
truth_relevance = np.array([T,F,T,F,T,T,T,F,F,F,F,F,T,F,F,T])

q_precision = np.zeros((4,1))

q_precision[0] = truth_relevance[q_id == 1].mean()
q_precision[1] = truth_relevance[q_id == 2].mean()
q_precision[2] = truth_relevance[q_id == 3].mean()
q_precision[3] = truth_relevance[q_id == 4].mean()

for i in range(4):
    print("Precision for id " + str(i) + ": " + str(q_precision[i]))

print("Average query precision: " + str(q_precision.mean()))

# Ejercicios 6, 7 y 8 para la clase que viene 

# Ejercicio 9

# Crear clase metrica base, template sobre la cual agregamos las otras e implementar metodo call en cada una. kwargs ?? (intentar hacer)

# Ejercicio 10 (hacer)

# No usar panda, usar numpy y pickle, cargar SCV usar singleton