import numpy as np

## TODO: pasar a notebook

# Ejercicio 1
p = "inf" # 0,1,2,3,4,5,...,"inf"

x = np.random.random((3,3))
print(x)

print("Ejercicio 1")

if p == 0:
    p_norms = np.sum(x > 0)
elif p == "inf":
    p_norms = np.max(x, axis=1)
else:
    p_norms = np.sum(np.abs(x)**p, axis=1)**(1/p)

print(p_norms)


# Ejercicio 2

print("Ejercicio 2")
indexes = np.argsort(p_norms * -1) # El -1 hace que los ordene de mayor a menor
print(indexes)
print(x[indexes])

# Ejercicio 3

class IdxManager:
    def __init__(self, ids):
        self.idx2id = ids
        self.id2idx = []

        # remove duplicated values in ids
        indexes = np.unique(ids, return_index=True)[1]
        ids = np.array([ids[index] for index in sorted(indexes)])

        self.id2idx = np.full((np.max(ids) + 1),-1)
        self.id2idx[ids] = np.arange(ids.size)

    def get_users_idx(self, user_id):
        try:
            return self.id2idx[user_id]
        except IndexError:
            return -1

    def get_users_id(self, idx):
        try:
            return self.idx2id[idx]
        except IndexError:
            return -1

idxManager = IdxManager([15, 12, 14, 10, 1, 2, 1])

print(idxManager.get_users_idx(15))
print(idxManager.get_users_idx(12))
print(idxManager.get_users_idx(3))

print(idxManager.get_users_id(0))
print(idxManager.get_users_id(4))

# Ejercicio 4

truth       = np.array([1,1,0,1,1,1,0,0,0,1])
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
print("Accuracy: " + str(accuracy))

# Ejercicio 5

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

# Ejercicios 6, 7 y 8 (los hacemos en la clase que viene )

# Ejercicio 9

# Crear clase metrica base, template sobre la cual agregamos las otras e implementar metodo call en cada una. kwargs ?? (intentar hacer)

# Ejercicio 10 (hacer)

# No usar panda, usar numpy y pickle, cargar CSV usar singleton