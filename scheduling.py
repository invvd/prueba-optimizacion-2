"""
Modelo de scheduling para la carga de camiones en el deposito
Traduccion a Python usando PuLP (CBC solver)
"""
from pulp import *
import csv

# ========================
# DATOS
# ========================

OPERACIONES = ['C1_T1', 'C1_T2', 'C2_T1', 'C2_T2']

duracion = {
    'C1_T1': 15.0,
    'C1_T2': 17.5,
    'C2_T1': 15.0,
    'C2_T2': 12.0,
}

camion_map = {
    'C1_T1': 'C1',
    'C1_T2': 'C1',
    'C2_T1': 'C2',
    'C2_T2': 'C2',
}

bahia_map = {
    'C1_T1': 'R',
    'C1_T2': 'D',
    'C2_T1': 'D',
    'C2_T2': 'R',
}

M = sum(duracion[op] for op in OPERACIONES)

# Pares de operaciones que comparten recurso (i,j y j,i por separado)
pares_ordenados = []
for i in OPERACIONES:
    for j in OPERACIONES:
        if i != j and (camion_map[i] == camion_map[j] or bahia_map[i] == bahia_map[j]):
            pares_ordenados.append((i, j))

# Conjunto de pares no ordenados (solo i < j lexicografico)
pares_no_ordenados = []
for i in OPERACIONES:
    for j in OPERACIONES:
        if i < j and (camion_map[i] == camion_map[j] or bahia_map[i] == bahia_map[j]):
            pares_no_ordenados.append((i, j))

# ========================
# MODELO
# ========================

prob = LpProblem("Scheduling", LpMinimize)

# Variables
inicio = LpVariable.dicts("inicio", OPERACIONES, lowBound=0)
Cmax = LpVariable("Cmax", lowBound=0)
y = LpVariable.dicts("y", pares_ordenados, cat='Binary')

# Objetivo
prob += Cmax

# Orden: para cada par en conflicto, uno precede al otro
for (i, j) in pares_no_ordenados:
    prob += y[(i, j)] + y[(j, i)] == 1

# Precedencia: si i precede a j, j no puede empezar hasta que i termine
for (i, j) in pares_ordenados:
    prob += inicio[j] >= inicio[i] + duracion[i] - M * (1 - y[(i, j)])

# Makespan es el maximo de los tiempos de termino
for i in OPERACIONES:
    prob += Cmax >= inicio[i] + duracion[i]

# ========================
# SOLVE
# ========================

print("Resolviendo modelo de scheduling...")
prob.solve(PULP_CBC_CMD(msg=True))
print(f"Status: {LpStatus[prob.status]}")
print(f"Makespan: {value(Cmax):.2f} minutos")

# ========================
# EXPORTAR RESULTADOS CSV
# ========================

# Secuencia de operaciones
with open('scheduling.csv', 'w', newline='') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(['operacion', 'camion', 'bahia', 'inicio', 'fin', 'duracion'])
    for i in OPERACIONES:
        ini = value(inicio[i])
        fin = ini + duracion[i]
        writer.writerow([i, camion_map[i], bahia_map[i],
                         round(ini, 2), round(fin, 2), round(duracion[i], 2)])

# Orden entre operaciones en conflicto
with open('orden_scheduling.csv', 'w', newline='') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(['operacion_i', 'operacion_j', 'precede'])
    for (i, j) in pares_ordenados:
        if value(y[(i, j)]) > 0.5:
            writer.writerow([i, j, 'si'])

print("Archivos CSV generados: scheduling.csv, orden_scheduling.csv")
