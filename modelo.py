"""
Modelo MILP determinista para la distribucion de combustibles
Traduccion a Python usando PuLP (CBC solver)
"""
from pulp import *
import csv

# ========================
# DATOS
# ========================

NODOS = ['D', '1', '2', '3', '4']
ESTACIONES = ['1', '2', '3', '4']
CAMIONES = ['C1', 'C2']
COMPARTIMENTOS = ['T1', 'T2']
PRODUCTOS = ['R', 'D']

demanda = {
    ('1', 'R'): 3000, ('1', 'D'): 2000,
    ('2', 'R'): 4000, ('2', 'D'): 1500,
    ('3', 'R'): 2500, ('3', 'D'): 3000,
    ('4', 'R'): 1000, ('4', 'D'): 2500,
}

capacidad = {
    ('C1', 'T1'): 8000, ('C1', 'T2'): 7000,
    ('C2', 'T1'): 6000, ('C2', 'T2'): 9000,
}

distancia = {
    ('D', 'D'): 0, ('D', '1'): 20, ('D', '2'): 35, ('D', '3'): 15, ('D', '4'): 40,
    ('1', 'D'): 20, ('1', '1'): 0,  ('1', '2'): 10, ('1', '3'): 25, ('1', '4'): 30,
    ('2', 'D'): 35, ('2', '1'): 10, ('2', '2'): 0,  ('2', '3'): 20, ('2', '4'): 15,
    ('3', 'D'): 15, ('3', '1'): 25, ('3', '2'): 20, ('3', '3'): 0,  ('3', '4'): 30,
    ('4', 'D'): 40, ('4', '1'): 30, ('4', '2'): 15, ('4', '3'): 30, ('4', '4'): 0,
}

velocidad = 60
t_servicio = 30
c_dist = 2
c_short = 10
c_fijo = {'C1': 500, 'C2': 400}
Delta = 0.30
h_salida = {'C1': 300, 'C2': 330}

ventana_inicio = {'1': 360, '2': 420, '3': 480, '4': 360}
ventana_fin    = {'1': 600, '2': 720, '3': 840, '4': 660}

M_carga = 20000
M_tiempo = 1500

t_viaje = {}
for i in NODOS:
    for j in NODOS:
        t_viaje[(i, j)] = (distancia[(i, j)] / velocidad) * 60

prod_cap = {}
for k in CAMIONES:
    prod_cap[k] = capacidad[(k, 'T1')] * capacidad[(k, 'T2')]

# ========================
# MODELO
# ========================

prob = LpProblem("Fuel_Distribution", LpMinimize)

# --- Variables de decision ---
x = LpVariable.dicts("x", [(k, i, j) for k in CAMIONES for i in NODOS for j in NODOS], cat='Binary')
y = LpVariable.dicts("y", [(k, t, p) for k in CAMIONES for t in COMPARTIMENTOS for p in PRODUCTOS], cat='Binary')
w = LpVariable.dicts("w", [(k, j, p) for k in CAMIONES for j in ESTACIONES for p in PRODUCTOS], lowBound=0)
s = LpVariable.dicts("s", [(j, p) for j in ESTACIONES for p in PRODUCTOS], lowBound=0)
t_arr = LpVariable.dicts("t", [(k, j) for k in CAMIONES for j in ESTACIONES], lowBound=0)
f = LpVariable.dicts("f", [(k, tank, i, j) for k in CAMIONES for tank in COMPARTIMENTOS for i in NODOS for j in NODOS], lowBound=0)
u = LpVariable.dicts("u", CAMIONES, cat='Binary')
z = LpVariable.dicts("z", [(k, tank, j, p) for k in CAMIONES for tank in COMPARTIMENTOS for j in ESTACIONES for p in PRODUCTOS], lowBound=0)

# --- Funcion objetivo ---
prob += (
    c_dist * lpSum(distancia[(i, j)] * x[(k, i, j)] for k in CAMIONES for i in NODOS for j in NODOS)
    + lpSum(c_fijo[k] * u[k] for k in CAMIONES)
    + c_short * lpSum(s[(j, p)] for j in ESTACIONES for p in PRODUCTOS)
)

# --- Restriccion 1: conservacion de flujo en las rutas ---
for k in CAMIONES:
    for i in NODOS:
        prob += x[(k, i, i)] == 0

for k in CAMIONES:
    prob += lpSum(x[(k, 'D', j)] for j in NODOS) == u[k]

for k in CAMIONES:
    prob += lpSum(x[(k, i, 'D')] for i in NODOS) == u[k]

for k in CAMIONES:
    for j in ESTACIONES:
        prob += lpSum(x[(k, i, j)] for i in NODOS) == lpSum(x[(k, j, m)] for m in NODOS)

for k in CAMIONES:
    for j in ESTACIONES:
        prob += lpSum(x[(k, i, j)] for i in NODOS) <= 1

# --- Restriccion 2: satisfaccion de demanda ---
for j in ESTACIONES:
    for p in PRODUCTOS:
        prob += lpSum(w[(k, j, p)] for k in CAMIONES) + s[(j, p)] == demanda[(j, p)]

# --- Restriccion 3: cada tanque lleva un solo tipo de combustible ---
for k in CAMIONES:
    for tank in COMPARTIMENTOS:
        prob += lpSum(y[(k, tank, p)] for p in PRODUCTOS) == 1

# --- Restriccion 4: capacidad de cada tanque ---
for k in CAMIONES:
    for tank in COMPARTIMENTOS:
        for i in NODOS:
            for j in NODOS:
                prob += f[(k, tank, i, j)] <= capacidad[(k, tank)] * x[(k, i, j)]

# --- Restriccion 5: relacion entre carga en transito y entregas ---
for k in CAMIONES:
    for j in ESTACIONES:
        for p in PRODUCTOS:
            prob += w[(k, j, p)] == lpSum(z[(k, tank, j, p)] for tank in COMPARTIMENTOS)

for k in CAMIONES:
    for tank in COMPARTIMENTOS:
        for j in ESTACIONES:
            for p in PRODUCTOS:
                prob += z[(k, tank, j, p)] <= M_carga * y[(k, tank, p)]

for k in CAMIONES:
    for tank in COMPARTIMENTOS:
        for j in ESTACIONES:
            prob += (
                lpSum(f[(k, tank, i, j)] for i in NODOS)
                - lpSum(z[(k, tank, j, p)] for p in PRODUCTOS)
                == lpSum(f[(k, tank, j, m)] for m in NODOS)
            )

for k in CAMIONES:
    for tank in COMPARTIMENTOS:
        prob += (
            lpSum(f[(k, tank, 'D', m)] for m in NODOS)
            - lpSum(f[(k, tank, i, 'D')] for i in NODOS)
            == lpSum(z[(k, tank, j, p)] for j in ESTACIONES for p in PRODUCTOS)
        )

# --- Restriccion 6: estabilidad de carga ---
for k in CAMIONES:
    for i in NODOS:
        for j in NODOS:
            prob += (
                f[(k, 'T1', i, j)] * capacidad[(k, 'T2')]
                - f[(k, 'T2', i, j)] * capacidad[(k, 'T1')]
                <= Delta * prod_cap[k]
            )
            prob += (
                f[(k, 'T2', i, j)] * capacidad[(k, 'T1')]
                - f[(k, 'T1', i, j)] * capacidad[(k, 'T2')]
                <= Delta * prod_cap[k]
            )

# --- Restriccion 7: ventanas de tiempo ---
for k in CAMIONES:
    for j in ESTACIONES:
        prob += t_arr[(k, j)] >= h_salida[k] + t_viaje[('D', j)] - M_tiempo * (1 - x[(k, 'D', j)])

for k in CAMIONES:
    for i in ESTACIONES:
        for j in ESTACIONES:
            if i != j:
                prob += t_arr[(k, j)] >= t_arr[(k, i)] + t_servicio + t_viaje[(i, j)] - M_tiempo * (1 - x[(k, i, j)])

for k in CAMIONES:
    for j in ESTACIONES:
        visitada = lpSum(x[(k, i, j)] for i in NODOS)
        prob += t_arr[(k, j)] >= ventana_inicio[j] - M_tiempo * (1 - visitada)
        prob += t_arr[(k, j)] <= ventana_fin[j] + M_tiempo * (1 - visitada)
        prob += t_arr[(k, j)] <= M_tiempo * visitada

# ========================
# SOLVE
# ========================

print("Resolviendo modelo determinista...")
prob.solve(PULP_CBC_CMD(msg=True))
print(f"Status: {LpStatus[prob.status]}")
print(f"Costo Total: {value(prob.objective):.2f}")

# ========================
# EXPORTAR RESULTADOS CSV
# ========================

# Rutas
with open('rutas.csv', 'w', newline='') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(['camion', 'origen', 'destino', 'usado'])
    for k in CAMIONES:
        for i in NODOS:
            for j in NODOS:
                if value(x[(k, i, j)]) > 0.5:
                    writer.writerow([k, i, j, 1])

# Entregas
with open('entregas.csv', 'w', newline='') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(['camion', 'estacion', 'producto', 'litros'])
    for k in CAMIONES:
        for j in ESTACIONES:
            for p in PRODUCTOS:
                if value(w[(k, j, p)]) > 0.01:
                    writer.writerow([k, j, p, round(value(w[(k, j, p)]), 2)])

# Shortage
with open('shortage.csv', 'w', newline='') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(['estacion', 'producto', 'litros_faltantes'])
    for j in ESTACIONES:
        for p in PRODUCTOS:
            if value(s[(j, p)]) > 0.01:
                writer.writerow([j, p, round(value(s[(j, p)]), 2)])

# Estabilidad de carga
with open('estabilidad.csv', 'w', newline='') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(['camion', 'tramo_i', 'tramo_j', 'fill_T1', 'fill_T2', 'diferencia', 'cumple'])
    for k in CAMIONES:
        for i in NODOS:
            for j in NODOS:
                if value(x[(k, i, j)]) > 0.5:
                    fill_T1 = value(f[(k, 'T1', i, j)]) / capacidad[(k, 'T1')]
                    fill_T2 = value(f[(k, 'T2', i, j)]) / capacidad[(k, 'T2')]
                    diff = abs(fill_T1 - fill_T2)
                    cumple = 1 if diff <= Delta + 1e-9 else 0
                    writer.writerow([k, i, j, round(fill_T1, 4), round(fill_T2, 4), round(diff, 4), cumple])

# Tiempos de llegada
with open('tiempos.csv', 'w', newline='') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(['camion', 'estacion', 'llegada_min', 'llegada_hora', 'ventana_inicio', 'ventana_fin', 'cumple'])
    for k in CAMIONES:
        for j in ESTACIONES:
            if value(lpSum(x[(k, i, j)] for i in NODOS)) > 0.5:
                llegada = int(round(value(t_arr[(k, j)])))
                horas = llegada // 60
                mins = llegada % 60
                cumple = 1 if ventana_inicio[j] <= llegada <= ventana_fin[j] else 0
                writer.writerow([k, j, llegada, f"{horas:02d}:{mins:02d}",
                                 ventana_inicio[j], ventana_fin[j], cumple])

# Desglose de costos
with open('costo.csv', 'w', newline='') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(['componente', 'valor'])
    costo_dist = c_dist * sum(value(x[(k, i, j)]) * distancia[(i, j)]
                               for k in CAMIONES for i in NODOS for j in NODOS)
    costo_fijo_total = sum(c_fijo[k] * value(u[k]) for k in CAMIONES)
    costo_short_total = c_short * sum(value(s[(j, p)]) for j in ESTACIONES for p in PRODUCTOS)
    writer.writerow(['distancia', round(costo_dist, 2)])
    writer.writerow(['fijo', round(costo_fijo_total, 2)])
    writer.writerow(['shortage', round(costo_short_total, 2)])
    writer.writerow(['total', round(value(prob.objective), 2)])

# Asignacion tanque-producto
with open('asignacion.csv', 'w', newline='') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(['camion', 'tanque', 'producto'])
    for k in CAMIONES:
        for tank in COMPARTIMENTOS:
            for p in PRODUCTOS:
                if value(y[(k, tank, p)]) > 0.5:
                    writer.writerow([k, tank, p])

print("\nArchivos CSV generados: rutas.csv, entregas.csv, shortage.csv, estabilidad.csv, tiempos.csv, costo.csv, asignacion.csv")
