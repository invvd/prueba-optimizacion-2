"""
Modelo estocastico de dos etapas para la distribucion de combustibles
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
ESCENARIOS = ['1', '2', '3']

prob_esc = {'1': 0.5, '2': 0.3, '3': 0.2}

demanda = {}
demanda[('1', 'R', '1')] = 3000
demanda[('1', 'D', '1')] = 2000
demanda[('2', 'R', '1')] = 4000
demanda[('2', 'D', '1')] = 1500
demanda[('3', 'R', '1')] = 2500
demanda[('3', 'D', '1')] = 3000
demanda[('4', 'R', '1')] = 1000
demanda[('4', 'D', '1')] = 2500

demanda[('1', 'R', '2')] = 4500
demanda[('1', 'D', '2')] = 2800
demanda[('2', 'R', '2')] = 5500
demanda[('2', 'D', '2')] = 2000
demanda[('3', 'R', '2')] = 3500
demanda[('3', 'D', '2')] = 3000
demanda[('4', 'R', '2')] = 1000
demanda[('4', 'D', '2')] = 2500

demanda[('1', 'R', '3')] = 2000
demanda[('1', 'D', '3')] = 1200
demanda[('2', 'R', '3')] = 3000
demanda[('2', 'D', '3')] = 1000
demanda[('3', 'R', '3')] = 1800
demanda[('3', 'D', '3')] = 3000
demanda[('4', 'R', '3')] = 1000
demanda[('4', 'D', '3')] = 2500

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

prob = LpProblem("Fuel_Stochastic", LpMinimize)

# --- Variables de primera etapa ---
x = LpVariable.dicts("x", [(k, i, j) for k in CAMIONES for i in NODOS for j in NODOS], cat='Binary')
y = LpVariable.dicts("y", [(k, t, p) for k in CAMIONES for t in COMPARTIMENTOS for p in PRODUCTOS], cat='Binary')
u = LpVariable.dicts("u", CAMIONES, cat='Binary')
L = LpVariable.dicts("L", [(k, t) for k in CAMIONES for t in COMPARTIMENTOS], lowBound=0)
t_arr = LpVariable.dicts("t", [(k, j) for k in CAMIONES for j in ESTACIONES], lowBound=0)

# --- Variables de segunda etapa (por escenario) ---
w = LpVariable.dicts("w", [(k, j, p, e) for k in CAMIONES for j in ESTACIONES for p in PRODUCTOS for e in ESCENARIOS], lowBound=0)
s = LpVariable.dicts("s", [(j, p, e) for j in ESTACIONES for p in PRODUCTOS for e in ESCENARIOS], lowBound=0)
f = LpVariable.dicts("f", [(k, tank, i, j, e) for k in CAMIONES for tank in COMPARTIMENTOS for i in NODOS for j in NODOS for e in ESCENARIOS], lowBound=0)
z = LpVariable.dicts("z", [(k, tank, j, p, e) for k in CAMIONES for tank in COMPARTIMENTOS for j in ESTACIONES for p in PRODUCTOS for e in ESCENARIOS], lowBound=0)

# --- Funcion objetivo ---
prob += (
    c_dist * lpSum(distancia[(i, j)] * x[(k, i, j)] for k in CAMIONES for i in NODOS for j in NODOS)
    + lpSum(c_fijo[k] * u[k] for k in CAMIONES)
    + c_short * lpSum(prob_esc[e] * s[(j, p, e)] for e in ESCENARIOS for j in ESTACIONES for p in PRODUCTOS)
)

# --- Restricciones de primera etapa ---

# Routing
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

# Un producto por tanque
for k in CAMIONES:
    for tank in COMPARTIMENTOS:
        prob += lpSum(y[(k, tank, p)] for p in PRODUCTOS) == 1

# Carga inicial no excede capacidad
for k in CAMIONES:
    for tank in COMPARTIMENTOS:
        prob += L[(k, tank)] <= capacidad[(k, tank)] * u[k]

# Ventanas de tiempo
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

# --- Restricciones de segunda etapa (por escenario) ---

for e in ESCENARIOS:

    # Satisfaccion de demanda
    for j in ESTACIONES:
        for p in PRODUCTOS:
            prob += lpSum(w[(k, j, p, e)] for k in CAMIONES) + s[(j, p, e)] == demanda[(j, p, e)]

    # Capacidad de tanque
    for k in CAMIONES:
        for tank in COMPARTIMENTOS:
            for i in NODOS:
                for j in NODOS:
                    prob += f[(k, tank, i, j, e)] <= capacidad[(k, tank)] * x[(k, i, j)]

    # Entrega total = suma de entregas por tanque
    for k in CAMIONES:
        for j in ESTACIONES:
            for p in PRODUCTOS:
                prob += w[(k, j, p, e)] == lpSum(z[(k, tank, j, p, e)] for tank in COMPARTIMENTOS)

    # Compatibilidad de entrega
    for k in CAMIONES:
        for tank in COMPARTIMENTOS:
            for j in ESTACIONES:
                for p in PRODUCTOS:
                    prob += z[(k, tank, j, p, e)] <= M_carga * y[(k, tank, p)]

    # Balance de tanque en estacion
    for k in CAMIONES:
        for tank in COMPARTIMENTOS:
            for j in ESTACIONES:
                prob += (
                    lpSum(f[(k, tank, i, j, e)] for i in NODOS)
                    - lpSum(z[(k, tank, j, p, e)] for p in PRODUCTOS)
                    == lpSum(f[(k, tank, j, m, e)] for m in NODOS)
                )

    # Carga inicial determina cuanto sale del deposito
    for k in CAMIONES:
        for tank in COMPARTIMENTOS:
            prob += lpSum(f[(k, tank, 'D', j, e)] for j in NODOS) == L[(k, tank)]

    # Balance global del tanque
    for k in CAMIONES:
        for tank in COMPARTIMENTOS:
            prob += (
                lpSum(f[(k, tank, 'D', m, e)] for m in NODOS)
                - lpSum(f[(k, tank, i, 'D', e)] for i in NODOS)
                == lpSum(z[(k, tank, j, p, e)] for j in ESTACIONES for p in PRODUCTOS)
            )

    # Estabilidad de carga
    for k in CAMIONES:
        for i in NODOS:
            for j in NODOS:
                prob += (
                    f[(k, 'T1', i, j, e)] * capacidad[(k, 'T2')]
                    - f[(k, 'T2', i, j, e)] * capacidad[(k, 'T1')]
                    <= Delta * prod_cap[k]
                )
                prob += (
                    f[(k, 'T2', i, j, e)] * capacidad[(k, 'T1')]
                    - f[(k, 'T1', i, j, e)] * capacidad[(k, 'T2')]
                    <= Delta * prod_cap[k]
                )

# ========================
# SOLVE
# ========================

print("Resolviendo modelo estocastico...")
prob.solve(PULP_CBC_CMD(msg=True))
print(f"Status: {LpStatus[prob.status]}")
print(f"Costo Total Esperado: {value(prob.objective):.2f}")

# ========================
# EXPORTAR RESULTADOS CSV
# ========================

# Rutas (primera etapa)
with open('estocastico_rutas.csv', 'w', newline='') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(['camion', 'origen', 'destino', 'usado'])
    for k in CAMIONES:
        for i in NODOS:
            for j in NODOS:
                if value(x[(k, i, j)]) > 0.5:
                    writer.writerow([k, i, j, 1])

# Asignacion tanque-producto (primera etapa)
with open('estocastico_asignacion.csv', 'w', newline='') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(['camion', 'tanque', 'producto'])
    for k in CAMIONES:
        for tank in COMPARTIMENTOS:
            for p in PRODUCTOS:
                if value(y[(k, tank, p)]) > 0.5:
                    writer.writerow([k, tank, p])

# Carga inicial (primera etapa)
with open('estocastico_carga.csv', 'w', newline='') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(['camion', 'tanque', 'carga_inicial_litros'])
    for k in CAMIONES:
        for tank in COMPARTIMENTOS:
            if value(L[(k, tank)]) > 0.01:
                writer.writerow([k, tank, round(value(L[(k, tank)]), 2)])

# Resultados por escenario
with open('estocastico_escenarios.csv', 'w', newline='') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(['escenario', 'estacion', 'producto', 'demanda', 'entregado', 'shortage'])
    for e in ESCENARIOS:
        for j in ESTACIONES:
            for p in PRODUCTOS:
                entregado = sum(value(w[(k, j, p, e)]) for k in CAMIONES)
                writer.writerow([e, j, p,
                                 round(demanda[(j, p, e)], 2),
                                 round(entregado, 2),
                                 round(value(s[(j, p, e)]), 2)])

# Costos
with open('estocastico_costo.csv', 'w', newline='') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(['componente', 'valor'])
    costo_dist = c_dist * sum(value(x[(k, i, j)]) * distancia[(i, j)]
                               for k in CAMIONES for i in NODOS for j in NODOS)
    costo_fijo_total = sum(c_fijo[k] * value(u[k]) for k in CAMIONES)
    writer.writerow(['distancia', round(costo_dist, 2)])
    writer.writerow(['fijo', round(costo_fijo_total, 2)])
    for e in ESCENARIOS:
        costo_short_esc = c_short * sum(value(s[(j, p, e)]) for j in ESTACIONES for p in PRODUCTOS)
        writer.writerow([f'shortage_esc{e}', round(costo_short_esc, 2)])
    costo_short_esp = c_short * sum(prob_esc[e] * value(s[(j, p, e)])
                                     for e in ESCENARIOS for j in ESTACIONES for p in PRODUCTOS)
    writer.writerow(['shortage_esperado', round(costo_short_esp, 2)])
    writer.writerow(['total', round(value(prob.objective), 2)])

print("\nArchivos CSV generados: estocastico_rutas.csv, estocastico_asignacion.csv, estocastico_carga.csv, estocastico_escenarios.csv, estocastico_costo.csv")
