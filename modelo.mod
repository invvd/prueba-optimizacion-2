# Modelo MILP determinista para la distribucion de combustibles

# Conjuntos del problema
set NODOS;                   # D (deposito) y estaciones 1..4
set ESTACIONES within NODOS; # solo las estaciones, sin el deposito
set CAMIONES;                # C1 y C2
set COMPARTIMENTOS;          # T1 y T2, los tanques de cada camion
set PRODUCTOS;               # R (Regular) y D (Diesel)

# Parametros
param demanda{ESTACIONES, PRODUCTOS} >= 0;   # litros que pide cada estacion
param capacidad{CAMIONES, COMPARTIMENTOS} >= 0;  # litros maximos por tanque
param distancia{NODOS, NODOS} >= 0;   # km entre nodos

param velocidad >= 0;        # km/h, igual para ambos camiones
param t_servicio >= 0;       # minutos que demora atender cada estacion

param ventana_inicio{ESTACIONES} >= 0;  # minutos desde las 00:00
param ventana_fin{ESTACIONES} >= 0;

param c_dist >= 0;           # costo por kilometro recorrido
param c_short >= 0;          # penalizacion por litro no entregado
param c_fijo{CAMIONES} >= 0; # costo fijo de usar cada camion

param Delta >= 0;            # maxima diferencia permitida entre fill ratios
param h_salida{CAMIONES} >= 0;  # hora de salida en minutos desde las 00:00

# El tiempo de viaje se calcula de la distancia y la velocidad
param t_viaje{i in NODOS, j in NODOS} := (distancia[i,j] / velocidad) * 60;

# Constantes grandes para relajar restricciones con big-M
param M_carga := 20000;
param M_tiempo := 1500;

# Variables de decision

var x{CAMIONES, NODOS, NODOS} binary;   # 1 si el camion k recorre el arco (i,j)
var y{CAMIONES, COMPARTIMENTOS, PRODUCTOS} binary;  # 1 si el tanque t del camion k lleva el producto p
var w{CAMIONES, ESTACIONES, PRODUCTOS} >= 0;  # litros entregados
var s{ESTACIONES, PRODUCTOS} >= 0;   # litros no entregados
var t{CAMIONES, ESTACIONES} >= 0;    # minuto de llegada a la estacion
var f{CAMIONES, COMPARTIMENTOS, NODOS, NODOS} >= 0;  # litros en el tanque durante el tramo (i,j)
var u{CAMIONES} binary;   # indica si el camion se usa
var z{CAMIONES, COMPARTIMENTOS, ESTACIONES, PRODUCTOS} >= 0;  # litros entregados desde un tanque especifico

# Funcion objetivo: minimizar el costo total

minimize Costo_Total:
    c_dist * sum{k in CAMIONES, i in NODOS, j in NODOS} distancia[i,j] * x[k,i,j]
  + sum{k in CAMIONES} c_fijo[k] * u[k]
  + c_short * sum{j in ESTACIONES, p in PRODUCTOS} s[j,p];

# Restriccion 1: conservacion de flujo en las rutas

subject to Sin_Bucles{k in CAMIONES, i in NODOS}:
    x[k,i,i] = 0;

subject to Sale_Deposito{k in CAMIONES}:
    sum{j in NODOS} x[k,'D',j] = u[k];

subject to Regresa_Deposito{k in CAMIONES}:
    sum{i in NODOS} x[k,i,'D'] = u[k];

subject to Flujo_Estacion{k in CAMIONES, j in ESTACIONES}:
    sum{i in NODOS} x[k,i,j] = sum{m in NODOS} x[k,j,m];

subject to Visita_Unica{k in CAMIONES, j in ESTACIONES}:
    sum{i in NODOS} x[k,i,j] <= 1;

# Restriccion 2: satisfaccion de demanda

subject to Satisfaccion_Demanda{j in ESTACIONES, p in PRODUCTOS}:
    sum{k in CAMIONES} w[k,j,p] + s[j,p] = demanda[j,p];

# Restriccion 3: cada tanque lleva un solo tipo de combustible

subject to Un_Producto_Por_Tanque{k in CAMIONES, t in COMPARTIMENTOS}:
    sum{p in PRODUCTOS} y[k,t,p] = 1;

# Restriccion 4: capacidad de cada tanque

subject to Capacidad_Tanque{k in CAMIONES, t in COMPARTIMENTOS, i in NODOS, j in NODOS}:
    f[k,t,i,j] <= capacidad[k,t] * x[k,i,j];

# Restriccion 5: relacion entre la carga en transito y las entregas

subject to Entrega_Total{k in CAMIONES, j in ESTACIONES, p in PRODUCTOS}:
    w[k,j,p] = sum{t in COMPARTIMENTOS} z[k,t,j,p];

subject to Compatibilidad_Entrega{k in CAMIONES, t in COMPARTIMENTOS, j in ESTACIONES, p in PRODUCTOS}:
    z[k,t,j,p] <= M_carga * y[k,t,p];

subject to Balance_Tanque_Estacion{k in CAMIONES, t in COMPARTIMENTOS, j in ESTACIONES}:
    sum{i in NODOS} f[k,t,i,j] - sum{p in PRODUCTOS} z[k,t,j,p] = sum{m in NODOS} f[k,t,j,m];

subject to Balance_Tanque_Global{k in CAMIONES, t in COMPARTIMENTOS}:
    sum{m in NODOS} f[k,t,'D',m] - sum{i in NODOS} f[k,t,i,'D'] = sum{j in ESTACIONES, p in PRODUCTOS} z[k,t,j,p];

# Restriccion 6: estabilidad de carga

param prod_cap{k in CAMIONES} := capacidad[k,'T1'] * capacidad[k,'T2'];

subject to Estabilidad_Carga_A{k in CAMIONES, i in NODOS, j in NODOS}:
    f[k,'T1',i,j] * capacidad[k,'T2'] - f[k,'T2',i,j] * capacidad[k,'T1'] 
    <= Delta * prod_cap[k];

subject to Estabilidad_Carga_B{k in CAMIONES, i in NODOS, j in NODOS}:
    f[k,'T2',i,j] * capacidad[k,'T1'] - f[k,'T1',i,j] * capacidad[k,'T2'] 
    <= Delta * prod_cap[k];

# Restriccion 7: ventanas de tiempo

subject to Tiempo_Desde_Deposito{k in CAMIONES, j in ESTACIONES}:
    t[k,j] >= h_salida[k] + t_viaje['D',j] - M_tiempo * (1 - x[k,'D',j]);

subject to Tiempo_Entre_Estaciones{k in CAMIONES, i in ESTACIONES, j in ESTACIONES: i <> j}:
    t[k,j] >= t[k,i] + t_servicio + t_viaje[i,j] - M_tiempo * (1 - x[k,i,j]);

subject to Ventana_Inicio{k in CAMIONES, j in ESTACIONES}:
    t[k,j] >= ventana_inicio[j] - M_tiempo * (1 - sum{i in NODOS} x[k,i,j]);

subject to Ventana_Fin{k in CAMIONES, j in ESTACIONES}:
    t[k,j] <= ventana_fin[j] + M_tiempo * (1 - sum{i in NODOS} x[k,i,j]);

subject to Tiempo_Cero_Si_No_Visita{k in CAMIONES, j in ESTACIONES}:
    t[k,j] <= M_tiempo * sum{i in NODOS} x[k,i,j];
