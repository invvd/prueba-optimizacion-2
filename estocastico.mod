# Modelo estocastico de dos etapas para la distribucion de combustibles
# La demanda es incierta y se modela con tres escenarios

# Conjuntos
set NODOS;
set ESTACIONES within NODOS;
set CAMIONES;
set COMPARTIMENTOS;
set PRODUCTOS;
set ESCENARIOS;   # 1=Normal, 2=Alta, 3=Baja

# Parametros
param prob{ESCENARIOS} >= 0;
param demanda{ESTACIONES, PRODUCTOS, ESCENARIOS} >= 0;
param capacidad{CAMIONES, COMPARTIMENTOS} >= 0;
param distancia{NODOS, NODOS} >= 0;
param velocidad >= 0;
param t_servicio >= 0;
param ventana_inicio{ESTACIONES} >= 0;
param ventana_fin{ESTACIONES} >= 0;
param c_dist >= 0;
param c_short >= 0;
param c_fijo{CAMIONES} >= 0;
param Delta >= 0;
param h_salida{CAMIONES} >= 0;

param t_viaje{i in NODOS, j in NODOS} := (distancia[i,j] / velocidad) * 60;

param M_carga := 20000;
param M_tiempo := 1500;

# Variables de primera etapa (se deciden antes de conocer la demanda)

var x{CAMIONES, NODOS, NODOS} binary;          # ruta
var y{CAMIONES, COMPARTIMENTOS, PRODUCTOS} binary;  # asignacion de producto a tanque
var u{CAMIONES} binary;                        # uso del camion
var L{CAMIONES, COMPARTIMENTOS} >= 0;          # carga inicial en el deposito
var t{CAMIONES, ESTACIONES} >= 0;              # tiempo de llegada

# Variables de segunda etapa (dependen del escenario)

var w{CAMIONES, ESTACIONES, PRODUCTOS, ESCENARIOS} >= 0;        # entrega total
var s{ESTACIONES, PRODUCTOS, ESCENARIOS} >= 0;                  # shortage
var f{CAMIONES, COMPARTIMENTOS, NODOS, NODOS, ESCENARIOS} >= 0; # carga en transito
var z{CAMIONES, COMPARTIMENTOS, ESTACIONES, PRODUCTOS, ESCENARIOS} >= 0;  # entrega por tanque

# Funcion objetivo: costo fijo + distancia + valor esperado del shortage

minimize Costo_Total_Estocastico:
    c_dist * sum{k in CAMIONES, i in NODOS, j in NODOS} distancia[i,j] * x[k,i,j]
  + sum{k in CAMIONES} c_fijo[k] * u[k]
  + c_short * sum{esc in ESCENARIOS} prob[esc] * sum{j in ESTACIONES, p in PRODUCTOS} s[j,p,esc];

# Restricciones de primera etapa (no dependen del escenario)

# Routing
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

# Compatibilidad tanque-producto
subject to Un_Producto_Por_Tanque{k in CAMIONES, t in COMPARTIMENTOS}:
    sum{p in PRODUCTOS} y[k,t,p] = 1;

# La carga inicial no puede exceder la capacidad
subject to Carga_Inicial_Capacidad{k in CAMIONES, t in COMPARTIMENTOS}:
    L[k,t] <= capacidad[k,t] * u[k];

# Ventanas de tiempo
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

# Restricciones de segunda etapa (dependen del escenario)

# Satisfaccion de demanda
subject to Satisfaccion_Demanda{j in ESTACIONES, p in PRODUCTOS, esc in ESCENARIOS}:
    sum{k in CAMIONES} w[k,j,p,esc] + s[j,p,esc] = demanda[j,p,esc];

# Capacidad de tanque
subject to Capacidad_Tanque{k in CAMIONES, t in COMPARTIMENTOS, i in NODOS, j in NODOS, esc in ESCENARIOS}:
    f[k,t,i,j,esc] <= capacidad[k,t] * x[k,i,j];

# Relacion entre entrega y carga en transito
subject to Entrega_Total{k in CAMIONES, j in ESTACIONES, p in PRODUCTOS, esc in ESCENARIOS}:
    w[k,j,p,esc] = sum{t in COMPARTIMENTOS} z[k,t,j,p,esc];

subject to Compatibilidad_Entrega{k in CAMIONES, t in COMPARTIMENTOS, j in ESTACIONES, p in PRODUCTOS, esc in ESCENARIOS}:
    z[k,t,j,p,esc] <= M_carga * y[k,t,p];

subject to Balance_Tanque_Estacion{k in CAMIONES, t in COMPARTIMENTOS, j in ESTACIONES, esc in ESCENARIOS}:
    sum{i in NODOS} f[k,t,i,j,esc] - sum{p in PRODUCTOS} z[k,t,j,p,esc] = sum{m in NODOS} f[k,t,j,m,esc];

# La carga inicial determina cuanto sale del deposito
subject to Carga_Inicial{k in CAMIONES, t in COMPARTIMENTOS, esc in ESCENARIOS}:
    sum{j in NODOS} f[k,t,'D',j,esc] = L[k,t];

# Balance global del tanque
subject to Balance_Tanque_Global{k in CAMIONES, t in COMPARTIMENTOS, esc in ESCENARIOS}:
    sum{m in NODOS} f[k,t,'D',m,esc] - sum{i in NODOS} f[k,t,i,'D',esc] 
    = sum{j in ESTACIONES, p in PRODUCTOS} z[k,t,j,p,esc];

# Estabilidad de carga
param prod_cap{k in CAMIONES} := capacidad[k,'T1'] * capacidad[k,'T2'];

subject to Estabilidad_A{k in CAMIONES, i in NODOS, j in NODOS, esc in ESCENARIOS}:
    f[k,'T1',i,j,esc] * capacidad[k,'T2'] - f[k,'T2',i,j,esc] * capacidad[k,'T1'] 
    <= Delta * prod_cap[k];

subject to Estabilidad_B{k in CAMIONES, i in NODOS, j in NODOS, esc in ESCENARIOS}:
    f[k,'T2',i,j,esc] * capacidad[k,'T1'] - f[k,'T1',i,j,esc] * capacidad[k,'T2'] 
    <= Delta * prod_cap[k];
