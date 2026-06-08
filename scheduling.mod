# Modelo de scheduling para la carga de camiones en el deposito
# Se busca minimizar el makespan (cuando termina el ultimo camion)

# Conjuntos
set OPERACIONES;   # cargar un tanque de un camion

# Parametros
param duracion{OPERACIONES} >= 0;    # tiempo de carga mas 5 min de limpieza
param camion{OPERACIONES} symbolic;  # a que camion pertenece la operacion
param bahia{OPERACIONES} symbolic;   # que bahia usa (R o D)

# Pares de operaciones que comparten recurso (mismo camion o misma bahia)
set PARES_CAMION := {i in OPERACIONES, j in OPERACIONES: i <> j and camion[i] = camion[j]};
set PARES_BAHIA := {i in OPERACIONES, j in OPERACIONES: i <> j and bahia[i] = bahia[j]};
set PARES := PARES_CAMION union PARES_BAHIA;

param M := sum{i in OPERACIONES} duracion[i];   # big-M

# Variables
var inicio{OPERACIONES} >= 0;   # minuto de inicio de cada operacion
var Cmax >= 0;                  # makespan
var y{(i,j) in PARES} binary;   # 1 si la operacion i precede a la j

# Objetivo
minimize Makespan: Cmax;

# Restricciones

# Para cada par en conflicto, se define quien va primero
subject to Orden{(i,j) in PARES: ord(i) < ord(j)}:
    y[i,j] + y[j,i] = 1;

# Si i precede a j, j no puede empezar hasta que i termine
subject to Precedencia{(i,j) in PARES}:
    inicio[j] >= inicio[i] + duracion[i] - M * (1 - y[i,j]);

# El makespan es el maximo de los tiempos de termino
subject to Def_Cmax{i in OPERACIONES}:
    Cmax >= inicio[i] + duracion[i];
