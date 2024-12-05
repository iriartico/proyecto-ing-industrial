import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from pulp import LpVariable, LpProblem, LpMinimize, GLPK, lpSum, value


matplotlib.use("Qt5Agg")


def create_variable_matrix(
    rows, cols, name="var", cat="Continuous", lowBound=None, upBound=None
):
    """
    Crea una matriz de variables de decisión para problemas de optimización.

    :param rows: Número de filas de la matriz.
    :param cols: Número de columnas de la matriz.
    :param name: Nombre base de las variables.
    :param cat: Tipo de variable ('Continuous', 'Integer', 'Binary').
    :param lowBound: Límite inferior para las variables.
    :param upBound: Límite superior para las variables.
    :return: Matriz de variables de decisión.
    """
    return [
        [
            LpVariable(f"{name}_{i}_{j}", lowBound=lowBound, upBound=upBound, cat=cat)
            for j in range(cols)
        ]
        for i in range(rows)
    ]


distribution_points = [
    [-19.074201, -65.2710989],  # ECOLCHON
    [-19.038636, -65.250002],  # IMPORTADORA SOLIVE
    [-19.018777, -65.279010],  # FABRICACION DE MAQUINARIA IND
    [-19.042055, -65.252021],  # SECOMCI SUCURSAL 1
    [-19.029592, -65.262512],  # INDUSTRIAS METALURGICAS TORRES
]

# Nombres de las empresas
names = [
    "ECOLCHON",
    "IMPORTADORA SOLIVE",
    "FABRICACION DE MAQUINARIA IND",
    "SECOMCI SUCURSAL 1",
    "INDUSTRIAS METALURGICAS TORRES",
]

# Crear variable 'prob'
distribution_points = np.array(distribution_points)

# Graficar
plt.figure(figsize=(10, 6))
# plt.text(-19.074201, -65.2710989, "EMPRESA", fontsize=9, color="red")
plt.scatter(distribution_points[:, 0], distribution_points[:, 1])

for (x, y), name in zip(distribution_points, names):
    plt.text(x, y, name, fontsize=8, color="gray", ha="right")

# Mostrar gráfica
plt.xlabel("Latitud")
plt.ylabel("Longitud")
plt.title("Puntos de distribución")
plt.show()
input("Presiona Enter para continuar con la ejecución...")
# plt.savefig("puntos_de_distribucion.png")
# plt.show()

# Calcular matriz de distancias
distance_matrix = distance_matrix(distribution_points, distribution_points)
number_distribution_points = len(distribution_points)
print(distance_matrix)


problem = LpProblem("traveling_salesman:problem", LpMinimize)
f = create_variable_matrix(
    number_distribution_points,
    number_distribution_points,
    name="f",
    cat="Integer",
    lowBound=0,
)

x = create_variable_matrix(
    number_distribution_points,
    number_distribution_points,
    name="x",
    cat="Integer",
    lowBound=0,
    upBound=1,
)
print("Tamaño", len(x), len(x[0]))


# Función objetivo
objective = 0
for i in range(number_distribution_points):
    for j in range(number_distribution_points):
        objective += distance_matrix[i][j] * x[i][j]
problem += objective  # Agregar la función objetivo al problema

# Restricción 1: Que a cada punto de distribución solamente llegue una arista
for i in range(number_distribution_points):
    restriction = lpSum(x[i][j] for j in range(number_distribution_points))
    problem += restriction == 1  # Cada punto debe recibir exactamente una conexión

# Restricción 2: Que a cada punto de distribución solamente salga una arista
for j in range(number_distribution_points):
    restriction = lpSum(x[i][j] for i in range(number_distribution_points))
    problem += restriction == 1  # Cada punto debe tener exactamente una salida

# Restricción 3: Balance de flujo (entrada y salida deben ser iguales)
for k in range(1, number_distribution_points):
    restriction = lpSum(f[i][k] for i in range(number_distribution_points)) - lpSum(
        f[k][j] for j in range(number_distribution_points)
    )
    problem += restriction == 1

# Restricción 4: Flujo condicionado por la conexión
for i in range(number_distribution_points):
    for j in range(number_distribution_points):
        if i != j:  # Evitar conexiones de un nodo consigo mismo
            problem += f[i][j] <= (number_distribution_points - 1) * x[i][j]
# Tiempo maximo de ejecución en segundos

GLPK(options=["--mipgap", "0.01", "--tmlim", "60"]).solve(problem)
for v in problem.variables():
    if v.varValue != 0:
        print(v.name, "=", v.varValue)
print("objetive = ", value(problem.objective))


conexiones = []
for i in range(number_distribution_points):
    conexiones_i = []
    for j in range(number_distribution_points):
        value = 0
        if x[i][j].varValue is not None:
            value = float(x[i][j].varValue)
        conexiones_i.append(value)
    conexiones.append(conexiones_i)
conexiones = np.array(conexiones)
print("Tamaño :", conexiones.shape)
conexiones


plt.figure(figsize=(10, 6))
plt.scatter(distribution_points[:, 0], distribution_points[:, 1], alpha=0.3)
for i in range(number_distribution_points):
    for j in range(number_distribution_points):
        if conexiones[i][j] != 0 and i != j:
            coordenate_i_x = distribution_points[i][0]
            coordenate_i_y = distribution_points[i][1]
            coordenate_j_x = distribution_points[j][0]
            coordenate_j_y = distribution_points[j][1]
            plt.plot(
                [coordenate_i_x, coordenate_j_x],
                [coordenate_i_y, coordenate_j_y],
                color="red",
                linestyle="-",
            )

for (x, y), name in zip(distribution_points, names):
    plt.text(x, y, name, fontsize=8, color="gray", ha="right")

plt.xlabel("Latitud")
plt.ylabel("Longitud")
plt.title("Rutas de distribución")
# plt.savefig("rutas.png")
plt.show()
