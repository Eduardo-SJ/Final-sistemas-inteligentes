import heapq
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

# Algoritmo A*
class Nodo:
    def __init__(self, nombre, conexiones):
        self.nombre = nombre
        self.conexiones = conexiones

class Conexion:
    def __init__(self, destino, costo):
        self.destino = destino
        self.costo = costo

class Grafo:
    def __init__(self):
        self.nodos = {}

    def agregar_nodo(self, nombre, conexiones):
        self.nodos[nombre] = Nodo(nombre, conexiones)

    def heuristica(self, nodo_actual, nodos_objetivo):
        return min([self.distancia(nodo_actual, objetivo) for objetivo in nodos_objetivo], default=0)

    def distancia(self, nodo_a, nodo_b):
        for conexion in self.nodos[nodo_a].conexiones:
            if conexion.destino == nodo_b:
                return conexion.costo
        return float('inf')

    def a_star(self, inicio, objetivos):
        open_list = []
        heapq.heappush(open_list, (0, inicio, frozenset(objetivos.copy())))
        came_from = {(inicio, frozenset(objetivos.copy())): None}
        g_cost = {(inicio, frozenset(objetivos.copy())): 0}

        while open_list:
            _, nodo_actual, objetivos_restantes = heapq.heappop(open_list)

            if not objetivos_restantes:
                return self.reconstruir_camino(came_from, inicio, nodo_actual)

            for conexion in self.nodos[nodo_actual].conexiones:
                nuevo_objetivo = objetivos_restantes - {conexion.destino}
                nuevo_costo = g_cost[(nodo_actual, objetivos_restantes)] + conexion.costo

                if (conexion.destino, frozenset(nuevo_objetivo)) not in g_cost or nuevo_costo < g_cost[(conexion.destino, frozenset(nuevo_objetivo))]:
                    g_cost[(conexion.destino, frozenset(nuevo_objetivo))] = nuevo_costo
                    prioridad = nuevo_costo + self.heuristica(conexion.destino, nuevo_objetivo)
                    heapq.heappush(open_list, (prioridad, conexion.destino, frozenset(nuevo_objetivo)))
                    came_from[(conexion.destino, frozenset(nuevo_objetivo))] = (nodo_actual, objetivos_restantes)

        return None

    def reconstruir_camino(self, came_from, inicio, fin):
        camino = []
        nodo_actual = (fin, frozenset())
        while nodo_actual[0] != inicio:
            camino.append(nodo_actual[0])
            nodo_actual = came_from[nodo_actual]
        camino.append(inicio)
        camino.reverse()
        return camino

# el programa se basará en las distancias al inicio del problema para poder buscar la ruta mas optima
grafo = Grafo()
grafo.agregar_nodo('E', [
    Conexion('T1', 10),
    Conexion('T4', 12),
    Conexion('T7', 15)
])
grafo.agregar_nodo('T1', [
    Conexion('T2', 9),
    Conexion('T4', 8)
])
grafo.agregar_nodo('T2', [
    Conexion('T3', 6),
    Conexion('T5', 3)
])
grafo.agregar_nodo('T3', [
    Conexion('T4', 6),
    Conexion('T7', 5),
    Conexion('T8', 2)
])
grafo.agregar_nodo('T4', [
    Conexion('T5', 6)
])
grafo.agregar_nodo('T5', [
    Conexion('T6', 2)
])
grafo.agregar_nodo('T6', [
    Conexion('T7', 3)
])
grafo.agregar_nodo('T7', [
    Conexion('T8', 2)
])
grafo.agregar_nodo('T8', [
    Conexion('T3', 2),
    Conexion('T7', 2)
])

# Definimos el punto de partida osea la empresa y los objetivos (los empleados)
inicio = 'E'
objetivos = {'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8'}   # listamos los empleados del 1 al 8

# Ejecutamo el algoritmo A* para encontrar la mejor ruta
camino_optimo = grafo.a_star(inicio, objetivos)
print("Camino óptimo:", " -> ".join(camino_optimo))

# Lógica difusa para estimar el consumo de combustible
trafico = ctrl.Antecedent(np.arange(0, 11, 1), 'trafico')
consumo_combustible = ctrl.Consequent(np.arange(0, 101, 1), 'consumo_combustible')
tiempo_llegada = ctrl.Antecedent(np.arange(0, 61, 1), 'tiempo_llegada')

trafico['bajo'] = fuzz.trimf(trafico.universe, [0, 0, 4])
trafico['medio'] = fuzz.trimf(trafico.universe, [2, 5, 8])
trafico['alto'] = fuzz.trimf(trafico.universe, [6, 10, 10])

tiempo_llegada['temprano'] = fuzz.trimf(tiempo_llegada.universe, [0, 0, 20])
tiempo_llegada['normal'] = fuzz.trimf(tiempo_llegada.universe, [15, 30, 45])
tiempo_llegada['tarde'] = fuzz.trimf(tiempo_llegada.universe, [40, 60, 60])

consumo_combustible['bajo'] = fuzz.trimf(consumo_combustible.universe, [0, 0, 30])
consumo_combustible['medio'] = fuzz.trimf(consumo_combustible.universe, [20, 50, 80])
consumo_combustible['alto'] = fuzz.trimf(consumo_combustible.universe, [70, 100, 100])

regla1 = ctrl.Rule(trafico['alto'] & tiempo_llegada['tarde'], consumo_combustible['alto'])
regla2 = ctrl.Rule(trafico['medio'] & tiempo_llegada['normal'], consumo_combustible['medio'])
regla3 = ctrl.Rule(trafico['bajo'] & tiempo_llegada['temprano'], consumo_combustible['bajo'])
regla4 = ctrl.Rule(trafico['alto'] & tiempo_llegada['temprano'], consumo_combustible['medio'])

control_consumo = ctrl.ControlSystem([regla1, regla2, regla3, regla4])
simulacion_consumo = ctrl.ControlSystemSimulation(control_consumo)

#  Simulación con tráfico alto y tiempo de llegada tarde
simulacion_consumo.input['trafico'] = 8
simulacion_consumo.input['tiempo_llegada'] = 50
simulacion_consumo.compute()

#  consumo de combustible esperado
consumo_estimado = simulacion_consumo.output['consumo_combustible']
print(f"Consumo de combustible estimado: {consumo_estimado}")


consumo_combustible.view(simulacion=simulacion_consumo)
plt.show()

# Clasificación Bayesiana para determinar si la ruta es "buena" o "mala"
# Datos de entrenamiento simplificados ;tráfico, tiempo de llegada, clase
X_train = np.array([
    [8, 50],  # tráfico alto y llegada tarde
    [3, 15],  # tráfico bajo y llegada temprano
    [5, 30],  # tráfico medio y llegada normal
    [9, 40]   # tráfico alto y llegada temprano
])
y_train = np.array(['mala', 'buena', 'buena', 'mala'])

# Entrenamiento del modelo Bayesiano
modelo_bayesiano = GaussianNB()
modelo_bayesiano.fit(X_train, y_train)

# Clasificación basada en el tráfico y tiempo de llegada actual
X_test = np.array([[8, 50]])
prediccion = modelo_bayesiano.predict(X_test)
print(f"Predicción para la ruta: {prediccion[0]}")
