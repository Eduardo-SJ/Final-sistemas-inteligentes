import heapq

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
        # heurística : distancia mínima a cualquier nodo objetivo restante
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

                if (conexion.destino, frozenset(nuevo_objetivo)) not in g_cost or nuevo_costo \
                        < g_cost[(conexion.destino, frozenset(nuevo_objetivo))]:
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

# Ejecutamos el algoritmo A* para encontrar la mejor ruta
camino_optimo = grafo.a_star(inicio, objetivos)
print("Camino óptimo:", " -> ".join(camino_optimo))
