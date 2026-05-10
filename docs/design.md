# Documento de diseño — PSO-parallel

## 1. Visión general

El proyecto implementa Particle Swarm Optimization (PSO) canónico para minimizar funciones continuas en R^d, con soporte para múltiples estrategias de evaluación del fitness: secuencial (V0), hilos (V1), procesos (V2) y concurrencia cooperativa con asyncio (V3). El objetivo principal es que el núcleo del algoritmo permanezca invariable y solo cambie la capa de evaluación.

---

## 2. Arquitectura modular

```
core/           Motor PSO: partícula, enjambre, bucle principal, abstracciones
objectives/     Funciones benchmark (incluye noisy_sphere para el caso de uso de V3)
parallel/       Implementaciones de evaluación (V0, V1, V2, V3)
experiments/    Orquestación: runner y grid search
storage/        Persistencia (JSON/CSV) y logging estructurado
viz/            Visualización: convergencia y animación 2D
tests/          Tests unitarios
```

Las dependencias fluyen en una sola dirección: `experiments` y scripts de entrada dependen de `core`, `parallel`, `storage` y `viz`. El módulo `core` no conoce ningún módulo externo a él.

### Interfaces clave

Tres ABCs desacoplan las estrategias del motor:

- `FitnessEvaluator` (`core/evaluator.py`): contrato `evaluate(positions) → List[float]` con ciclo de vida `open()`/`close()`. Permite intercambiar V0/V1/V2 sin tocar el bucle PSO.
- `BoundsPolicy` (`core/bounds.py`): contrato `apply(position, velocity, lo, hi) → (position, velocity)`. Encapsula la estrategia de manejo de límites.
- `Topology` (`core/topology.py`): contrato `get_best(particle_idx, swarm) → position`. Permite cambiar la topología de vecindad sin modificar la actualización de velocidad.

---

## 3. Decisiones de diseño

### 3.1 Separación evaluador / motor PSO

La evaluación del fitness es el único punto de variabilidad entre V0, V1 y V2. Aislarla en `FitnessEvaluator` hace que `PSO.run()` sea idéntico para las tres versiones. La consecuencia directa es que los resultados son reproducibles y comparables: dado el mismo seed, las tres versiones producen el mismo `best_fitness`.

### 3.2 Generador de números aleatorios local

Se usa `np.random.default_rng(seed)` en lugar del estado global `np.random.seed()`. El generador se instancia en `PSO.__init__` y se pasa explícitamente a `build_swarm` y a `particle.update_velocity`. Esto evita que llamadas externas (p. ej. NumPy en los workers de multiprocessing) contaminen la secuencia aleatoria del enjambre.

### 3.3 Estrategia de límites: ClampBounds

Se eligió `ClampBounds` sobre reflect o penalty por dos razones: es simple de razonar (no introduce energía ficticia en el sistema) y es fácil de verificar en tests. El comportamiento es: la posición se recorta a `[lo, hi]` y la velocidad se pone a cero en las dimensiones que colisionaron. El coste es que las partículas pueden acumularse en las paredes si el espacio es muy restrictivo, pero no es relevante para las funciones benchmark usadas.

### 3.4 Topología global-best

Se implementa `GlobalBestTopology` como única topología. Todas las partículas son atraídas por el mejor global del enjambre. Converge más rápido que topologías locales pero es más susceptible a mínimos locales. Es la elección estándar para comparar tiempos entre estrategias de paralelismo, ya que su comportamiento es predecible.

### 3.5 Criterio de parada por estancamiento

Además del límite de iteraciones, el PSO para si el mejor fitness no mejora más de `tol` durante `patience` iteraciones consecutivas. Esto evita iteraciones innecesarias y reduce el tiempo de cómputo en funciones fáciles como Sphere, sin afectar la calidad del resultado.

### 3.6 Carpeta `storage/` en lugar de `io/`

El enunciado sugiere el nombre `io/` para el módulo de persistencia. Se renombró a `storage/` porque `io` es un módulo de la biblioteca estándar de Python; crear un paquete con ese nombre hace que los imports fallen en cualquier contexto donde Python busque `import io`. El comportamiento es idéntico, solo cambia el nombre del directorio.

### 3.7 Directorio de resultados incluye evaluator

El path de cada experimento es `results/<objective>_d<dim>_s<seed>_<evaluator>/`. Incluir el evaluator evita que las ejecuciones de V0, V1 y V2 con los mismos parámetros se sobreescriban, lo que es necesario para el script de análisis comparativo.

---

## 4. Estrategias de paralelismo

### V0 — Secuencial (`parallel/sequential.py`)

```python
[float(self.objective(x)) for x in positions]
```

List comprehension Python puro. Referencia de rendimiento. Sin overhead. Es la estrategia más rápida para funciones de evaluación barata (< 1ms por partícula).

### V1 — ThreadPoolExecutor (`parallel/threading_eval.py`)

```python
list(self._executor.map(self.objective, positions))
```

El pool de hilos se crea una vez en `open()` y se reutiliza en todas las iteraciones. `executor.map` preserva el orden de los resultados, garantizando reproducibilidad.

**Limitación — GIL**: Python solo ejecuta un hilo a la vez para código Python puro. Las funciones benchmark (sphere, ackley, etc.) son CPU-bound en Python, por lo que V1 es sistemáticamente más lento que V0: el overhead de crear/gestionar hilos no se amortiza. V1 sería beneficioso si la función objetivo liberara el GIL (código C/Fortran vía NumPy, I/O de red, consultas a base de datos).

### V2 — ProcessPoolExecutor (`parallel/multiprocessing_eval.py`)

Cada worker corre en un proceso separado con su propio intérprete Python. No hay GIL. Sin embargo, cada llamada requiere serializar (pickle) las posiciones, enviarlas por IPC, ejecutar la función en el worker, serializar el resultado y devolverlo.

**Batching**: para reducir el número de round-trips de IPC, las posiciones se agrupan en lotes (`batch_size = max(1, n_particles // (n_workers * 2))`). Cada worker recibe un lote y lo evalúa secuencialmente. Con funciones baratas, el overhead de serialización sigue dominando y V2 es más lento que V0 (factor ~5–10x en los benchmarks). Con funciones caras (>10ms por partícula), V2 proporciona speedup real.

### V3 — asyncio + run_in_executor (`parallel/asyncio_eval.py`)

```python
async def _eval_all(self, positions):
    tasks = [self._eval_one(x, semaphore) for x in positions]
    return list(await asyncio.gather(*tasks))
```

`asyncio.gather` lanza todas las corrutinas simultáneamente. Cada una delega la evaluación al thread pool del event loop mediante `loop.run_in_executor(None, self.objective, x)`. Mientras un hilo está bloqueado esperando I/O, el event loop programa los demás: el tiempo total es aproximadamente `max(latencias)` en lugar de `sum(latencias)`.

**Caso de uso — `noisy_sphere`**: las funciones benchmark estándar son CPU-bound y no se benefician de asyncio (los hilos siguen compitiendo por el GIL, igual que en V1). Para que V3 tenga sentido se diseñó `objectives/noisy_service.py`, que envuelve la función Sphere con una pausa aleatoria de 5–50 ms por partícula, simulando una consulta a un servicio externo (REST, simulador, base de datos). Con 30 partículas, V0 tardaría ~750 ms por iteración; V3 tarda ~50 ms.

**Parámetro `max_concurrent`**: si el servicio tiene un límite de peticiones concurrentes, se puede pasar un `asyncio.Semaphore` internamente mediante `max_concurrent=N`. Equivale al parámetro `max_workers` de V1/V2.

**Limitación**: `asyncio.run()` crea y destruye un event loop por cada llamada a `evaluate()` (una por iteración del PSO). El overhead es despreciable frente a cualquier latencia realista de I/O, pero sería relevante si la función objetivo fuera sub-milisegundo.

---

## 5. Persistencia

Cada experimento guarda dos ficheros en `results/<nombre>/`:

- `summary.json`: configuración completa, métricas finales, hash del commit git e información de hardware. El hash del commit permite reproducir exactamente cualquier resultado.
- `history.csv`: mejor fitness por iteración, para plots de convergencia.

Se eligieron JSON y CSV sobre YAML o bases de datos por ser legibles sin herramientas adicionales y fácilmente cargables con la biblioteca estándar de Python.

---

## 6. Limitaciones conocidas

| Limitación | Impacto |
|---|---|
| V1 y V2 más lentos que V0 para funciones baratas | Esperado por GIL e IPC; se documenta en los resultados |
| V3 sin ventaja sobre V1 para funciones CPU-bound | asyncio no elimina el GIL; usar V3 solo con objetivos I/O-bound |
| V3 crea un event loop por iteración (`asyncio.run`) | Overhead despreciable para latencias >1 ms, pero subóptimo para funciones muy rápidas |
| Reproducibilidad de V1/V2 depende del orden de `executor.map` | `map` preserva orden, pero no garantizado para `submit` con orden manual |
| Sin topología local (ring, von Neumann) | Solo global-best; fácil de añadir implementando `Topology` |
| Animación solo para d=2 | Limitación de visualización 2D; para d=3 se requeriría proyección |
| Sin compresión de trayectorias | Para experimentos largos, `history.csv` puede crecer; se puede añadir muestreo |
