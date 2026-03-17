# CBN-LIF: Discrete Mapping of Excitable Dynamics

Este repositorio contiene la implementación oficial del operador de proyección $\Phi_{\Delta t}$, diseñado para mapear trayectorias continuas de modelos **Leaky Integrate-and-Fire (LIF)** hacia el dominio de **Redes Booleanas Acopladas (CBN)**.

Este trabajo ha sido desarrollado para su presentación en el **V Congreso Internacional de Neurociencias (NeuroUTEC 2026)**.

## 🧠 Descripción del Proyecto

El núcleo de esta investigación es el operador $\Phi_{\Delta t}$, el cual permite discretizar la actividad neuronal preservando las propiedades de sincronización y suma espacial. A diferencia de las discretizaciones estáticas, nuestro enfoque considera una ventana temporal de integración que permite filtrar el ruido sináptico (Poisson) y capturar la causalidad biológica en modelos de alta eficiencia computacional.

### Características principales:
* **LIF Engine:** Integrador vectorial de ecuaciones diferenciales para neuronas de tipo *Integrate-and-Fire*.
* **Stochastic Robustness:** Simulación de bombardeo sináptico mediante procesos de Poisson.
* **CBN Mapping:** Generación automática de estados booleanos a partir de spikes biofísicos.
* **Analysis:** Script de validación para la topología "Diamond" y análisis de sensibilidad de $\Delta t$.

## 🚀 Instalación y Uso

### Requisitos
* Python 3.12+
* NumPy
* Matplotlib (para la generación de gráficas)

### Ejecución
Para replicar los resultados presentados en el paper:
```bash
python3 main_simulation.py