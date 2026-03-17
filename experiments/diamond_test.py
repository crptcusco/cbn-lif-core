import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Manejo de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
from core.lif_engine import LIFNetwork, phi_operator

# --- 1. CONFIGURACIÓN ---
T_MAX = 600.0   # ms
DT = 0.1        # Paso de tiempo
DELTA_T = 10.0  # Ventana Booleana (reducida para ver más detalle)
steps = int(T_MAX / DT)

params = {'tau_m': 20.0, 'v_reset': 0.0, 'theta': 1.0, 'R_m': 1.0}
net = LIFNetwork(n_nodes=4, params=params)

# Almacenamiento para las ondas de voltaje
v_history = np.zeros((4, steps))
time_axis = np.linspace(0, T_MAX, steps)

# --- 2. SIMULACIÓN CON ENUMERATE ---
for i, t in enumerate(time_axis):
    # --- AJUSTE DE RUIDO DE POISSON ---
    # lambda_p es la tasa de disparo (Hz). 0.05 significa unos 50 spikes por segundo.
    lambda_p = 0.05 
    # Generamos una máscara de Poisson: 1 si hay ruido en este DT, 0 si no.
    poisson_noise = np.random.random(size=4) < (lambda_p * DT)

    # Si hay un "spike" de ruido, le damos un empujoncito al voltaje
    net.v += poisson_noise * 0.1  # 0.1 mV por cada entrada aleatoria
    
    currents = np.zeros(4)
    # --- AJUSTES EN EL BUCLE ---
    if (t % 120) < 5:  # Reducimos el pulso de 15ms a 5ms
        currents[0] = 25.0 # Un golpe fuerte pero muy breve
    
    fired = net.step(t, DT, currents)
    v_history[:, i] = net.v
    
    # 2. Rompemos la simetría de los pesos
    # Si n2 y n3 tienen pesos diferentes, no dispararán siempre al mismo tiempo
    if fired[0]:
        net.v[1] += 0.45  # Valor más bajo para que necesite subir un poco más por su cuenta
        net.v[2] += 0.40
        
    # 3. La regla de coincidencia para n4
    # Ahora n4 necesita que AMBOS (2 y 3) aporten para llegar al umbral de 1.0
    if fired[1]: net.v[3] += 0.45 
    if fired[2]: net.v[3] += 0.45

print(f"Spikes detectados: {[len(s) for s in net.spikes]}")

# --- 3. PROCESAMIENTO BOOLEANO ---
boolean_matrix = []
for i in range(4):
    b_vec = phi_operator(net.spikes[i], T_MAX, DELTA_T)
    boolean_matrix.append(b_vec)

# --- 4. VISUALIZACIÓN TRIPLE ---
# Usamos subplots con height_ratios para que el primer panel sea el doble de alto
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), 
                                     gridspec_kw={'height_ratios': [2, 1, 1]})

# PANEL 1: Potencial de Membrana (Las "Ondas")
ax1.plot(time_axis, v_history[0], label='n1 (Pacemaker)', color='C0', lw=1.2, alpha=0.8)
ax1.plot(time_axis, v_history[3], label='n4 (Integrator)', color='C3', lw=1.5)
ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.6, label='Threshold')
ax1.set_ylabel("Voltage $V_m(t)$")
ax1.set_title("Continuous Dynamics: Membrane Potential (Temporal Summation)")
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, T_MAX)

# PANEL 2: Raster Plot (Los disparos)
for i in range(4):
    if len(net.spikes[i]) > 0:
        ax2.vlines(net.spikes[i], i + 0.6, i + 1.4, color=f'C{i}', lw=2)
ax2.set_yticks([1, 2, 3, 4])
ax2.set_yticklabels(['n1', 'n2', 'n3', 'n4'])
ax2.set_ylabel("Neuron Index")
ax2.set_title("Event Dynamics: Spike Raster Plot")
ax2.set_xlim(0, T_MAX)
ax2.grid(True, axis='x', alpha=0.2)

# PANEL 3: Matriz Booleana (El mapeo CBN)
ax3.imshow(boolean_matrix, aspect='auto', cmap='binary', origin='lower', extent=[0, T_MAX, 0.5, 4.5])
ax3.set_yticks([1, 2, 3, 4])
ax3.set_yticklabels(['n1', 'n2', 'n3', 'n4'])
ax3.set_title(rf"Discrete Dynamics: Boolean Mapping $\Phi$ ($\Delta t = {DELTA_T}ms$)")
ax3.set_xlabel("Time (ms)")
ax3.set_xlim(0, T_MAX)

plt.tight_layout()
save_path = os.path.join(BASE_DIR, '..', 'results', 'diamond_validation.png')
plt.savefig(save_path, dpi=300)
print(f"Gráfica científica generada en: {save_path}")