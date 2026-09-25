import sys
import os
import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

# Manejo de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
from core.lif_engine import LIFNetwork, phi_operator

# --- FUNCIONES DE MÉTRICAS ---

def calculate_dwell_times(spike_times, t_max):
    """Calcula los tiempos entre spikes (Inter-Spike Intervals - ISI), que son el equivalente al Dwell Time continuo."""
    if len(spike_times) < 2:
        return np.array([t_max])
    return np.diff(spike_times)

def calculate_boolean_dwell_times(b_vector, delta_t):
    """Calcula el dwell time en el espacio discreto (cuántos deltas permanece en 0 o 1)."""
    # Buscamos los índices donde cambia de estado (de 0 a 1 o de 1 a 0)
    transitions = np.where(np.diff(b_vector) != 0)[0]
    if len(transitions) == 0:
        return np.array([len(b_vector) * delta_t])
    
    # La duración de cada estado es la diferencia de índices multiplicada por delta_t
    durations = np.diff(np.concatenate(([0], transitions, [len(b_vector)]))) * delta_t
    return durations

def compute_metrics_for_topology(delta_t=10.0, lambda_p=0.05, naive_threshold=0.5):
    """Corre una simulación y devuelve las métricas comparativas."""
    T_MAX = 600.0
    DT = 0.1
    steps = int(T_MAX / DT)
    params = {'tau_m': 20.0, 'v_reset': 0.0, 'theta': 1.0, 'R_m': 1.0}
    net = LIFNetwork(n_nodes=4, params=params)

    # Simulación (similar a diamond_test.py pero condensada)
    v_history = np.zeros((4, steps))
    for i, t in enumerate(np.linspace(0, T_MAX, steps)):
        poisson_noise = np.random.random(size=4) < (lambda_p * DT)
        net.v += poisson_noise * 0.1 
        currents = np.zeros(4)
        if (t % 120) < 5: currents[0] = 25.0
        fired = net.step(t, DT, currents)
        v_history[:, i] = net.v
        
        if fired[0]:
            net.v[1] += 0.45; net.v[2] += 0.40
        if fired[1]: net.v[3] += 0.45 
        if fired[2]: net.v[3] += 0.45

    # 1. Nuestro Operador Phi (El "Integrator" Node n4)
    b_phi = phi_operator(net.spikes[3], T_MAX, delta_t)

    # 2. Naive Thresholding (Mapeo "Ingenuo" directo de voltaje a booleano)
    # Hacemos submuestreo al mismo delta_t para ser justos en la comparación
    subsample_step = int(delta_t / DT)
    v_subsampled = v_history[3, ::subsample_step][:len(b_phi)] 
    b_naive = (v_subsampled >= naive_threshold).astype(int)

    # --- CÁLCULO DE MÉTRICAS ---
    
    # A) Mutual Information (Fidelidad temporal respecto al pacemaker n1 original)
    # Necesitamos el pacemaker (n1) mapeado con Phi como la "señal original ideal"
    b_ideal_input = phi_operator(net.spikes[0], T_MAX, delta_t)
    
    # Aseguramos el mismo tamaño
    min_len = min(len(b_ideal_input), len(b_phi), len(b_naive))
    b_ideal_input, b_phi, b_naive = b_ideal_input[:min_len], b_phi[:min_len], b_naive[:min_len]

    mi_phi = mutual_info_score(b_ideal_input, b_phi)
    mi_naive = mutual_info_score(b_ideal_input, b_naive)

    # B) Dwell Time Distribution Error (Wasserstein Distance)
    # Calculamos ISI reales continuos del integrador
    real_dwells = calculate_dwell_times(net.spikes[3], T_MAX)
    # Calculamos distribuciones para Phi y Naive
    phi_dwells = calculate_boolean_dwell_times(b_phi, delta_t)
    naive_dwells = calculate_boolean_dwell_times(b_naive, delta_t)
    
    # Distancia de Wasserstein (menor es mejor)
    wd_phi = wasserstein_distance(real_dwells, phi_dwells)
    wd_naive = wasserstein_distance(real_dwells, naive_dwells)

    # C) Symbolic False Nearest Neighbors (Proxy: Causal Jitter / Falsos Positivos)
    # Un FNN aquí ocurre cuando el umbral se cruza por ruido pero no hay un spike real causal.
    # Contamos activaciones discretas vs spikes reales.
    real_spike_count = len(net.spikes[3])
    phi_activation_count = np.sum(b_phi)
    naive_activation_count = np.sum(np.diff(np.concatenate(([0], b_naive))) == 1) # Cuenta cruces de 0 a 1

    fnn_phi_error = abs(phi_activation_count - real_spike_count) / max(1, real_spike_count)
    fnn_naive_error = abs(naive_activation_count - real_spike_count) / max(1, real_spike_count)

    return (mi_phi, mi_naive), (wd_phi, wd_naive), (fnn_phi_error, fnn_naive_error)

# --- EJECUCIÓN Y GENERACIÓN DE RESULTADOS ---

# --- EJECUCIÓN Y GENERACIÓN DE RESULTADOS ---

if __name__ == "__main__":
    print("Corriendo análisis comparativo...")
    
    # Parámetros base
    delta_t = 15.0 # Mismo que el paper
    lambda_p = 0.08
    
    (mi_phi, mi_naive), (wd_phi, wd_naive), (fnn_phi, fnn_naive) = compute_metrics_for_topology(delta_t=delta_t, lambda_p=lambda_p)

    # 1. Impresión en consola para revisión rápida
    print("\n" + "="*50)
    print(" TABLA DE INDICADORES CUANTITATIVOS (Generando LaTeX...)")
    print("="*50)
    print(f"Metric\t\t\tPhi_Operator\tNaive Threshold")
    print("-" * 50)
    print(f"Mutual Info (Bits) \t{mi_phi:.4f}\t\t{mi_naive:.4f}")
    print(f"Dwell-Time Error (W-D)\t{wd_phi:.2f}\t\t{wd_naive:.2f}")
    print(f"Topology Error (FNN) \t{fnn_phi:.1%}\t\t{fnn_naive:.1%}")
    print("="*50)

    # 2. Generación automática del código LaTeX
    latex_code = f"""\\begin{{table}}[ht]
\\centering
\\caption{{Quantitative Performance Metrics ($\\Delta t = {delta_t}ms$, Poisson Noise $\\lambda = {lambda_p}$)}}
\\label{{tab:quantitative_metrics}}
\\begin{{tabular}}{{@{{}}lcc@{{}}}}
\\toprule
\\textbf{{Metric}} & \\textbf{{$\\Phi_{{\\Delta t}}$ (Ours)}} & \\textbf{{Naive Thresholding}} \\\\ \\midrule
Mutual Information $\\uparrow$ & {mi_phi:.4f} & {mi_naive:.4f} \\\\
Dwell-Time Error (W.D.) $\\downarrow$ & {wd_phi:.2f} & {wd_naive:.2f} \\\\
Topology Error (FNN Proxy) $\\downarrow$ & {fnn_phi*100:.1f}\\% & {fnn_naive*100:.1f}\\% \\\\ \\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    # 3. Guardar el archivo .tex en la carpeta results/
    results_dir = os.path.join(BASE_DIR, '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    tex_path = os.path.join(results_dir, 'quantitative_metrics.tex')
    
    with open(tex_path, 'w') as f:
        f.write(latex_code)
        
    print(f"\n¡Éxito! Archivo LaTeX generado en: {tex_path}")
    print("Puedes insertarlo en tu documento usando: \\input{results/quantitative_metrics.tex}")