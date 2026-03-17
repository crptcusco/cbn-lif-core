import numpy as np

class LIFNetwork:
    def __init__(self, n_nodes, params):
        self.n_nodes = n_nodes
        self.params = params  # <--- ESTA ES LA LÍNEA QUE FALTA
        
        # Inicializamos voltajes en el valor de reset
        self.v = np.full(n_nodes, params['v_reset'], dtype=float)
        
        # Lista de listas para guardar los tiempos de disparo
        self.spikes = [[] for _ in range(n_nodes)]

    def step(self, t, dt, I_ext):
        # Ahora self.params['v_reset'] ya no dará error
        dv = (-(self.v - self.params['v_reset']) + self.params['R_m'] * I_ext) / self.params['tau_m']
        self.v += dv * dt
        
        fired = self.v >= self.params['theta']
        
        for i in range(self.n_nodes):
            if fired[i]:
                self.spikes[i].append(t)
                self.v[i] = self.params['v_reset']
                
        return fired

def phi_operator(spike_times, t_max, delta_t):
    """
    Implementación del Operador Phi_Delta_t.
    Proyecta eventos discretos (spikes) a un espacio booleano.
    """
    num_bins = int(np.ceil(t_max / delta_t))
    b_vector = np.zeros(num_bins, dtype=int)
    
    for ts in spike_times:
        bin_idx = int(ts // delta_t)
        if bin_idx < num_bins:
            b_vector[bin_idx] = 1
    return b_vector