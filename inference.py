
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde

# 1. Configurazione 
os.environ["DDE_BACKEND"] = "pytorch"
Re_min, Re_max = 20.0, 150.0
T_max = 10.0

# Definizione geometria (necessaria per inizializzare il modello)
rect = dde.geometry.Rectangle([-5, -2], [15, 2])
cyl = dde.geometry.Disk([0, 0], 0.5)
geom = dde.geometry.CSGDifference(rect, cyl)
timedomain = dde.geometry.TimeDomain(0, T_max)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Architettura della rete (uguale a train.py)
net = dde.nn.FNN([4] + [128] * 6 + [3], "tanh", "Glorot normal")

# Funzione dummy per la PDE (necessaria per caricare il modello)
def dummy_pde(x, y):
    return [torch.tensor([0])]

data = dde.data.TimePDE(geomtime, dummy_pde, [], num_domain=100)
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)

# 2. CARICAMENTO DEI PESI
# Sostituisci 'pinn_karman-30000.pt' con il nome esatto del file generato nella cartella
checkpoint_path = "model_parametric/pinn_karman-30000.pt" 

if os.path.exists(checkpoint_path):
    model.restore(checkpoint_path, verbose=1)
    print(f"Modello caricato da {checkpoint_path}")
else:
    print("Errore: file dei pesi non trovato. Controlla il numero di iterazioni nel nome del file.")

# 3. FUNZIONE PER PLOTTARE UN "ISTANTANEO"
def plot_snapshot(re_value, t_value):
    # Creazione griglia di punti
    x = np.linspace(-5, 15, 400)
    y = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x, y)
    
    # Prepariamo l'input: [x, y, t, Re]
    pts = np.vstack((X.flatten(), Y.flatten())).T
    t_col = np.full((pts.shape[0], 1), t_value)
    re_col = np.full((pts.shape[0], 1), re_value)
    test_input = np.hstack((pts, t_col, re_col)).astype(np.float32)
    
    # Predizione
    res = model.predict(test_input)
    u, v = res[:, 0], res[:, 1]
    vel_mag = np.sqrt(u**2 + v**2).reshape(X.shape)
    
    # Plot
    plt.figure(figsize=(12, 5))
    plt.contourf(X, Y, vel_mag, levels=50, cmap="jet")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar(label="Magnitudo Velocità")
    plt.title(f"Simulazione: Re={re_value}, t={t_value}s")
    plt.xlabel("x")
    plt.ylabel("y")
    
    # Disegniamo il cilindro (nero)
    theta = np.linspace(0, 2*np.pi, 100)
    plt.fill(0.5*np.cos(theta), 0.5*np.sin(theta), "k")
    
    plt.savefig(f"test_Re{re_value}_t{t_value}.png")
    print(f"Grafico salvato: test_Re{re_value}_t{t_value}.png")
    plt.show()

# Esempio di test: Reynolds 120 a metà simulazione (t=5s)
plot_snapshot(re_value=120.0, t_value=5.0)
