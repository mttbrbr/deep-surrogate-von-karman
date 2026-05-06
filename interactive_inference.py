import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import deepxde as dde
import torch

# 1. SETUP BACKEND E GEOMETRIA (Indispensabile)
os.environ["DDE_BACKEND"] = "pytorch"

Re_min, Re_max = 20.0, 150.0
T_max = 10.0

rect = dde.geometry.Rectangle([-5, -2], [15, 2])
cyl = dde.geometry.Disk([0, 0], 0.5)
geom = dde.geometry.CSGDifference(rect, cyl)
timedomain = dde.geometry.TimeDomain(0, T_max)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Funzione dummy (non serve la vera PDE per l'inferenza, ma l'oggetto data sì)
def dummy_pde(x, y):
    return [torch.tensor([0])]

data = dde.data.TimePDE(geomtime, dummy_pde, [], num_domain=100)
net = dde.nn.FNN([4] + [128] * 6 + [3], "tanh", "Glorot normal")
model = dde.Model(data, net)

# 2. CARICAMENTO MODELLO
model.compile("adam", lr=1e-3)
checkpoint_path = "model_parametric/pinn_karman-30000.pt"

if os.path.exists(checkpoint_path):
    model.restore(checkpoint_path)
    print("Modello caricato correttamente.")
else:
    print(f"Errore: {checkpoint_path} non trovato!")
    exit()

# 3. PREPARAZIONE GRIGLIA PER IL PLOT
# Ridotto a 200x100 per rendere lo slider fluido
x_lin = np.linspace(-5, 15, 200)
y_lin = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x_lin, y_lin)
pts = np.vstack((X.flatten(), Y.flatten())).T

fig, ax = plt.subplots(figsize=(12, 6))
plt.subplots_adjust(bottom=0.25)

# Stato iniziale
re_init = 100.0
t_init = 5.0

def get_vel(re_val, t_val):
    t_col = np.full((pts.shape[0], 1), t_val)
    re_col = np.full((pts.shape[0], 1), re_val)
    test_input = np.hstack((pts, t_col, re_col)).astype(np.float32)
    res = model.predict(test_input)
    # Calcolo magnitudo velocità: sqrt(u^2 + v^2)
    return np.sqrt(res[:, 0]**2 + res[:, 1]**2).reshape(X.shape)

# Plot iniziale
vel = get_vel(re_init, t_init)
cont = ax.contourf(X, Y, vel, levels=50, cmap="jet")
ax.set_aspect('equal') # CORREZIONE: Mantiene il cerchio tondo
fig.colorbar(cont, ax=ax, label="Velocità")

# Aggiunta degli Slider
ax_re = plt.axes([0.25, 0.1, 0.5, 0.03])
ax_t = plt.axes([0.25, 0.05, 0.5, 0.03])

s_re = Slider(ax_re, 'Reynolds', 20.0, 150.0, valinit=re_init)
s_t = Slider(ax_t, 'Tempo', 0.0, 10.0, valinit=t_init)

def update(val):
    re = s_re.val
    t = s_t.val
    new_vel = get_vel(re, t)
    
    # Pulizia e ridisegno
    ax.clear()
    ax.contourf(X, Y, new_vel, levels=50, cmap="jet")
    
    # Disegniamo il cilindro fisico (nero)
    theta = np.linspace(0, 2*np.pi, 100)
    ax.fill(0.5*np.cos(theta), 0.5*np.sin(theta), "k")
    
    ax.set_title(f"Re={re:.1f}, t={t:.1f}s")
    ax.set_aspect('equal')
    ax.set_xlim(-5, 15)
    ax.set_ylim(-2, 2)
    fig.canvas.draw_idle()

s_re.on_changed(update)
s_t.on_changed(update)

plt.show()
