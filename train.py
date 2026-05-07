import os

# Specifichiamo il backend e l'architettura GPU PRIMA di importare deepxde
os.environ["DDE_BACKEND"] = "pytorch"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

import deepxde as dde
import numpy as np
import torch

# Creazione cartella per i salvataggi (evita errori se non esiste)
if not os.path.exists("model_parametric"):
    os.makedirs("model_parametric")

# --- 1. DOMINI PARAMETRICI E SPAZIO-TEMPO ---
Re_min, Re_max = 20.0, 150.0
T_max = 10.0

rect = dde.geometry.Rectangle([-5, -2], [15, 2])
cyl = dde.geometry.Disk([0, 0], 0.5)
geom = dde.geometry.CSGDifference(rect, cyl)
timedomain = dde.geometry.TimeDomain(0, T_max)

# Creiamo il dominio 3D base (x, y, t)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# --- 2. FISICA PARAMETRICA (Navier-Stokes) ---
def navier_stokes(x, y):
    # x è ora [x, y, t, Re]
    # y è [u, v, p]
    u, v, p = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    
    # Recuperiamo Re dall'input per calcolare nu dinamico
    re_input = x[:, 3:4]
    nu = 1.0 / re_input

    # Derivate
    du_x = dde.grad.jacobian(y, x, i=0, j=0)
    du_y = dde.grad.jacobian(y, x, i=0, j=1)
    du_t = dde.grad.jacobian(y, x, i=0, j=2)
    dv_x = dde.grad.jacobian(y, x, i=1, j=0)
    dv_y = dde.grad.jacobian(y, x, i=1, j=1)
    dv_t = dde.grad.jacobian(y, x, i=1, j=2)
    dp_x = dde.grad.jacobian(y, x, i=2, j=0)
    dp_y = dde.grad.jacobian(y, x, i=2, j=1)
    
    du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    du_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    dv_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    dv_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)

    # Equazioni
    continuity = du_x + dv_y
    mom_u = du_t + u*du_x + v*du_y + dp_x - nu*(du_xx + du_yy)
    mom_v = dv_t + u*dv_x + v*dv_y + dp_y - nu*(dv_xx + dv_yy)
    
    return [continuity, mom_u, mom_v]

# --- 3. CONDIZIONI AL CONTORNO E INIZIALI ---
def boundary_inlet(x, on_boundary):
    return on_boundary and np.isclose(x[0], -5)

def boundary_outlet(x, on_boundary):
    return on_boundary and np.isclose(x[0], 15)

def boundary_cylinder(x, on_boundary):
    # Passiamo solo x[:2] a cyl.on_boundary per ignorare t ed Re
    return on_boundary and cyl.on_boundary(x[:2])

# Inizializziamo le BC/IC
bc_u_in = dde.icbc.DirichletBC(geomtime, lambda x: 1.0, boundary_inlet, component=0)
bc_v_in = dde.icbc.DirichletBC(geomtime, lambda x: 0.0, boundary_inlet, component=1)
bc_u_cyl = dde.icbc.DirichletBC(geomtime, lambda x: 0.0, boundary_cylinder, component=0)
bc_v_cyl = dde.icbc.DirichletBC(geomtime, lambda x: 0.0, boundary_cylinder, component=1)
bc_p_out = dde.icbc.DirichletBC(geomtime, lambda x: 0.0, boundary_outlet, component=2)

ic_u = dde.icbc.IC(geomtime, lambda x: 1.0, lambda _, on_initial: on_initial, component=0)
ic_v = dde.icbc.IC(geomtime, lambda x: 0.0, lambda _, on_initial: on_initial, component=1)

# --- 4. CAMPIONAMENTO E INIEZIONE DEL REYNOLDS (La 4a dimensione) ---
data = dde.data.TimePDE(
    geomtime, navier_stokes,
    [bc_u_in, bc_v_in, bc_u_cyl, bc_v_cyl, bc_p_out, ic_u, ic_v],
    num_domain=50000, num_boundary=5000, num_initial=5000
)

# Funzione per aggiungere la colonna di Re ai punti campionati
def inject_re(pts):
    re_samples = np.random.uniform(Re_min, Re_max, (pts.shape[0], 1))
    return np.hstack((pts, re_samples)).astype(np.float32)

# Hack: Sovrascriviamo i punti del dominio per farli diventare [x, y, t, Re]
data.train_x_all = inject_re(data.train_x_all)
data.train_x_bc = inject_re(data.train_x_bc)
data.train_x = inject_re(data.train_x)
data.test_x = inject_re(data.test_x)

# --- 5. RETE NEURALE E TRAINING ---
# Rete con 4 input -> 6 layer da 128 -> 3 output
net = dde.nn.FNN([4] + [128] * 6 + [3], "sin", "Glorot normal")
model = dde.Model(data, net)

# Callback per i salvataggi
checkpointer = dde.callbacks.ModelCheckpoint(
    "model_parametric/pinn_karman", period=2000, verbose=1
)

# Pesi per forzare la rete a rispettare di più il cilindro e la continuità
# Ordine: [Cont, MomU, MomV, InletU, InletV, CylU, CylV, OutP, ICU, ICV]
loss_weights = [20, 10, 10, 5, 5, 100, 100, 2, 5, 5]

print(f"\n--- INIZIO TRAINING SU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} ---")

# Fase 1: Adam (Sgrossatura)
model.compile("adam", lr=1e-3, loss_weights=loss_weights)
losshistory, train_state = model.train(iterations=100000, callbacks=[checkpointer])

# Fase 2: L-BFGS (Rifinitura di precisione)
model.compile("L-BFGS", loss_weights=loss_weights)
losshistory, train_state = model.train()

# --- 6. DIAGNOSTICA ---
# Salva il grafico della discesa dell'errore (indispensabile per documentare)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# --- 7. ESPORTAZIONE PER PARAVIEW ---
def export_results(re_value, filename):
    x_pts = np.linspace(-5, 15, 200) # Alta risoluzione X
    y_pts = np.linspace(-2, 2, 100)  # Alta risoluzione Y
    t_pts = np.linspace(0, T_max, 40) # Frame per il video in ParaView
    
    # Creiamo la griglia 3D
    X, Y, T = np.meshgrid(x_pts, y_pts, t_pts)
    test_pts = np.vstack((X.flatten(), Y.flatten(), T.flatten())).T
    
    # Aggiungiamo il Reynolds scelto come costante per questa esportazione
    re_column = np.full((test_pts.shape[0], 1), re_value)
    test_pts_parametric = np.hstack((test_pts, re_column)).astype(np.float32)
    
    # Facciamo predire il risultato alla rete addestrata
    pred = model.predict(test_pts_parametric)
    
    # Uniamo coordinate e risultati per salvare il CSV
    output = np.hstack((test_pts_parametric, pred))
    np.savetxt(filename, output, header="x,y,t,Re,u,v,p", delimiter=",", comments='')
    print(f"Esportato con successo: {filename}")

print("\n--- GENERAZIONE RISULTATI PER PARAVIEW ---")
export_results(20.0, "results_Re20_steady.csv")
export_results(80.0, "results_Re80_transition.csv")
export_results(150.0, "results_Re150_unsteady.csv")
print("Simulazione completata con successo!")
