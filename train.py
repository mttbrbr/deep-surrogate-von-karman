import os
os.environ["DDE_BACKEND"] = "pytorch"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

import deepxde as dde
import numpy as np
import torch

# --- 1. DOMINI PARAMETRICI ---
Re_min, Re_max = 20.0, 150.0
T_max = 10.0

rect = dde.geometry.Rectangle([-5, -2], [15, 2])
cyl = dde.geometry.Disk([0, 0], 0.5)
geom = dde.geometry.CSGDifference(rect, cyl)
timedomain = dde.geometry.TimeDomain(0, T_max)

# Trasformiamo il tutto in un dominio 4D: (x, y, t, Re)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# --- 2. FISICA CON PESI ADATTIVI ---
def navier_stokes(x, y):
    u, v, p = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    re_input = x[:, 3:4] # Quarto input
    nu = 1.0 / re_input

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

    continuity = du_x + dv_y
    mom_u = du_t + u*du_x + v*du_y + dp_x - nu*(du_xx + du_yy)
    mom_v = dv_t + u*dv_x + v*dv_y + dp_y - nu*(dv_xx + dv_yy)
    
    return [continuity, mom_u, mom_v]

# --- 3. CAMPIONAMENTO PARAMETRICO ---
# Funzione per iniettare Re casuali nei punti del dominio
def modify_domain_with_re(x):
    re_samples = np.random.uniform(Re_min, Re_max, (len(x), 1))
    return np.hstack((x, re_samples))

# Applichiamo il campionamento
data = dde.data.TimePDE(
    geomtime, navier_stokes,
    [bc_u_in, bc_v_in, bc_u_cyl, bc_v_cyl, bc_p_out, ic_u, ic_v],
    num_domain=30000, num_boundary=5000, num_initial=5000
)
# TRUCCO: Iniettiamo il range di Re nel dataset
data.train_x_all = modify_domain_with_re(data.train_x_all)

# --- 4. STRUTTURA RETE E PESI NON UNIFORMI ---
net = dde.nn.FNN([4] + [128] * 6 + [3], "tanh", "Glorot normal")
model = dde.Model(data, net)

# PESI: [Cont, MomU, MomV, InletU, InletV, CylU, CylV, OutP, ICU, ICV]
# Diamo peso 2 alla continuità e 10 al cilindro (no-slip fondamentale)
loss_weights = [2, 1, 1, 5, 5, 10, 10, 1, 2, 2]

model.compile("adam", lr=1e-3, loss_weights=loss_weights)

# --- 5. TRAINING E OTTIMIZZAZIONE ---
print(f"Inizio training su: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Fase 1: Adam
model.compile("adam", lr=1e-3, loss_weights=loss_weights)
# Salviamo lo storico per i grafici
losshistory, train_state = model.train(iterations=30000, callbacks=[checkpointer])

# Fase 2: L-BFGS (Rifinitura di precisione)
model.compile("L-BFGS", loss_weights=loss_weights)
losshistory, train_state = model.train()

# --- 6. DIAGNOSTICA E GRAFICI (Il pezzo mancante) ---
# Questo comando genera in automatico i grafici della Loss e li salva come PDF/PNG
dde.saveplot(losshistory, train_state, issave=True, isplot=True)


# --- 7. ESPORTAZIONE PER PARAVIEW ---
def export_results(re_value, filename):
    x_pts = np.linspace(-5, 15, 200) # Aumentata la risoluzione X
    y_pts = np.linspace(-2, 2, 100)  # Aumentata la risoluzione Y
    t_pts = np.linspace(0, T_max, 40) # Più frame temporali
    
    X, Y, T = np.meshgrid(x_pts, y_pts, t_pts)
    test_pts = np.vstack((X.flatten(), Y.flatten(), T.flatten())).T
    
    # Aggiungiamo il parametro Re fisso per questa esportazione
    re_column = np.full((test_pts.shape[0], 1), re_value)
    test_pts_parametric = np.hstack((test_pts, re_column))
    
    # Predizione
    pred = model.predict(test_pts_parametric)
    
    # Salvataggio
    output = np.hstack((test_pts_parametric, pred))
    np.savetxt(filename, output, header="x,y,t,Re,u,v,p", delimiter=",", comments='')
    print(f"Esportato con successo: {filename}")

# Generiamo i 3 scenari di test
print("Generazione dei risultati per ParaView in corso...")
export_results(20.0, "results_Re20_steady.csv")
export_results(80.0, "results_Re80_transition.csv")
export_results(150.0, "results_Re150_unsteady.csv")
print("Simulazione completata!")
