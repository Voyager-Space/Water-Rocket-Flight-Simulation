import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import io
import time
import math
import matplotlib.pyplot as plt

G = 9.81              
P_ATM = 101325.0      
RHO_WATER = 1000.0    
RHO_AIR = 1.225       
GAMMA = 1.4           

M_EMPTY = 0.15        
V_BOTTLE = 0.002      
A_NOZZLE = np.pi * (0.021 / 2)**2  
A_BODY   = np.pi * (0.105 / 2)**2  
CD = 0.3              

def get_derivatives(state, P_init, V_air_init, m_air_init, wind_speed):
    x, y, vx, vy, m_water, m_air = state

    vol_water = m_water / RHO_WATER
    vol_air = V_BOTTLE - vol_water
    if vol_air < 1e-6: vol_air = 1e-6
    
    P_internal = P_init * (V_air_init / vol_air)**GAMMA
    if m_water <= 0:
        P_internal = P_internal * (m_air / m_air_init)**GAMMA

    pressure_diff = P_internal - P_ATM
    

    thrust = 0.0
    dm_water = 0.0
    dm_air = 0.0
    
    if pressure_diff > 0:
        if m_water > 0: 
            v_exit = math.sqrt(2 * pressure_diff / RHO_WATER)
            mdot = RHO_WATER * A_NOZZLE * v_exit
            thrust = mdot * v_exit
            dm_water = -mdot
        else: 
            thrust = pressure_diff * A_NOZZLE
            dm_air = -0.05 * (pressure_diff / P_ATM)

    vx_rel = vx - wind_speed
    vy_rel = vy
    v_rel = math.sqrt(vx_rel**2 + vy_rel**2)
    
    drag = 0.5 * RHO_AIR * v_rel**2 * CD * A_BODY

    theta = math.atan2(vy, vx) if (vx**2+vy**2) > 0 else 0
    Fx = thrust * math.cos(theta)
    Fy = thrust * math.sin(theta)
    
    if v_rel > 0:
        Fx -= drag * (vx_rel / v_rel)
        Fy -= drag * (vy_rel / v_rel)
        
    Fy -= (M_EMPTY + m_water + m_air) * G
    
    total_mass = M_EMPTY + m_water + m_air
    ax = Fx / total_mass
    ay = Fy / total_mass
    
    return np.array([vx, vy, ax, ay, dm_water, dm_air])


def run_simulation(angle, pressure_psi, water_ml, wind_speed=0.0):
    dt = 0.01

    P_init = (pressure_psi * 6894.76) + P_ATM
    vol_water = water_ml * 1e-6
    vol_air_init = V_BOTTLE - vol_water
    m_water = vol_water * RHO_WATER
    m_air_init = (P_init * vol_air_init) / (287 * 293)
    
    state = np.array([0.0, 0.1, 0.0, 0.0, m_water, m_air_init])
    
    rad = math.radians(angle)
    state[2] = 1.0 * math.cos(rad)
    state[3] = 1.0 * math.sin(rad)
    
    trajectory_x = []
    trajectory_y = []
    
    t = 0
    while t < 20:
        trajectory_x.append(state[0])
        trajectory_y.append(state[1])
        if state[1] < 0: break
        
        args = (P_init, vol_air_init, m_air_init, wind_speed)
        k1 = get_derivatives(state, *args)
        k2 = get_derivatives(state + k1*dt/2, *args)
        k3 = get_derivatives(state + k2*dt/2, *args)
        k4 = get_derivatives(state + k3*dt, *args)
        
        state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        if state[4] < 0: state[4] = 0
        if state[5] < 0: state[5] = 0
        t += dt
        
    return trajectory_x, trajectory_y


def generate_analysis_graphs(pressure_psi, water_ml, optimal_angle=None):
    """
    Generates 3 separate windows analyzing Angle performance.
    """
    print(f"Generatng Analytical Graphs for {pressure_psi} PSI / {water_ml} mL...")
    
    angles = range(10, 86, 2)
    ranges_calm = []
    ranges_wind = []
    altitudes = []
    TEST_WIND = 5.0 
    
    for ang in angles:
        x1, y1 = run_simulation(ang, pressure_psi, water_ml, wind_speed=0.0)
        ranges_calm.append(x1[-1])
        altitudes.append(max(y1))
        
        x2, y2 = run_simulation(ang, pressure_psi, water_ml, wind_speed=TEST_WIND)
        ranges_wind.append(x2[-1])

    plt.figure(f"Analysis 1: Range ({pressure_psi} PSI, {water_ml} mL)", figsize=(7, 5))
    plt.plot(angles, ranges_calm, 'b-o', markersize=3)
    if optimal_angle:
        plt.axvline(optimal_angle, color='g', linestyle='--', label=f"Optimum: {optimal_angle}°")
    plt.title("Launch Angle vs Total Range (No Wind)")
    plt.xlabel("Angle (Deg)"); plt.ylabel("Distance (m)")
    plt.grid(True); plt.legend()
    plt.show(block=False)
    plt.figure("Analysis 2: Wind Drift Effect", figsize=(7, 5))
    plt.plot(angles, ranges_calm, 'b--', label="No Wind")
    plt.plot(angles, ranges_wind, 'r-', label=f"With {TEST_WIND} m/s Wind")
    plt.title(f"Wind Drift Sensitivity")
    plt.xlabel("Angle (Deg)"); plt.ylabel("Distance (m)")
    plt.grid(True); plt.legend()
    plt.show(block=False)
    plt.figure("Analysis 3: Max Altitude", figsize=(7, 5))
    plt.plot(angles, altitudes, 'purple')
    plt.title("Launch Angle vs Peak Altitude")
    plt.xlabel("Angle (Deg)"); plt.ylabel("Height (m)")
    plt.grid(True)
    plt.show(block=False)

def find_best_launch(target_dist):
    print(f"Optimizing for {target_dist} meters (Brute Force)...")
    best_error = 9999
    best_config = (0, 0, 0)
    for angle in range(30, 65, 5):
        for psi in range(60, 125, 10):
            for ml in range(300, 1000, 100):
                xs, ys = run_simulation(angle, psi, ml)
                error = abs(xs[-1] - target_dist)
                if error < best_error:
                    best_error = error
                    best_config = (angle, psi, ml)
                    
    return best_config

def simulate(pressure: float, angle_deg: float, water_ml: float, wind_speed: float, dt: float=0.01, input_df=None):
    try:
        acc = app.accuracy_choice.get()
    except Exception:
        acc = 'Normal'

    if acc == 'Normal':
        xs, ys = run_simulation(angle_deg, pressure, water_ml, wind_speed=wind_speed)
        if len(xs) == 0:
            return pd.DataFrame({'t':[], 'x':[], 'y':[]})
        times = np.arange(len(xs)) * 0.01
        df = pd.DataFrame({'t': times, 'x': np.array(xs), 'y': np.array(ys)})
        return df
    else:
        P_init = (pressure * 6894.76) + P_ATM
        vol_water = water_ml * 1e-6
        vol_air_init = V_BOTTLE - vol_water
        m_water = vol_water * RHO_WATER
        m_air_init = (P_init * vol_air_init) / (287 * 293)

        rad = math.radians(angle_deg)
        state = np.array([0.0, 0.1, 1.0 * math.cos(rad), 1.0 * math.sin(rad), m_water, m_air_init])

        t = 0.0
        dt_local = float(app.dt_var.get()) if hasattr(app, 'dt_var') else dt
        xs = []
        ys = []
        max_t = 20.0
        while t < max_t:
            xs.append(state[0])
            ys.append(state[1])
            if state[1] < 0:
                break
            k1 = get_derivatives(state, P_init, vol_air_init, m_air_init, wind_speed)
            k2 = get_derivatives(state + 0.5*dt_local*k1, P_init, vol_air_init, m_air_init, wind_speed)
            k3 = get_derivatives(state + 0.5*dt_local*k2, P_init, vol_air_init, m_air_init, wind_speed)
            k4 = get_derivatives(state + dt_local*k3, P_init, vol_air_init, m_air_init, wind_speed)
            state = state + (dt_local/6.0) * (k1 + 2*k2 + 2*k3 + k4)
            if state[4] < 0: state[4] = 0
            if state[5] < 0: state[5] = 0
            t += dt_local
        times = np.arange(len(xs)) * dt_local
        df = pd.DataFrame({'t': times, 'x': np.array(xs), 'y': np.array(ys)})
        return df

class SimApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Water Rocket Simulator - Tkinter')
        self.geometry('1280x760')

        self.sim_thread = None
        self.sim_queue = queue.Queue()
        self.input_df = None
        self.trajectory = None
        self.predicted_config = None 

        self._build_ui()
        self._periodic_check()

    def _build_ui(self):
        ctrl_outer = ttk.Frame(self)
        ctrl_outer.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        ctrl_row1 = ttk.Frame(ctrl_outer)
        ctrl_row1.pack(side=tk.TOP, fill=tk.X)
        ctrl_row2 = ttk.Frame(ctrl_outer)
        ctrl_row2.pack(side=tk.TOP, fill=tk.X, pady=(6,0))

        ttk.Label(ctrl_row1, text='Angle (deg):').grid(row=0, column=0, sticky=tk.W)
        self.angle_var = tk.DoubleVar(value=45.0)
        ttk.Entry(ctrl_row1, textvariable=self.angle_var, width=8).grid(row=0, column=1, padx=6)

        ttk.Label(ctrl_row1, text='Pressure (PSI):').grid(row=0, column=2, sticky=tk.W)
        self.pressure_var = tk.DoubleVar(value=80.0)
        ttk.Entry(ctrl_row1, textvariable=self.pressure_var, width=8).grid(row=0, column=3, padx=6)

        ttk.Label(ctrl_row1, text='Water (mL):').grid(row=0, column=4, sticky=tk.W)
        self.water_var = tk.DoubleVar(value=600.0)
        ttk.Entry(ctrl_row1, textvariable=self.water_var, width=8).grid(row=0, column=5, padx=6)

        ttk.Label(ctrl_row1, text='Wind (m/s):').grid(row=0, column=6, sticky=tk.W)
        self.wind_var = tk.DoubleVar(value=0.0)
        ttk.Entry(ctrl_row1, textvariable=self.wind_var, width=8).grid(row=0, column=7, padx=6)

        btn_padx = 4
        ttk.Button(ctrl_row1, text='Load CSV (optional)', command=self.load_csv).grid(row=0, column=8, padx=btn_padx)
        ttk.Button(ctrl_row1, text='Simulate Input Config', command=self.run_simulation).grid(row=0, column=9, padx=btn_padx)
        ttk.Button(ctrl_row1, text='Simulate Predicted Config', command=self.run_predicted_simulation).grid(row=0, column=10, padx=btn_padx)
        ttk.Button(ctrl_row1, text='Save Trajectory CSV', command=self.save_csv).grid(row=0, column=11, padx=btn_padx)
        ttk.Button(ctrl_row1, text='Clear', command=self.clear).grid(row=0, column=12, padx=btn_padx)

        ttk.Button(ctrl_row2, text='Analyze Angle Performance (embedded)', command=self._open_analysis_embedded).grid(row=0, column=0, padx=6)
        ttk.Button(ctrl_row2, text='Predict Best Config', command=self._predict_prompt).grid(row=0, column=1, padx=6)


        ttk.Label(ctrl_row2, text='Predicted:').grid(row=0, column=2, sticky=tk.W, padx=(12,0))
        self.pred_label = ttk.Label(ctrl_row2, text='—')
        self.pred_label.grid(row=0, column=3, columnspan=2, sticky=tk.W)

        ttk.Label(ctrl_row2, text='Graph:').grid(row=0, column=5, sticky=tk.W, padx=(12,0))
        self.graph_choice = ttk.Combobox(ctrl_row2, values=['Trajectory','Angle vs Range','Wind Drift','Max Altitude'], state='readonly', width=16)
        self.graph_choice.current(0)
        self.graph_choice.grid(row=0, column=6, padx=6)
        ttk.Button(ctrl_row2, text='Show Graph', command=self._show_selected_graph).grid(row=0, column=7, padx=6)
        ttk.Label(ctrl_row2, text='Accuracy:').grid(row=0, column=8, sticky=tk.W, padx=(12,0))
        self.accuracy_choice = ttk.Combobox(ctrl_row2, values=['Normal','High (refined RK4)'], state='readonly', width=18)
        self.accuracy_choice.current(0)
        self.accuracy_choice.grid(row=0, column=9, padx=6)
        ttk.Label(ctrl_row2, text='dt (s):').grid(row=0, column=10, sticky=tk.W)
        self.dt_var = tk.DoubleVar(value=0.005)
        ttk.Entry(ctrl_row2, textvariable=self.dt_var, width=8).grid(row=0, column=11, padx=6)


        ctrl_row3 = ttk.Frame(ctrl_outer)
        ctrl_row3.pack(side=tk.TOP, fill=tk.X, pady=(6,0))
        env_frame = ttk.LabelFrame(ctrl_row3, text='Environment (external)')
        env_frame.pack(side=tk.LEFT, padx=6)
        ttk.Label(env_frame, text='Ambient Pressure (Pa):').grid(row=0, column=0, sticky=tk.W)
        self.ambient_pressure_var = tk.DoubleVar(value=P_ATM)
        ttk.Entry(env_frame, textvariable=self.ambient_pressure_var, width=12).grid(row=0, column=1, padx=6)
        ttk.Label(env_frame, text='Ambient Temp (K):').grid(row=1, column=0, sticky=tk.W)
        self.ambient_temp_var = tk.DoubleVar(value=293.0)
        ttk.Entry(env_frame, textvariable=self.ambient_temp_var, width=12).grid(row=1, column=1, padx=6)
        ttk.Label(env_frame, text='Air density (kg/m³):').grid(row=2, column=0, sticky=tk.W)
        self.rho_air_var = tk.DoubleVar(value=RHO_AIR)
        ttk.Entry(env_frame, textvariable=self.rho_air_var, width=12).grid(row=2, column=1, padx=6)
        rock_frame = ttk.LabelFrame(ctrl_row3, text='Rocket details (constants)')
        rock_frame.pack(side=tk.LEFT, padx=6)
        ttk.Label(rock_frame, text='Nozzle diameter (m):').grid(row=0, column=0, sticky=tk.W)
        self.nozzle_dia_var = tk.DoubleVar(value=0.021)
        ttk.Entry(rock_frame, textvariable=self.nozzle_dia_var, width=10).grid(row=0, column=1, padx=6)
        ttk.Label(rock_frame, text='Bottle volume (m³):').grid(row=1, column=0, sticky=tk.W)
        self.bottle_vol_var = tk.DoubleVar(value=V_BOTTLE)
        ttk.Entry(rock_frame, textvariable=self.bottle_vol_var, width=10).grid(row=1, column=1, padx=6)
        ttk.Label(rock_frame, text='Empty mass (kg):').grid(row=2, column=0, sticky=tk.W)
        self.empty_mass_var = tk.DoubleVar(value=M_EMPTY)
        ttk.Entry(rock_frame, textvariable=self.empty_mass_var, width=10).grid(row=2, column=1, padx=6)
        ttk.Label(rock_frame, text='Body diameter (m):').grid(row=3, column=0, sticky=tk.W)
        self.body_dia_var = tk.DoubleVar(value=0.105)
        ttk.Entry(rock_frame, textvariable=self.body_dia_var, width=10).grid(row=3, column=1, padx=6)
        ttk.Label(rock_frame, text='Cd:').grid(row=4, column=0, sticky=tk.W)
        self.cd_var = tk.DoubleVar(value=CD)
        ttk.Entry(rock_frame, textvariable=self.cd_var, width=10).grid(row=4, column=1, padx=6)
        ttk.Button(ctrl_row3, text='Apply constants', command=self.apply_constants).pack(side=tk.LEFT, padx=12)
        left_frame = ttk.Frame(self)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=6)

        self.fig = Figure(figsize=(8,6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('x (m)')
        self.ax.set_ylabel('y (m)')
        self.ax.grid(True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        right_frame = ttk.Frame(self, width=360)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=8, pady=6)

        ttk.Label(right_frame, text='Trajectory (first rows)').pack(anchor=tk.W)
        self.tree = ttk.Treeview(right_frame, columns=('t','x','y'), show='headings', height=18)
        self.tree.heading('t', text='t (s)')
        self.tree.heading('x', text='x (m)')
        self.tree.heading('y', text='y (m)')
        self.tree.pack(fill=tk.BOTH, expand=False)

        ttk.Label(right_frame, text='Log').pack(anchor=tk.W, pady=(8,0))
        self.log_text = tk.Text(right_frame, height=12, width=45)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def log(self, msg: str):
        self.log_text.insert(tk.END, f"{msg}")
        self.log_text.see(tk.END)

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[('CSV files','*.csv'),('All files','*.*')])
        if not path:
            return
        try:
            self.input_df = pd.read_csv(path)
            self.log(f'Loaded CSV: {path} (rows={len(self.input_df)})')
        except Exception as e:
            messagebox.showerror('Error', f'Failed to load CSV: {e}')

    def apply_constants(self):
        global P_ATM, RHO_AIR, V_BOTTLE, M_EMPTY, A_NOZZLE, A_BODY, CD
        try:
            P_ATM = float(self.ambient_pressure_var.get())
            ambient_temp = float(self.ambient_temp_var.get())
            RHO_AIR = float(self.rho_air_var.get())
            V_BOTTLE = float(self.bottle_vol_var.get())
            M_EMPTY = float(self.empty_mass_var.get())
            nozzle_dia = float(self.nozzle_dia_var.get())
            body_dia = float(self.body_dia_var.get())
            CD = float(self.cd_var.get())

            A_NOZZLE = math.pi * (nozzle_dia / 2.0)**2
            A_BODY = math.pi * (body_dia / 2.0)**2

            self.log('Applied constants: environment and rocket details updated')
        except Exception as e:
            messagebox.showerror('Error', f'Failed to apply constants: {e}')

    def save_csv(self):
        if self.trajectory is None:
            messagebox.showinfo('Info','No trajectory to save')
            return
        path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV','*.csv')])
        if not path:
            return
        try:
            self.trajectory.to_csv(path, index=False)
            self.log(f'Saved trajectory to {path}')
        except Exception as e:
            messagebox.showerror('Error', f'Failed to save CSV: {e}')

    def clear(self):
        self.input_df = None
        self.trajectory = None
        self.predicted_config = None
        self.pred_label.config(text='—')
        self.ax.cla()
        self.ax.set_xlabel('x (m)')
        self.ax.set_ylabel('y (m)')
        self.ax.grid(True)
        self.canvas.draw()
        for i in self.tree.get_children():
            self.tree.delete(i)
        self.log('Cleared')

    def run_simulation(self):
        angle = float(self.angle_var.get())
        pressure = float(self.pressure_var.get())
        water = float(self.water_var.get())
        wind = float(self.wind_var.get())

        if self.sim_thread and self.sim_thread.is_alive():
            messagebox.showinfo('Info', 'Simulation already running')
            return
        self.sim_thread = threading.Thread(target=self._sim_worker, args=(pressure, angle, water, wind, self.input_df))
        self.sim_thread.daemon = True
        self.sim_thread.start()
        self.log('Simulation (input config) started...')

    def run_predicted_simulation(self):
        if not self.predicted_config:
            messagebox.showinfo('Info', 'No predicted config available — run "Predict Best Config" first')
            return
        angle, psi, ml = self.predicted_config
        wind = float(self.wind_var.get())
        if self.sim_thread and self.sim_thread.is_alive():
            messagebox.showinfo('Info', 'Simulation already running')
            return
        self.sim_thread = threading.Thread(target=self._sim_worker, args=(psi, angle, ml, wind, self.input_df))
        self.sim_thread.daemon = True
        self.sim_thread.start()
        self.log('Simulation (predicted config) started...')

    def _sim_worker(self, pressure, angle, water, wind, input_df):
        try:
            traj = simulate(pressure=pressure, angle_deg=angle, water_ml=water, wind_speed=wind, dt=0.01, input_df=input_df)
            self.sim_queue.put(('done', traj))
        except Exception as e:
            self.sim_queue.put(('error', str(e)))

    def _periodic_check(self):
        try:
            while True:
                item = self.sim_queue.get_nowait()
                if item[0] == 'done':
                    traj = item[1]
                    self.trajectory = traj
                    self._on_sim_done(traj)
                elif item[0] == 'error':
                    self.log(f'Error during simulation: {item[1]}')
                    messagebox.showerror('Simulation error', item[1])
        except queue.Empty:
            pass
        self.after(100, self._periodic_check)

    def _on_sim_done(self, traj: pd.DataFrame):
        self.log('Simulation finished')
        self.ax.cla()
        self.ax.plot(traj['x'], traj['y'], '-', linewidth=2)
        self.ax.set_xlabel('x (m)')
        self.ax.set_ylabel('y (m)')
        self.ax.set_title(f"Trajectory — Range: {traj['x'].values[-1]:.2f} m")
        self.ax.grid(True)
        self.canvas.draw()

        for i in self.tree.get_children():
            self.tree.delete(i)
        max_rows = min(len(traj), 30)
        for idx in range(max_rows):
            row = traj.iloc[idx]
            self.tree.insert('', 'end', values=(f"{row['t']:.3f}", f"{row['x']:.3f}", f"{row['y']:.3f}"))

    def _predict_prompt(self):
        def do_predict():
            try:
                target = float(entry.get())
            except:
                messagebox.showerror('Error', 'Enter a valid number')
                return
            win.destroy()
            self.log(f'Starting prediction (optimize) for {target} m...')
            threading.Thread(target=self._run_optimizer, args=(target,), daemon=True).start()

        win = tk.Toplevel(self)
        win.title('Predict best config')
        ttk.Label(win, text='Target distance (m):').pack(side=tk.LEFT, padx=6, pady=8)
        entry = ttk.Entry(win)
        entry.pack(side=tk.LEFT, padx=6, pady=8)
        ttk.Button(win, text='Start', command=do_predict).pack(side=tk.LEFT, padx=6, pady=8)

    def _run_optimizer(self, target):
        ang, psi, ml = find_best_launch(target)
        self.predicted_config = (ang, psi, ml)
        self.after(0, lambda: self._on_prediction_done(ang, psi, ml))

    def _on_prediction_done(self, ang, psi, ml):
        self.log(f'Prediction finished — Best config: {ang}°, {psi} PSI, {ml} mL')
        self.pred_label.config(text=f'{ang}° | {psi} PSI | {ml} mL')
        self.angle_var.set(ang)
        self.pressure_var.set(psi)
        self.water_var.set(ml)

    def _open_analysis_embedded(self):
        pressure = float(self.pressure_var.get())
        water = float(self.water_var.get())
        threading.Thread(target=self._compute_and_plot_analysis, args=(pressure, water), daemon=True).start()
        self.log('Launched embedded analysis (background thread)')

    def _compute_and_plot_analysis(self, pressure, water):
        angles = list(range(10, 86, 2))
        ranges_calm = []
        ranges_wind = []
        altitudes = []
        TEST_WIND = 5.0
        total = len(angles)
        for i, ang in enumerate(angles):
            self.log(f'Analysis: sim {i+1}/{total} — angle {ang}°')
            x1, y1 = run_simulation(ang, pressure, water, wind_speed=0.0)
            ranges_calm.append(x1[-1])
            altitudes.append(max(y1))
            x2, y2 = run_simulation(ang, pressure, water, wind_speed=TEST_WIND)
            ranges_wind.append(x2[-1])

        self.after(0, lambda: self._plot_analysis_embedded(angles, ranges_calm, ranges_wind, altitudes))

    def _plot_analysis_embedded(self, angles, ranges_calm, ranges_wind, altitudes):
        self.ax.cla()
        choice = self.graph_choice.get()
        if choice == 'Trajectory':
            if self.trajectory is not None:
                self.ax.plot(self.trajectory['x'], self.trajectory['y'], '-', linewidth=2)
                self.ax.set_title('Trajectory (last run)')
            else:
                self.ax.text(0.5,0.5,'Run a simulation first', ha='center')
        elif choice == 'Angle vs Range':
            self.ax.plot(angles, ranges_calm, 'b-o', markersize=4)
            self.ax.set_title('Angle vs Range (No Wind)')
            self.ax.set_xlabel('Angle (deg)'); self.ax.set_ylabel('Range (m)')
        elif choice == 'Wind Drift':
            self.ax.plot(angles, ranges_calm, 'b--', label='No Wind')
            self.ax.plot(angles, ranges_wind, 'r-', label='With 5 m/s Wind')
            self.ax.set_title('Wind Drift Sensitivity')
            self.ax.set_xlabel('Angle (deg)'); self.ax.set_ylabel('Range (m)')
            self.ax.legend()
        elif choice == 'Max Altitude':
            self.ax.plot(angles, altitudes, 'purple')
            self.ax.set_title('Angle vs Peak Altitude')
            self.ax.set_xlabel('Angle (deg)'); self.ax.set_ylabel('Altitude (m)')
        self.ax.grid(True)
        self.canvas.draw()

    def _show_selected_graph(self):
        choice = self.graph_choice.get()
        if choice == 'Trajectory':
            if self.trajectory is None:
                messagebox.showinfo('Info', 'Run a simulation first to show trajectory')
                return
            self._plot_analysis_embedded([], [], [], [])
        else:
            self._open_analysis_embedded()

if __name__ == '__main__':
    app = SimApp()
    app.mainloop()

