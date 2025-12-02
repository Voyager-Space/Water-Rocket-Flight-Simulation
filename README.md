# Water-Rocket-Flight-Simulation
A high-accuracy Python water-rocket flight simulator with optimization, full physics engine, Tkinter GUI, trajectory visualization, angle–pressure sweep analysis, and configurable rocket/environment parameters. Designed for rocketry competitions.

# Water-Rocket-Flight-Simulation 🚀

A high-accuracy Python water-rocket flight simulator with optimization, full physics engine, Tkinter GUI, trajectory visualization, angle–pressure sweep analysis, and configurable rocket/environment parameters — designed for rocketry competitions, undergraduate research, and practical avionics experimentation.

---

## ✨ Features

### 🔧 Physics Engine
- Realistic thrust modeling: water phase + compressed-air phase  
- Variable internal pressure (adiabatic expansion)  
- Drag force using rocket cross-section area & drag coefficient  
- Mass variation as water/air exhaust  
- Optional wind effects on trajectory  

### 📐 Configurable Rocket & Environment
Via GUI you can set:  
- Ambient pressure, temperature, air density  
- Bottle volume, nozzle diameter, rocket body diameter, empty mass, drag coefficient  
- Wind speed  

Constants are applied globally — convenient for comparing different bottle designs or atmospheric conditions.

### 🖥️ GUI Interface (Tkinter + Matplotlib)
- Embedded real-time trajectory plot  
- Table view of simulation data (time, position)  
- Log console for progress, analysis & optimizer messages  
- Graph selector:  
  - Trajectory  
  - Angle vs Range  
  - Wind-Drift vs Range  
  - Max Altitude vs Launch Angle  

### 🎯 Optimization Mode
Enter a **target distance (m)** ➝ brute-force search over launch parameters:  
- Launch angle  
- Initial pressure (PSI)  
- Water volume (mL)

Automatically returns best configuration and displays trajectory.

### 🧪 High-Accuracy Mode
Choose between:  
- **Normal mode** — original integrator (fast)  
- **High-accuracy mode** — RK4 integrator with user-controlled time-step for more precise simulation  

### 📄 CSV Import / Export
- Load external trajectory data (CSV)  
- Save simulated trajectories to CSV for post-processing  

---

## 🛠️ Installation & Setup

```
bash'''
git clone https://github.com/Voyager-Space/Water-Rocket-Flight-Simulation.git
cd Water-Rocket-Flight-Simulation
pip install numpy pandas matplotlib'''


Water volume from 300–1000 mL

Finds configuration closest to your target distance.
