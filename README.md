# Water-Rocket-Flight-Simulation
A high-accuracy Python water-rocket flight simulator with optimization, full physics engine, Tkinter GUI, trajectory visualization, angle–pressure sweep analysis, and configurable rocket/environment parameters. Designed for rocketry competitions.

🚀 Water Rocket Simulation & Optimization Toolkit

High-accuracy physics-based simulator with Tkinter GUI, optimizer, trajectory plots, and configurable rocket/environment parameters.

This project is a complete engineering-grade toolkit for simulating the physics of water-rockets, visualizing their flight paths, optimizing launch parameters, and analyzing performance under different environmental conditions.

Built for rocketry competitions, undergraduate engineering research, and practical avionics experimentation.

✨ Features
🔥 Accurate Physics Engine

Full RK4 integrator

Water phase + air phase thrust modeling

Variable air pressure using adiabatic expansion

Drag force using Cd and frontal area

Wind effects included

Realistic mass variation (water + compressed air)

🛰 Configurable Rocket & Environment

The GUI lets you modify:

Ambient pressure

Ambient temperature

Air density

Bottle volume

Nozzle diameter

Rocket body diameter

Drag coefficient

Empty mass

Wind speed

All constants update instantly using Apply Constants.

📊 Tkinter Graphical Interface

Real-time trajectory visualization (Matplotlib embedded)

Table view of simulation datapoints

Log console (progress, analysis, optimizer output)

Switch between multiple graphs:

Trajectory

Angle vs Range

Wind Drift Sensitivity

Max Altitude vs Angle

🎯 Auto-Optimization Mode

Input a target distance → the simulator brute-forces the best combination of:

Launch angle

Pressure (PSI)

Water volume (mL)

Plotted trajectory + recommended optimal configuration.

📁 CSV Import/Export

Load external trajectory data

Save simulation results to CSV

⚙️ High Accuracy Mode

Two simulation modes:

Normal: uses original integrator

High (Refined RK4): user-defined dt for precision research simulations

🖥️ Installation
1. Clone the repository:
git clone https://github.com/YOUR_USERNAME/water-rocket-simulator.git
cd water-rocket-simulator

2. Install dependencies:
pip install numpy pandas matplotlib


Tkinter comes pre-installed with Python on Windows/Linux/Mac. No additional GUI libraries needed.

▶️ Running the Application

Run:

python simulation_gui.py


This opens the Tkinter interface.

📚 How It Works
1. Physics Model

The simulator models:

Pressure evolution using adiabatic expansion

Water thrust (momentum of expelled mass)

Air thrust once water is exhausted

Gravity + drag

Changing mass through both phases

Wind-modified relative velocity

2. Numerical Integration

Uses 4th-order Runge-Kutta (RK4) for smooth and accurate trajectories.
High-accuracy mode allows configurable time-step.

3. Optimization Engine

Brute-force search across:

Angles from 30°–65°

Pressure from 60–120 PSI

Water volume from 300–1000 mL

Finds configuration closest to your target distance.
