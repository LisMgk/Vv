import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from simulation_core import AsphalteneSimulation
import threading
import time

class AsphalteneSimulationGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Агрегация асфальтенов")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        self.simulation = None
        self.is_running = False
        self.is_paused = False
        
        self._create_widgets()
        self._create_plots()
        
    def _create_widgets(self):
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        control_frame = tk.Frame(main_frame, bg='#34495e', width=250)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)
        
        tk.Label(control_frame, text="АГРЕГАЦИЯ АСФАЛЬТЕНОВ", 
                font=('Arial', 14, 'bold'), bg='#34495e', fg='white').pack(pady=10)
        
        # Режим
        self.mode_var = tk.StringVar(value="CCA")
        tk.Label(control_frame, text="Режим", bg='#34495e', fg='white').pack()
        tk.Radiobutton(control_frame, text="DLA", variable=self.mode_var, value="DLA", bg='#34495e', fg='white').pack()
        tk.Radiobutton(control_frame, text="CCA", variable=self.mode_var, value="CCA", bg='#34495e', fg='white').pack()
        
        # Тип агрегации
        self.agg_var = tk.StringVar(value="sticky")
        tk.Label(control_frame, text="Тип взаимодействия", bg='#34495e', fg='white').pack(pady=(10,0))
        tk.Radiobutton(control_frame, text="Липкие", variable=self.agg_var, value="sticky", bg='#34495e', fg='white').pack()
        tk.Radiobutton(control_frame, text="С зарядом", variable=self.agg_var, value="charged", bg='#34495e', fg='white').pack()
        tk.Radiobutton(control_frame, text="Вязкие", variable=self.agg_var, value="viscous", bg='#34495e', fg='white').pack()
        
        # Параметры
        tk.Label(control_frame, text="Температура", bg='#34495e', fg='white').pack(pady=(10,0))
        self.temp_var = tk.DoubleVar(value=300)
        tk.Scale(control_frame, from_=273, to=500, orient=tk.HORIZONTAL, variable=self.temp_var, bg='#34495e', fg='white').pack()
        
        tk.Label(control_frame, text="Растворитель", bg='#34495e', fg='white').pack()
        self.solvent_var = tk.DoubleVar(value=0.5)
        tk.Scale(control_frame, from_=0, to=1, resolution=0.05, orient=tk.HORIZONTAL, variable=self.solvent_var, bg='#34495e', fg='white').pack()
        
        tk.Label(control_frame, text="Заряд", bg='#34495e', fg='white').pack()
        self.charge_var = tk.DoubleVar(value=0.5)
        tk.Scale(control_frame, from_=0, to=2, resolution=0.1, orient=tk.HORIZONTAL, variable=self.charge_var, bg='#34495e', fg='white').pack()
        
        tk.Label(control_frame, text="Кол-во частиц", bg='#34495e', fg='white').pack()
        self.n_part_var = tk.IntVar(value=100)
        tk.Scale(control_frame, from_=20, to=300, orient=tk.HORIZONTAL, variable=self.n_part_var, bg='#34495e', fg='white').pack()
        
        # Кнопки
        self.start_btn = tk.Button(control_frame, text="СТАРТ", command=self.start_simulation, bg='#27ae60', fg='white')
        self.start_btn.pack(fill=tk.X, pady=5)
        
        self.pause_btn = tk.Button(control_frame, text="ПАУЗА", command=self.pause_simulation, bg='#f39c12', fg='white', state=tk.DISABLED)
        self.pause_btn.pack(fill=tk.X, pady=5)
        
        self.stop_btn = tk.Button(control_frame, text="СТОП", command=self.stop_simulation, bg='#e74c3c', fg='white')
        self.stop_btn.pack(fill=tk.X, pady=5)
        
        # Инфо
        self.info_label = tk.Label(control_frame, text="Кластеров: 0\nФрактальная размерность: --", bg='#34495e', fg='white')
        self.info_label.pack(pady=10)
        
        # Область для графиков
        viz_frame = tk.Frame(main_frame, bg='#2c3e50')
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.fig = plt.Figure(figsize=(8, 8), facecolor='#2c3e50')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#1a252f')
        self.ax.set_xlim(0, 500)
        self.ax.set_ylim(0, 500)
        self.ax.set_title("Агрегация асфальтенов", color='white')
        self.ax.tick_params(colors='white')
        
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.scatter = self.ax.scatter([], [], s=10, alpha=0.7)
        
    def _create_plots(self):
        pass
    
    def start_simulation(self):
        if self.is_running:
            return
        
        self.is_running = True
        self.is_paused = False
        
        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)
        
        self.simulation = AsphalteneSimulation(
            width=500,
            height=500,
            n_particles=self.n_part_var.get(),
            temperature=self.temp_var.get(),
            solvent_solubility=self.solvent_var.get(),
            charge_strength=self.charge_var.get(),
            mode=self.mode_var.get(),
            aggregation_type=self.agg_var.get(),
            seed=int(time.time())
        )
        
        self.simulation_thread = threading.Thread(target=self._run_simulation, daemon=True)
        self.simulation_thread.start()
    
    def _run_simulation(self):
        while self.is_running:
            if not self.is_paused:
                for _ in range(5):
                    if not self.simulation.step():
                        self.is_running = False
                        break
                self.root.after(0, self._update_ui)
            time.sleep(0.05)
    
    def _update_ui(self):
        if not self.simulation:
            return
        
        positions, colors = self.simulation.get_cluster_positions()
        if positions:
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            self.scatter.set_offsets(np.c_[xs, ys])
            
        self.canvas.draw()
        
        fd = self.simulation.get_fractal_dimension()
        self.info_label.config(text=f"Кластеров: {len(self.simulation.clusters)}\nФрактальная размерность: {fd:.3f}")
        
        if len(self.simulation.clusters) <= 1:
            self._simulation_finished()
    
    def pause_simulation(self):
        self.is_paused = not self.is_paused
        self.pause_btn.config(text="ПРОДОЛЖИТЬ" if self.is_paused else "ПАУЗА")
    
    def stop_simulation(self):
        self.is_running = False
        self._simulation_finished()
    
    def _simulation_finished(self):
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.pause_btn.config(text="ПАУЗА")
        
        fd = self.simulation.get_fractal_dimension()
        messagebox.showinfo("Завершено", f"Фрактальная размерность: {fd:.3f}\n(ожидается 1.8-2.2)")
    
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = AsphalteneSimulationGUI()
    app.run()
