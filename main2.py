"""
Главный файл с графическим интерфейсом
Запустите этот файл для начала симуляции
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from simulation_core import AsphalteneSimulation
from PIL import Image, ImageTk
import threading
import time

class AsphalteneSimulationGUI:
    """Графический интерфейс для симуляции агрегации асфальтенов"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Агрегация асфальтенов - Симуляция")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2c3e50')
        
        # Состояние симуляции
        self.simulation = None
        self.is_running = False
        self.is_paused = False
        self.speed = 10  # шагов между обновлениями
        self.simulation_thread = None
        
        # Результаты
        self.fractal_dimensions_history = []
        
        self._create_widgets()
        self._create_plots()
        
    def _create_widgets(self):
        """Создание всех виджетов управления"""
        
        # Основной контейнер
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Левая панель с управлением
        control_frame = tk.Frame(main_frame, bg='#34495e', width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)
        
        # Заголовок
        title = tk.Label(control_frame, text="АГРЕГАЦИЯ АСФАЛЬТЕНОВ", 
                        font=('Arial', 16, 'bold'), bg='#34495e', fg='white')
        title.pack(pady=15)
        
        # Режим симуляции
        mode_frame = tk.LabelFrame(control_frame, text="Режим симуляции", 
                                   bg='#34495e', fg='white', font=('Arial', 11, 'bold'))
        mode_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.mode_var = tk.StringVar(value="CCA")
        dla_radio = tk.Radiobutton(mode_frame, text="DLA (одна частица движется)", 
                                   variable=self.mode_var, value="DLA", bg='#34495e', fg='white',
                                   selectcolor='#34495e')
        cca_radio = tk.Radiobutton(mode_frame, text="CCA (все кластеры движутся)", 
                                   variable=self.mode_var, value="CCA", bg='#34495e', fg='white',
                                   selectcolor='#34495e')
        dla_radio.pack(anchor=tk.W, padx=5, pady=2)
        cca_radio.pack(anchor=tk.W, padx=5, pady=2)
        
        # Тип агрегации
        agg_frame = tk.LabelFrame(control_frame, text="Тип взаимодействия", 
                                  bg='#34495e', fg='white', font=('Arial', 11, 'bold'))
        agg_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.agg_type_var = tk.StringVar(value="sticky")
        sticky_radio = tk.Radiobutton(agg_frame, text="Липкие частицы", 
                                      variable=self.agg_type_var, value="sticky", 
                                      bg='#34495e', fg='white', selectcolor='#34495e')
        charged_radio = tk.Radiobutton(agg_frame, text="С электростатикой", 
                                       variable=self.agg_type_var, value="charged", 
                                       bg='#34495e', fg='white', selectcolor='#34495e')
        viscous_radio = tk.Radiobutton(agg_frame, text="Вязкая среда", 
                                       variable=self.agg_type_var, value="viscous", 
                                       bg='#34495e', fg='white', selectcolor='#34495e')
        sticky_radio.pack(anchor=tk.W, padx=5)
        charged_radio.pack(anchor=tk.W, padx=5)
        viscous_radio.pack(anchor=tk.W, padx=5)
        
        # Параметры
        params_frame = tk.LabelFrame(control_frame, text="Параметры среды", 
                                     bg='#34495e', fg='white', font=('Arial', 11, 'bold'))
        params_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Температура
        tk.Label(params_frame, text="Температура (K):", bg='#34495e', fg='white').grid(row=0, column=0, sticky=tk.W, padx=5)
        self.temp_var = tk.DoubleVar(value=300)
        temp_scale = tk.Scale(params_frame, from_=273, to=500, orient=tk.HORIZONTAL, 
                              variable=self.temp_var, bg='#34495e', fg='white', length=150)
        temp_scale.grid(row=0, column=1, padx=5)
        
        # Растворитель
        tk.Label(params_frame, text="Растворитель (0=плохой,1=хороший):", bg='#34495e', fg='white').grid(row=1, column=0, sticky=tk.W, padx=5)
        self.solvent_var = tk.DoubleVar(value=0.5)
        solvent_scale = tk.Scale(params_frame, from_=0, to=1, resolution=0.05, orient=tk.HORIZONTAL,
                                 variable=self.solvent_var, bg='#34495e', fg='white', length=150)
        solvent_scale.grid(row=1, column=1, padx=5)
        
        # Заряд
        tk.Label(params_frame, text="Сила заряда:", bg='#34495e', fg='white').grid(row=2, column=0, sticky=tk.W, padx=5)
        self.charge_var = tk.DoubleVar(value=0.5)
        charge_scale = tk.Scale(params_frame, from_=0, to=2, resolution=0.1, orient=tk.HORIZONTAL,
                                variable=self.charge_var, bg='#34495e', fg='white', length=150)
        charge_scale.grid(row=2, column=1, padx=5)
        
        # Вязкость
        tk.Label(params_frame, text="Вязкость:", bg='#34495e', fg='white').grid(row=3, column=0, sticky=tk.W, padx=5)
        self.viscosity_var = tk.DoubleVar(value=1.0)
        viscosity_scale = tk.Scale(params_frame, from_=0.1, to=10, resolution=0.1, orient=tk.HORIZONTAL,
                                   variable=self.viscosity_var, bg='#34495e', fg='white', length=150)
        viscosity_scale.grid(row=3, column=1, padx=5)
        
        # Количество частиц
        tk.Label(params_frame, text="Кол-во частиц:", bg='#34495e', fg='white').grid(row=4, column=0, sticky=tk.W, padx=5)
        self.n_particles_var = tk.IntVar(value=100)
        particles_scale = tk.Scale(params_frame, from_=20, to=500, orient=tk.HORIZONTAL,
                                   variable=self.n_particles_var, bg='#34495e', fg='white', length=150)
        particles_scale.grid(row=4, column=1, padx=5)
        
        # Скорость симуляции
        tk.Label(params_frame, text="Скорость:", bg='#34495e', fg='white').grid(row=5, column=0, sticky=tk.W, padx=5)
        self.speed_var = tk.IntVar(value=10)
        speed_scale = tk.Scale(params_frame, from_=1, to=50, orient=tk.HORIZONTAL,
                               variable=self.speed_var, bg='#34495e', fg='white', length=150)
        speed_scale.grid(row=5, column=1, padx=5)
        
        # Кнопки управления
        buttons_frame = tk.Frame(control_frame, bg='#34495e')
        buttons_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.start_btn = tk.Button(buttons_frame, text="▶ СТАРТ", command=self.start_simulation,
                                   bg='#27ae60', fg='white', font=('Arial', 12, 'bold'))
        self.start_btn.pack(fill=tk.X, pady=2)
        
        self.pause_btn = tk.Button(buttons_frame, text="⏸ ПАУЗА", command=self.pause_simulation,
                                   bg='#f39c12', fg='white', font=('Arial', 12, 'bold'), state=tk.DISABLED)
        self.pause_btn.pack(fill=tk.X, pady=2)
        
        self.stop_btn = tk.Button(buttons_frame, text="⏹ СТОП", command=self.stop_simulation,
                                  bg='#e74c3c', fg='white', font=('Arial', 12, 'bold'))
        self.stop_btn.pack(fill=tk.X, pady=2)
        
        # Кнопка анализа чувствительности
        sensitivity_btn = tk.Button(buttons_frame, text="📊 Анализ чувствительности", 
                                    command=self.sensitivity_analysis,
                                    bg='#3498db', fg='white', font=('Arial', 10, 'bold'))
        sensitivity_btn.pack(fill=tk.X, pady=5)
        
        # Кнопка экспорта
        export_btn = tk.Button(buttons_frame, text="💾 Экспорт GIF", command=self.export_gif,
                               bg='#9b59b6', fg='white', font=('Arial', 10, 'bold'))
        export_btn.pack(fill=tk.X, pady=5)
        
        # Информационная панель
        info_frame = tk.LabelFrame(control_frame, text="Информация", 
                                   bg='#34495e', fg='white', font=('Arial', 11, 'bold'))
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.cluster_count_label = tk.Label(info_frame, text="Кластеров: 0", 
                                            bg='#34495e', fg='white')
        self.cluster_count_label.pack(anchor=tk.W, padx=5, pady=2)
        
        self.fractal_label = tk.Label(info_frame, text="Фрактальная размерность: --", 
                                      bg='#34495e', fg='white')
        self.fractal_label.pack(anchor=tk.W, padx=5, pady=2)
        
        self.time_label = tk.Label(info_frame, text="Шаг: 0", 
                                   bg='#34495e', fg='white')
        self.time_label.pack(anchor=tk.W, padx=5, pady=2)
        
        # Правая панель с визуализацией
        viz_frame = tk.Frame(main_frame, bg='#2c3e50')
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # График симуляции
        self.sim_fig = Figure(figsize=(8, 8), facecolor='#2c3e50')
        self.sim_ax = self.sim_fig.add_subplot(111)
        self.sim_ax.set_facecolor('#1a252f')
        self.sim_ax.set_xlim(0, 500)
        self.sim_ax.set_ylim(0, 500)
        self.sim_ax.set_title("Агрегация асфальтенов", color='white')
        self.sim_ax.set_xlabel("X", color='white')
        self.sim_ax.set_ylabel("Y", color='white')
        self.sim_ax.tick_params(colors='white')
        
        self.canvas = FigureCanvasTkAgg(self.sim_fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Графики анализа
        analysis_frame = tk.Frame(viz_frame, bg='#2c3e50')
        analysis_frame.pack(fill=tk.X, pady=5)
        
        self.analysis_fig = Figure(figsize=(10, 3), facecolor='#2c3e50')
        
        # График фрактальной размерности
        self.fractal_ax = self.analysis_fig.add_subplot(131)
        self.fractal_ax.set_title("Фрактальная размерность", color='white', fontsize=9)
        self.fractal_ax.set_xlabel("Время", color='white', fontsize=8)
        self.fractal_ax.set_ylabel("Df", color='white', fontsize=8)
        self.fractal_ax.tick_params(colors='white')
        self.fractal_line, = self.fractal_ax.plot([], [], 'b-', linewidth=1)
        
        # График количества кластеров
        self.cluster_ax = self.analysis_fig.add_subplot(132)
        self.cluster_ax.set_title("Количество кластеров", color='white', fontsize=9)
        self.cluster_ax.set_xlabel("Время", color='white', fontsize=8)
        self.cluster_ax.set_ylabel("N", color='white', fontsize=8)
        self.cluster_ax.tick_params(colors='white')
        self.cluster_line, = self.cluster_ax.plot([], [], 'g-', linewidth=1)
        
        # График радиуса вращения
        self.radius_ax = self.analysis_fig.add_subplot(133)
        self.radius_ax.set_title("Радиус вращения (лог-лог)", color='white', fontsize=9)
        self.radius_ax.set_xlabel("log(Размер)", color='white', fontsize=8)
        self.radius_ax.set_ylabel("log(Rg)", color='white', fontsize=8)
        self.radius_ax.tick_params(colors='white')
        self.radius_line, = self.radius_ax.plot([], [], 'r-', linewidth=1)
        
        self.analysis_canvas = FigureCanvasTkAgg(self.analysis_fig, analysis_frame)
        self.analysis_canvas.get_tk_widget().pack(fill=tk.X)
        
    def _create_plots(self):
        """Инициализация пустых графиков"""
        self.sim_scatter = self.sim_ax.scatter([], [], c=[], s=10, alpha=0.7)
        self.canvas.draw()
        
    def start_simulation(self):
        """Запуск симуляции"""
        if self.is_running:
            return
            
        self.is_running = True
        self.is_paused = False
        
        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)
        
        # Создаём новую симуляцию
        self.simulation = AsphalteneSimulation(
            width=500,
            height=500,
            n_particles=self.n_particles_var.get(),
            temperature=self.temp_var.get(),
            solvent_solubility=self.solvent_var.get(),
            charge_strength=self.charge_var.get(),
            viscosity=self.viscosity_var.get(),
            mode=self.mode_var.get(),
            aggregation_type=self.agg_type_var.get(),
            seed=int(time.time())
        )
        
        # Запускаем в отдельном потоке
        self.simulation_thread = threading.Thread(target=self._run_simulation, daemon=True)
        self.simulation_thread.start()
        
    def _run_simulation(self):
        """Основной цикл симуляции в потоке"""
        while self.is_running:
            if not self.is_paused:
                for _ in range(self.speed_var.get()):
                    if not self.simulation.step():
                        self.is_running = False
                        self.root.after(0, self._simulation_finished)
                        break
                
                # Обновляем UI
                self.root.after(0, self._update_ui)
            time.sleep(0.05)
    
    def _update_ui(self):
        """Обновление интерфейса"""
        if not self.simulation:
            return
            
        # Получаем позиции частиц
        positions, colors = self.simulation.get_cluster_positions()
        
        if positions:
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            self.sim_scatter.set_offsets(np.c_[xs, ys])
            
            # Нормализация цветов
            rgb_colors = []
            for c in colors:
                rgb_colors.append(c)
            self.sim_scatter.set_color(rgb_colors)
        
        # Обновляем графики анализа
        if self.simulation.history['time']:
            times = self.simulation.history['time']
            
            # Фрактальная размерность
            if self.simulation.history['fractal_dimensions']:
                self.fractal_line.set_data(times[:len(self.simulation.history['fractal_dimensions'])], 
                                           self.simulation.history['fractal_dimensions'])
                self.fractal_ax.relim()
                self.fractal_ax.autoscale_view()
            
            # Количество кластеров
            self.cluster_line.set_data(times, self.simulation.history['n_clusters'])
            self.cluster_ax.relim()
            self.cluster_ax.autoscale_view()
            
            # Лог-лог график радиуса
            if len(self.simulation.clusters) > 5:
                sizes = [len(c.particles) for c in self.simulation.clusters]
                radii = [c.radius_of_gyration for c in self.simulation.clusters]
                log_sizes = np.log(sizes)
                log_radii = np.log(radii)
                self.radius_line.set_data(log_sizes, log_radii)
                self.radius_ax.relim()
                self.radius_ax.autoscale_view()
        
        self.canvas.draw()
        self.analysis_canvas.draw()
        
        # Обновляем информацию
        self.cluster_count_label.config(text=f"Кластеров: {len(self.simulation.clusters)}")
        fd = self.simulation.get_fractal_dimension()
        if fd:
            self.fractal_label.config(text=f"Фрактальная размерность: {fd:.3f}")
        self.time_label.config(text=f"Шаг: {self.simulation.time_step}")
        
    def pause_simulation(self):
        """Пауза"""
        self.is_paused = not self.is_paused
        self.pause_btn.config(text="▶ ПРОДОЛЖИТЬ" if self.is_paused else "⏸ ПАУЗА")
        
    def stop_simulation(self):
        """Остановка"""
        self.is_running = False
        self._simulation_finished()
        
    def _simulation_finished(self):
        """Действия при завершении симуляции"""
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self.pause_btn.config(text="⏸ ПАУЗА")
        
        # Финальная фрактальная размерность
        fd = self.simulation.get_fractal_dimension()
        messagebox.showinfo("Симуляция завершена", 
                           f"Фрактальная размерность агрегата: {fd:.3f}\n"
                           f"Ожидаемое значение для асфальтенов: 1.8-2.2")
        
    def sensitivity_analysis(self):
        """Анализ чувствительности - варьирование параметров"""
        messagebox.showinfo("Анализ чувствительности", 
                           "Запускается анализ...\n"
                           "Будут проверены:\n"
                           "- Температура (280-400K)\n"
                           "- Растворитель (0-1)\n"
                           "- Заряд (0-2)")
        
        # Запускаем в отдельном потоке
        thread = threading.Thread(target=self._run_sensitivity, daemon=True)
        thread.start()
        
    def _run_sensitivity(self):
        """Запуск анализа чувствительности"""
        results = []
        
        # Температура
        for temp in [280, 320, 360, 400]:
            sim = AsphalteneSimulation(
                n_particles=100,
                temperature=temp,
                mode='CCA',
                seed=42
            )
            for _ in range(200):
                sim.step()
            results.append({
                'param': 'temperature',
                'value': temp,
                'fd': sim.get_fractal_dimension()
            })
        
        # Растворитель
        for solv in [0.1, 0.3, 0.5, 0.7, 0.9]:
            sim = AsphalteneSimulation(
                n_particles=100,
                solvent_solubility=solv,
                mode='CCA',
                seed=42
            )
            for _ in range(200):
                sim.step()
            results.append({
                'param': 'solvent',
                'value': solv,
                'fd': sim.get_fractal_dimension()
            })
        
        # Выводим результаты
        self.root.after(0, lambda: self._show_sensitivity_results(results))
        
    def _show_sensitivity_results(self, results):
        """Показ результатов анализа"""
        fig = Figure(figsize=(10, 5), facecolor='white')
        
        # График температуры
        ax1 = fig.add_subplot(121)
        temp_data = [r for r in results if r['param'] == 'temperature']
        ax1.plot([r['value'] for r in temp_data], [r['fd'] for r in temp_data], 'ro-')
        ax1.set_xlabel("Температура (K)")
        ax1.set_ylabel("Фрактальная размерность")
        ax1.set_title("Влияние температуры")
        ax1.grid(True, alpha=0.3)
        
        # График растворителя
        ax2 = fig.add_subplot(122)
        solv_data = [r for r in results if r['param'] == 'solvent']
        ax2.plot([r['value'] for r in solv_data], [r['fd'] for r in solv_data], 'bo-')
        ax2.set_xlabel("Растворимость (0=плохой, 1=хороший)")
        ax2.set_ylabel("Фрактальная размерность")
        ax2.set_title("Влияние растворителя")
        ax2.grid(True, alpha=0.3)
        
        # Показываем окно с графиком
        window = tk.Toplevel(self.root)
        window.title("Анализ чувствительности")
        window.geometry("800x400")
        
        canvas = FigureCanvasTkAgg(fig, window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def export_gif(self):
        """Экспорт анимации в GIF"""
        if not self.simulation:
            messagebox.showwarning("Нет данных", "Сначала запустите симуляцию")
            return
            
        messagebox.showinfo("Экспорт", "Функция экспорта GIF требует установки библиотеки imageio\n"
                                      "Установите: pip install imageio imageio-ffmpeg")
        
    def run(self):
        """Запуск GUI"""
        self.root.mainloop()


if __name__ == "__main__":
    app = AsphalteneSimulationGUI()
    app.run()
