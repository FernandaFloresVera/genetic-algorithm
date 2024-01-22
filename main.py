import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from genetic_algorithm import optimize
import sympy as sp

window = tk.Tk()  
window.title("Algoritmo Genético")
window.geometry("1200x600")
window.resizable(False, False)
button_font = ("Arial", 12, "bold")
frame_bg_color = "#d9d9d9"

top_frame = tk.Frame(window)
top_frame.pack(side=tk.TOP, padx=10)

frame_parametros = tk.LabelFrame(top_frame, text="Parámetros",bg=frame_bg_color)  
frame_parametros.grid(row=0, column=0, padx=20, pady=30, sticky="ew")

expression_label = tk.Label(frame_parametros, text="Funcion:",bg=frame_bg_color)
expression_label.grid(row=0, column=0, sticky="w")
expression_entry = tk.Entry(frame_parametros)  
expression_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")

resolution_label = tk.Label(frame_parametros, text="Resolucion inicial:",bg=frame_bg_color)
resolution_label.grid(row=1, column=0, sticky="w")
resolution_entry = tk.Entry(frame_parametros)  
resolution_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")

generations_label = tk.Label(frame_parametros, text="Generaciones:",bg=frame_bg_color)
generations_label.grid(row=2, column=0, sticky="w")
generations_entry = tk.Entry(frame_parametros)  
generations_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")

min_interval_label = tk.Label(frame_parametros, text="Min. intervalo:",bg=frame_bg_color)
min_interval_label.grid(row=0, column=2, sticky="w")
min_interval_entry = tk.Entry(frame_parametros)  
min_interval_entry.grid(row=0, column=3, padx=10, sticky="w")

max_interval_label = tk.Label(frame_parametros, text="Max. intervalo:",bg=frame_bg_color)
max_interval_label.grid(row=1, column=2, sticky="w")
max_interval_entry = tk.Entry(frame_parametros)  
max_interval_entry.grid(row=1, column=3, padx=10, sticky="w")

initial_population_label = tk.Label(frame_parametros, text="Poblacion inicial:",bg=frame_bg_color)
initial_population_label.grid(row=2, column=2, sticky="w")
initial_population_entry = tk.Entry(frame_parametros) 
initial_population_entry.grid(row=2, column=3, padx=10, pady=10, sticky="w")

max_population_label = tk.Label(frame_parametros, text="Poblacion maxima:",bg=frame_bg_color)
max_population_label.grid(row=0, column=4, sticky="w")
max_population_entry = tk.Entry(frame_parametros)  
max_population_entry.grid(row=0, column=5, padx=10, pady=10, sticky="w")

crossover_prob_label = tk.Label(frame_parametros, text="Prob. de cruza:",bg=frame_bg_color)
crossover_prob_label.grid(row=1, column=4, sticky="w")
crossover_prob_entry = tk.Entry(frame_parametros)  
crossover_prob_entry.grid(row=1, column=5, padx=10, pady=10, sticky="w")

mutation_prob_label = tk.Label(frame_parametros, text="Prob. de mutación:",bg=frame_bg_color)
mutation_prob_label.grid(row=2, column=4, sticky="w")
mutation_prob_entry = tk.Entry(frame_parametros)  
mutation_prob_entry.grid(row=2, column=5, padx=10, pady=10, sticky="w")

maximize = tk.StringVar()
maximize.set("Maximizar")
max_menu = tk.OptionMenu(frame_parametros, maximize, "Maximizar", "Minimizar")  
max_menu.grid(row=0, column=7, padx=10, sticky="w")


def evaluate_function_values(func, values):
    x = sp.symbols("x")
    results = np.array([])
    for value in values:
        expression = sp.sympify(func)
        result = expression.subs(x, value)
        results = np.append(results, result)
    return results


def print_graphs(
        expression,
        initial_resolution,
        generations_number,
        min_range,
        max_range,
        initial_population,
        max_population,
        crossover_probability,
        mutation_probability,
        minimize,
):  
    statistics, population = optimize(
        expression,
        initial_resolution,
        generations_number,
        min_range,
        max_range,
        initial_population,
        max_population,
        crossover_probability,
        mutation_probability,
        minimize,
    ) 
    statistics = np.array(statistics)  

    population = np.array(population)

    for widget in frame_plot.winfo_children():  
        widget.destroy()

  
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.grid(True)
    ax.set_title("Aptitud del fitness")
    ax.set_xlabel("Generación")
    ax.set_ylabel("Fitness")

    generations_number = generations_number + 1

    generations_number = np.arange(0, generations_number, 1)
    best = np.array([])
    worst = np.array([])
    average = np.array([])

    for i in range(len(statistics)):  
        best = np.append(best, statistics[i]["best"]["f(x)"])
        worst = np.append(worst, statistics[i]["worst"]["f(x)"])
        average = np.append(average, statistics[i]["average"])

    ax.plot(generations_number, best, label="Mejor")  
    ax.plot(generations_number, average, label="Promedio")
    ax.plot(generations_number, worst, label="Peor")

    ax.legend() 

    canvas = FigureCanvasTkAgg(fig, master=frame_plot)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

    
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.grid(True)
    ax.set_title("Población final")
    ax.set_xlabel("x")
    ax.set_ylabel("Fitness")

    x = np.array([])
    y = np.array([])

    max_fitness = max(population, key=lambda x: x["f(x)"])["f(x)"]
    min_fitness = min(population, key=lambda x: x["f(x)"])["f(x)"]

    max_x = np.array([])
    max_y = np.array([])
    min_x = np.array([])
    min_y = np.array([])

    x_graph = np.arange(min_range, max_range, 0.1)
    y_graph = evaluate_function_values(expression, x_graph)

    for i in range(len(population)):
        if min_range <= population[i]["x"] <= max_range:
            if population[i]["f(x)"] == max_fitness:
                max_x = np.append(max_x, population[i]["x"])
                max_y = np.append(max_y, population[i]["f(x)"])
            elif population[i]["f(x)"] == min_fitness:
                min_x = np.append(min_x, population[i]["x"])
                min_y = np.append(min_y, population[i]["f(x)"])
            x = np.append(x, population[i]["x"])
            y = np.append(y, population[i]["f(x)"])

    if minimize:
        ax.scatter(min_x, min_y, label="Mejor individuo", zorder=3)
        ax.scatter(x, y, label="Individuos", zorder=2)
        ax.scatter(max_x, max_y, label="Peor individuo", zorder=3)
    else:
        ax.scatter(max_x, max_y, label="Mejor individuo", zorder=3)
        ax.scatter(x, y, label="Individuos", zorder=2)
        ax.scatter(min_x, min_y, label="Peor individuo", zorder=3)
    ax.plot(x_graph, y_graph, label="Funcion", color="black", zorder=1)
    ax.legend()

    canvas = FigureCanvasTkAgg(fig, master=frame_plot)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)


def execute_algorithm():
    expression = str(expression_entry.get())
    initial_resolution = float(resolution_entry.get())
    generations_number = int(generations_entry.get())
    min_range = float(min_interval_entry.get())
    max_range = float(max_interval_entry.get())
    initial_population = int(initial_population_entry.get())
    max_population = int(max_population_entry.get())
    crossover_probability = float(crossover_prob_entry.get())
    individual_mutation_probability = float(mutation_prob_entry.get())
    minimize = maximize.get() == "Minimizar"

    print_graphs(
        expression,
        initial_resolution,
        generations_number,
        min_range,
        max_range,
        initial_population,
        max_population,
        crossover_probability,
        individual_mutation_probability,
        minimize,
    )

execute_button = tk.Button(frame_parametros, text="Ejecutar", command=lambda: execute_algorithm(),
    bg="#4CAF50", fg="white", font=button_font)
execute_button.grid(row=2, column=6, padx=10, pady=10, sticky="w")

bottom_frame = tk.Frame(window)  
bottom_frame.pack(side=tk.BOTTOM, padx=10)

frame_plot = tk.Frame(bottom_frame)
frame_plot.pack()

window.mainloop() 
