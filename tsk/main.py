import matplotlib.pyplot as plt
import numpy as np

# Tarefa
# Utilizando o Método do gradiente mostrados nos slides e vídeos do tópico 3,
# aproximar a saída para f(x) = x^2

def generate_points():
    return np.linspace(-10, 10, 1000)

def create_base_function():
    x = generate_points()
    return (x**2, x)

def approximate_function():
    x = generate_points()
    return ((2*x)**2, x)

def plot_function(x, y, color, label):
    plt.plot(x, y, label=label, color=color)

def plot_base_function():
    y,x = create_base_function()
    plot_function(x, y, 'r', 'f(x) = x²')

def plot_approximate_function():
    y, x= approximate_function()
    plot_function(x, y, 'b', 'Curva estimada ajustada')    

def configure_plot():
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc='upper right')
    plt.title('Curva da função f(x)=x² e Curva estimada ajustada')

def show_plot():
    plt.show()

def show():
    configure_plot()
    show_plot()

def main():
    plot_base_function()
    plot_approximate_function()
    show()

if __name__ == '__main__':
    main()

