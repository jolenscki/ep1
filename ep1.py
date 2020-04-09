################################
######## NUMERICO - EP1 ########
################################
# Joao Rodrigo Windisch Olenscki
# NUSP 10773224
# Luca Rodrigues Miguel
# NUSP 10705655

# Bibliotecas
import matplotlib.pyplot as plt
import numpy
import os
import sys
import time
import datetime
import math
import pandas as pd # importando pandas para melhor visualizar as matrizes

# Parametros estaticos do matplotlib
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

# Funcoes de miscelânia, uteis para alguns fins
def create_folder(folder_list, path = os.getcwd()):
    '''
    funcao que cria pastas em um certo diretório
    @parameters:
    - folder_list: lista, cada elemento e uma string contendo o nome da pasta a ser criada
    - path: string, caminho absoluto anterior as pastas a serem criadas
    -- default_value: os.getcwd(), caminho até a pasta onde o programa está rodando
    @output:
    - None
    '''
    for folder_name in folder_list:
        try:
            os.mkdir(os.path.join(path, str(folder_name)))
            print("Foi!!!")
        except FileExistsError:
            print("Folder {} already exists in this directory".format(folder_name))
        except TypeError:
            print("TypeError, path = {}, folder_name = {}".format(path, folder_name))
            
# Definindo funções iniciais
def get_M_parameter(T, lambda_val, N):
    '''
    funcao para calcular o parametro M
    @parameters:
    - T: float, constante de tempo T
    - lambda_val: float, constante do problema
    - N: inteiro, numero de divisoes feitas na barra
    @output:
    - M: inteiro, numero de divisoes no tempo
    '''
    M =T*(N**2)/lambda_val
    return int(M)

def create_m_n_matrix(M, N):
    '''
    funcao para criar uma matriz nula para ser preenchida futuramente
    @parameters:
    - M: inteiro, numero de divisoes no tempo
    - N: inteiro, numero de divisoes na barra
    @output:
    - u: array ((M+1)x(N+1)), array de zeros, de dimensao (M+1)x(N+1) 
    '''
    u = numpy.zeros((M+1, N+1))
    return u

def get_time_array(M, T):
    '''
    funcao para criar uma array do tempo, sera usada de forma auxiliar para aplicacao de outras funcoes
    @parameters:
    - M: inteiro, numero de divisoes no tempo
    - T: float, constante de tempo T
    @return:
    - time_array: array (1x(M+1)), contem todos os instantes de tempo
    -- example: [0*(T/M), 1*(T/M), ... , (M-1)*(T/M), M*(T/M)]
    '''
    time_array = numpy.linspace(0, T, num = M+1)
    return time_array

def get_space_array(N):
    '''
    funcao para criar uma array do espaco, sera usada de forma auxiliar para aplicacao de outras funcoes
    @parameters:
    - N: inteiro, numero de divisoes na barra
    @return:
    - space_array: array (1x(N+1)), contem todas as posicoes da barra
    -- example: [0*(1/N), 1*(1/N), ... , (N-1)*(1/N), N*(1/N)]
    '''
    space_array = numpy.linspace(0, 1, num = N+1)
    return space_array

def apply_boundary_conditions(u, u0, g1, g2):
    '''
    funcao que aplica condicoes de contorno na matriz de temperaturas
    @parameters:
    - u: array ((M+1)x(N+1)), matriz de temperaturas (suposta nula no início)
    - u0: array (1x(N+1)), temperaturas na barra em t=0
    - g1 = array(1x(M+1)), temperaturas na barra para x=0
    - g2 = array(1x(M+1)), temperaturas na barra para x=1
    @return:
    -u: array ((M+1)x(N+1)), matriz de temperaturas com 3 bordas ajustadas pelas condicoes iniciais
    '''
    u[0] = u0
    u[:,0] = g1
    u[:,-1] = g2
    return u

def plot_temperatures(T, lambda_val, N, delta_time, space_array, temperature_matrix, title, path, filename):
    '''
    funcao que plota um gráfico de temperaturas para cada 0.1 segundos (1/10 do tempo total)
    @parameters:
    - T: float, constante de tempo T
    - lambda_val: float, constante do problema
    - N: inteiro, numero de divisoes na barra
    - delta_time: float, 
    - space_array:
    - temperature_matrix:
    - title:
    - path:
    - filename:
    '''
    L = temperature_matrix.shape[0]
    for time_step in range(11):
        temperature_array = temperature_matrix[(L//10)*time_step]
        plt.plot(space_array, temperature_array, label = r'${}$ segundos'.format(time_step/10))
        
    ax = plt.gca()
    title_string = r'{} em função da posição para certas séries temporais'.format(title)
    subtitle_string = r'$T = {},\; \lambda = {},\; N = {},\;$ Tempo de execução$:\; {}$ segundos'.format(T, lambda_val, N, delta_time)
    plt.suptitle(title_string, y=1.0, fontsize = 18)
    ax.set_title(subtitle_string, fontsize = 14)
    ax.set_xlabel(r'Posição na barra ($x$)')
    ax.set_ylabel(r'{}'.format(title))
    ax.legend(loc='right', bbox_to_anchor=(1.25, 0.5))
    
    savedir = os.path.join(path, filename + '.png')
    plt.savefig(savedir, dpi = 300, bbox_inches="tight")
    plt.close()
    return ax
    
def plot_heatmap(T, lambda_val, N, delta_time, space_array, time_array, temperature_matrix, title, path, filename):
    title_string = r'Mapa de {}'.format(title) + r' para a barra inteira em todos os $\mathit{ticks}$ de tempo'
    subtitle_string = r'$T = {},\; \lambda = {},\; N = {},\;$ Tempo de execução$:\; {}$ segundos'.format(T, lambda_val, N, delta_time)
    
    x_min = space_array[0]
    x_max = space_array[-1]
    t_min = time_array[0]
    t_max = time_array[-1]
    
    plt.imshow(temperature_matrix, cmap = 'jet', aspect = 'auto', extent = [x_min, x_max, t_max, t_min])
    plt.colorbar()
    ax = plt.gca()
    plt.suptitle(title_string, y=1.0, fontsize = 18)
    ax.set_title(subtitle_string, fontsize = 14)
    ax.set_xlabel(r'Posição na barra ($x$)')
    ax.set_ylabel(r'Tempo ($t$)')
    
    
    savedir = os.path.join(path, filename + '.png')
    plt.savefig(savedir, dpi = 300, bbox_inches="tight")
    plt.close()
    return ax

def add_initial_ending_zeros(array):
    final_array = numpy.zeros(1)
    final_array = numpy.concatenate((final_array, array))
    final_array = numpy.concatenate((final_array, numpy.zeros(1)))
    return final_array

def get_D_matrix(N, lambda_val):
    a = numpy.diagflat(lambda_val * numpy.ones(N), 1)
    b = numpy.diagflat((1-2*lambda_val) * numpy.ones(N+1))
    c = numpy.diagflat(lambda_val * numpy.ones(N), -1)
    D_matrix = numpy.matrix(a+b+c)
    D_matrix[:,0] = numpy.zeros((N+1, 1))
    D_matrix[:,-1] = numpy.zeros((N+1, 1))
    return D_matrix

def get_e_matrix(time_array, space_array):
    s_1, t_1 = numpy.meshgrid(space_array, time_array)
    return 10*t_1*(s_1**2)*(s_1 - 1)

def f(space_array, k, T, M):
    return 10*(space_array**2)*(space_array - 1) - 60*space_array*((k*T)/M) + 20*((k*T)/M)

def get_f_matrix(space_array, k, T, M):
    f_array = numpy.apply_along_axis(f, 0, space_array[1:-1], k, T, M)
    return numpy.matrix(add_initial_ending_zeros(f_array))

def apply_equation_11(T, lambda_val, u, space_array):
    M = u.shape[0] - 1
    N = u.shape[1] - 1
    D = get_D_matrix(N, lambda_val)
    for k,_ in enumerate(u[1:], start = 1):
        u[k, 1:N] = numpy.asarray(u[k-1].dot(D) + (T/M)*(get_f_matrix(space_array, k, T, M)))[0,1:N].reshape(N-1,)
    return u

def get_error(u, e):
    error_matrix = numpy.subtract(u, e)
    return error_matrix

def run_test_vectorized(T, lambda_val, N):
    
    start_time = time.time()
    M = get_M_parameter(T, lambda_val, N)

    zeros = create_m_n_matrix(M,N)

    time_array = get_time_array(M, T)
    space_array = get_space_array(N)

    u0 = numpy.zeros((space_array.shape))
    g1 = numpy.zeros((time_array.shape))
    g2 = numpy.zeros((time_array.shape))

    u = apply_boundary_conditions(zeros, u0, g1, g2)

    zeros = 0

    u = apply_equation_11(T, lambda_val, u, space_array)
    exact_matrix = get_e_matrix(time_array, space_array)
    error_matrix = get_error(u, exact_matrix)
    
    end_time = time.time()
    delta_time = round(end_time - start_time, 3)
    return M, delta_time, time_array, space_array, u, exact_matrix, error_matrix

def main():
    T = 1
    lambda_list = [0.25, 0.5]
    N_list = [10, 20, 40, 80, 160, 320, 640]
    main_dir = os.getcwd()
    create_folder(lambda_list, path = main_dir)
    for lambda_val in lambda_list:
        lambda_dir = os.path.join(main_dir, str(lambda_val))
        create_folder(N_list, path = lambda_dir)
        for N in N_list:
            n_dir = os.path.join(lambda_dir, str(N))
            M, delta_time, time_array, space_array, temperature_matrix, exact_matrix, error_matrix = run_test_vectorized(T, lambda_val, N)
            
            plot_temperatures(T, lambda_val, N, delta_time, space_array, temperature_matrix, 'Temperatura', n_dir, 'time_series')
            plot_heatmap(T, lambda_val, N, delta_time, space_array, time_array, temperature_matrix, 'Temperatura', n_dir, 'heatmap')
            
            plot_temperatures(T, lambda_val, N, delta_time, space_array, exact_matrix, 'Solução exata', n_dir, 'exact_time_series')
            plot_heatmap(T, lambda_val, N, delta_time, space_array, time_array, exact_matrix, 'Solução exata', n_dir, 'exact_heatmap')

            plot_temperatures(T, lambda_val, N, delta_time, space_array, error_matrix, 'Erro', n_dir, 'error_time_series')
            plot_heatmap(T, lambda_val, N, delta_time, space_array, time_array, error_matrix, 'Erro', n_dir, 'error_heatmap')

            
main()


