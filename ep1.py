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
from termcolor import colored

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
            print("Folder", colored("{}".format(folder_name), "blue"), "criado")
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
    funcao que plota um grafico de temperaturas para cada 0.1 segundos (1/10 do tempo total)
    @parameters:
    - T: float, constante de tempo T
    - lambda_val: float, constante do problema
    - N: inteiro, numero de divisoes na barra
    - delta_time: float, tempo total decorrido para a execucao do programa, em segundos
    - space_array: array (1x(N+1)), contem todas as posicoes da barra
    - temperature_matrix: array ((M+1)x(N+1)), matriz de temperaturas
    - title: string, titulo a ser colocado no grafico
    - path: string, caminho ate o local onde o arquivo sera salvo
    - filename: string, nome com o qual o arquivo sera salvo
    @output:
    - ax: axis (objeto de eixo do mpl)
    '''
    L = temperature_matrix.shape[0]
    for time_step in range(11):
        temperature_array = temperature_matrix[(L//10)*time_step]
        plt.plot(space_array, temperature_array, label = r'${} s$'.format(time_step/10))
        
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
    
def plot_error_array(T, lambda_val, N, delta_time, space_array, error_array, path, filename):
    '''
    funcao que plota o grafico da array do erro no ultimo instante t = T
    - T: float, constante de tempo T
    - lambda_val: float, constante do problema
    - N: inteiro, numero de divisoes na barra
    - delta_time: float, tempo total decorrido para a execucao do programa, em segundos
    - space_array: array (1x(N+1)), contem todas as posicoes da barra
    - error_matrix: array (1x(N+1)), array de erros associados a aproximacao
    - path: string, caminho ate o local onde o arquivo sera salvo
    - filename: string, nome com o qual o arquivo sera salvo
    @output:
    - ax: axis (objeto de eixo do mpl)
    '''
    plt.plot(space_array, error_array)
    
    ax = plt.gca()
    title_string = r'Erro em função da posição para o instante $t = T$'
    subtitle_string = r'$T = {},\; \lambda = {},\; N = {},\;$ Tempo total de execução$:\; {}$ segundos'.format(T, lambda_val, N, delta_time)    
    plt.suptitle(title_string, y=1.0, fontsize = 18)
    ax.set_title(subtitle_string, fontsize = 14)
    ax.set_xlabel(r'Posição na barra ($x$)')
    ax.set_ylabel(r'Magnitude do erro')
    savedir = os.path.join(path, filename + '.png')
    plt.savefig(savedir, dpi = 300, bbox_inches="tight")
    plt.close()
    return ax
    
def plot_heatmap(T, lambda_val, N, delta_time, space_array, time_array, temperature_matrix, title, path, filename):
    '''
    funcao que plota um heatmap da temperatura para todos os valores de espaco e tempo
    @parameters:
    - T: float, constante de tempo T
    - lambda_val: float, constante do problema
    - N: inteiro, numero de divisoes na barra
    - delta_time: float, tempo total decorrido para a execucao do programa, em segundos
    - space_array: array (1x(N+1)), contem todas as posicoes da barra
    - time_array: array (1x(M+1)), contem todos os instantes de tempo
    - temperature_matrix: array ((M+1)x(N+1)), matriz de temperaturas
    - title: string, titulo a ser colocado no grafico
    - path: string, caminho ate o local onde o arquivo sera salvo
    - filename: string, nome com o qual o arquivo sera salvo
    @output:
    - ax: axis (objeto de eixo do mpl)
    '''
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
    '''
    funcao para adicionar zeros nos extremos de uma array unidimensional, usada para computar a funcao f na eq 11
    @parameters:
    - array: array, uma array generica
    @output:
    - final_array: array, a array de entrada com 2 itens adicionados: inicial e final, ambos 0
    '''
    final_array = numpy.zeros(1)
    final_array = numpy.concatenate((final_array, array))
    final_array = numpy.concatenate((final_array, numpy.zeros(1)))
    return final_array

def get_D_matrix(N, lambda_val):
    '''
    funcao para obter a matriz tri-diagonal D a partir do valor de lambda
    @parameters:
    - N: inteiro, numero de divisoes na barra
    - lambda_val: float, constante do problema
    @output:
    - D_matrix: matrix ((M+1)x(N+1)), matriz tridiagonal
    '''
    a = numpy.diagflat(lambda_val * numpy.ones(N), 1)
    b = numpy.diagflat((1-2*lambda_val) * numpy.ones(N+1))
    c = numpy.diagflat(lambda_val * numpy.ones(N), -1)
    D_matrix = numpy.matrix(a+b+c)
    D_matrix[:,0] = numpy.zeros((N+1, 1))
    D_matrix[:,-1] = numpy.zeros((N+1, 1))
    return D_matrix

def e(space_array, k, T, M):
    '''
    funcao e associada a solucao exata
    @parameters:
    - space_array: array (1x(N+1)), contem todas as posicoes da barra
    - k: inteiro, indice da linha (ou seja, o instante de tempo) em que estamos calculado f
    - T: float, constante de tempo T
    - M: inteiro, numero de divisoes no tempo
    @output:
    - e: array (1x(N+1)), contem os valores de e calculados para um instante especifico
         t = (k*T/M) para todas as posicoes da barra
    '''
    # e = 10*(k*T/M)*(space_array**2)*(space_array - 1) # deprecated since EP1_v0.8
    e = (1+numpy.sin(10*(k*T/M)))*(space_array**2)*((1-space_array)**2)
    return e

def get_e_matrix(space_array, k, T, M):
    '''
    funcao que gera uma matriz de temperaturas a partir da equacao exata de difusao
    @parameters:
    - time_array: array (1x(M+1)), contem todos os instantes de tempo
    - space_array: array (1x(N+1)), contem todas as posicoes da barra
    @output:
    - e_array: array ((M+1)x(N+1)), contem os valores de temperatura calculados 
                em todos os instantes de tempo para todas as posicoes da barra
    '''
    e_array = numpy.apply_along_axis(e, 0, space_array, k, T, M)
    return e_array

def apply_exact_solution(T, lambda_val, exact_matrix, space_array):
    '''
    funcao para criar uma array com os valores da solucao exata
    @parameters:
    - T: float, constante de tempo T
    - lambda_val: float, constante do problema
    - exact_matrix: array ((M+1)x(N+1)), matriz das solucoes exatas (suposta nula neste momento)
    - space_array: array (1x(N+1)), contem todas as posicoes da barra
    @output:
    - exact_matrix: array ((M+1)x(N+1)), matriz das solucoes exatas preenchida segundo a solucao exata
    '''
    M = exact_matrix.shape[0] - 1 
    N = exact_matrix.shape[1] - 1
    for k in range(M+1):
        exact_matrix[k] = get_e_matrix(space_array, k, T, M)
    return exact_matrix
    
def f(space_array, k, T, M):
    '''
    funcao f associada a eq 11
    @parameters:
    - space_array: array (1x(N+1)), contem todas as posicoes da barra
    - k: inteiro, indice da linha (ou seja, o instante de tempo) em que estamos calculado f
    - T: float, constante de tempo T
    - M: inteiro, numero de divisoes no tempo
    @output:
    - f: array (1x(N+1)), contem os valores de f calculados para um instante especifico 
         t = (k*T/M) para todas as posicoes da barra
    '''
    # f = 10*(space_array**2)*(space_array - 1) - 60*space_array*((k*T)/M) + 20*((k*T)/M) # deprecated since EP1_v0.8
    f = (10*(numpy.cos(10*(k*T/M)))*(space_array**2)*((1-space_array)**2) -
         (1 + numpy.sin(10*(k*T/M)))*(12*(space_array**2) - 12*(space_array)) + 2)
    return f

def get_f_matrix(space_array, k, T, M):
    '''
    funcao que aplica o resultado da funcao anterior na space_array de fato
    @parameters:
    - space_array: array (1x(N+1)), contem todas as posicoes da barra
    - k: inteiro, indice da linha (ou seja, o instante de tempo) em que estamos calculado f
    - T: float, constante de tempo T
    - M: inteiro, numero de divisoes no tempo
    @output:
    - f_array: matrix ((M+1)x(N+1)), contem os valores de f calculados para um instante especifico k 
                para as posicoes da barra exceto as extremas, estas sao substituidas por zeros
    '''
    f_array = numpy.apply_along_axis(f, 0, space_array[1:-1], k, T, M)
    return numpy.matrix(add_initial_ending_zeros(f_array))

def get_u0_array(space_array):
    '''
    funcao que obtem a condicao de contorno u0, array das temperaturas da barra no tempo
    inicial
    @parameters:
    - space_array: array (1x(N+1)), contem todas as posicoes da barra
    @output:
    - u0: array (1x(N+1)), array que indica a temperatura da barra no instante t = 0
    '''
    bondary = lambda x: (x**2)*((1-x)**2)
    u0 = numpy.array([bondary(xi) for xi in space_array])
    return u0

def apply_equation_11(T, lambda_val, u, space_array):
    '''
    funcao que aplica a equacao 11 do enunciado do ep
    @parameters:
    - T: float, constante de tempo T
    - lambda_val: float, constante do problema
    - u: array ((M+1)x(N+1)), matriz de temperaturas (uma matriz quase nula com bordas ajustadas 
         pelas condicoes de contorno)
    - space_array: array (1x(N+1)), contem todas as posicoes da barra
    @output:
    - u: array ((M+1)x(N+1)), matriz de temperaturas com seus valores calculados segundo a equacao 11
    '''
    M = u.shape[0] - 1
    N = u.shape[1] - 1
    D = get_D_matrix(N, lambda_val)
    for k,_ in enumerate(u[1:], start = 1):
        u[k, 1:N] = numpy.asarray(u[k-1].dot(D) + (T/M)*(get_f_matrix(space_array, k, T, M)))[0,1:N].reshape(N-1,)
    return u

def get_error(u, e):
    '''
    funcao que executa a subtracao das arrays de ultimos valores das solucoes exata e aproximada,
    encontrando o erro agregado a aproximacao no fim das iteracoes
    @parameters:
    - u: array (1x(N+1)), array de temperaturas no instante final t = T
    - e: array (1x(N+1)), array de temperaturas calculada a partir da funcao exata no instante final t = T
    @output:
    - error_matrix: array (1x(N+1)), array de erro agregado ao instante t = T da aproximacao
    '''
    error_matrix = numpy.subtract(u, e)
    return error_matrix
def run_test_vectorized(T, lambda_val, N, exact = False):
    '''
    funcao que define a rotina de execucao do programa para determinados valores de T, lambda_val e N
    @parameters:
    - T: float, constante de tempo T
    - lambda_val: float, constante do problema
    - N: inteiro, numero de divisoes feitas na barra
    - exact: bool, indicador se calcularemos a equacao exata ou a aproximacao
    @output:
    -~ M: inteiro, numero de divisoes no tempo
    - delta_time: float, tempo total decorrido para a execucao do programa, em segundos
    -~ time_array: array (1x(M+1)), contem todos os instantes de tempo
    -~ space_array: array (1x(N+1)), contem todas as posicoes da barra
    -~ u: array ((M+1)x(N+1)), matriz de temperaturas
    -~ exact_matrix: array ((M+1)x(N+1)), matriz de temperaturas calculada a partir da funcao exata
    - last_line: array, (1x(N+1)), array da ultima linha de uma matriz (u ou exact_matrix)
    '''
    start_time = time.time()
    M = get_M_parameter(T, lambda_val, N)
    zeros = create_m_n_matrix(M,N)

    time_array = get_time_array(M, T)
    space_array = get_space_array(N)
    
    if exact:
        exact_matrix = numpy.copy(zeros)
        zeros = None
        exact_matrix = apply_exact_solution(T, lambda_val, exact_matrix, space_array)
        last_line = exact_matrix[-1]
        delta_time = round(time.time() - start_time, 3)
        return delta_time, exact_matrix, last_line
    else:
        u0 = get_u0_array(space_array)
        g1 = numpy.zeros((time_array.shape))
        g2 = numpy.zeros((time_array.shape))
        u = apply_boundary_conditions(zeros, u0, g1, g2)
        zeros = None
        u = apply_equation_11(T, lambda_val, u, space_array)
        last_line = u[-1]
        delta_time = round(time.time() - start_time, 3)
        return M, delta_time, time_array, space_array, u, last_line

def _1a(T = 1, lambda_list = [0.25, 0.5], N_list = [10, 20, 40, 80, 160, 320]):
    '''
    funcao que roda o script do item a) da primeira tarefa,
    rotina de execucao de todos os valores de T, lambda_val e N que precisam ser testados
    @parameters:
    - T: float, constante de tempo T
    - lambda_list: lista, contém os valores de lambda_val
    - N_list: lista, contém os valores de N
    @output:
    -
    '''
    main_dir = os.getcwd()
    create_folder(lambda_list, path = main_dir)
    for lambda_val in lambda_list:
        lambda_dir = os.path.join(main_dir, str(lambda_val))
        create_folder(N_list, path = lambda_dir)
        for N in N_list:
            n_dir = os.path.join(lambda_dir, str(N))
            print('Iniciando execucao -', 
                  colored('lambda_val = {}'.format(lambda_val), 'blue'), 
                  colored('N = {}'.format(N), 'red'),
                  colored('local_time = {}'.format(time.strftime('%H:%M:%S', time.localtime())), 'green')
                 )
            # Solucao aproximada
            print(' '*18 + ' - Solucao ', colored('aproximada', 'yellow'), colored('local_time = {}'.format(time.strftime('%H:%M:%S', time.localtime())), 'green'))
            M, delta_time_a, time_array, space_array, temperature_matrix, last_line_a = run_test_vectorized(T, lambda_val, N)
            plot_temperatures(T, lambda_val, N, delta_time_a, space_array, temperature_matrix, 'Temperatura', n_dir, 'time_series')
            plot_heatmap(T, lambda_val, N, delta_time_a, space_array, time_array, temperature_matrix, 'Temperatura', n_dir, 'heatmap')
            temperature_matrix = None

            # Solucao exata
            print(' '*18 + ' - Solucao ', colored('exata', 'magenta'), colored('local_time = {}'.format(time.strftime('%H:%M:%S', time.localtime())), 'green'))
            delta_time_e, exact_matrix, last_line_e = run_test_vectorized(T, lambda_val, N, exact = True)
            plot_temperatures(T, lambda_val, N, delta_time_e, space_array, exact_matrix, 'Solução exata', n_dir, 'exact_time_series')
            plot_heatmap(T, lambda_val, N, delta_time_e, space_array, time_array, exact_matrix, 'Solução exata', n_dir, 'exact_heatmap')
            exact_matrix = None

            # Erro associado
            print(' '*18 + ' -', colored('Erro associado', 'red'), colored('local_time = {}'.format(time.strftime('%H:%M:%S', time.localtime())), 'green'))
            error_array = get_error(last_line_a, last_line_e)
            delta_time_total = delta_time_a + delta_time_e
            plot_error_array(T, lambda_val, N, delta_time_total, space_array, error_array, n_dir, 'error_series')

# Teste para lambda_val = 0.51
def _1a_lambda(N_list):
    '''
    funcao que testa o comportamento da funcao para lambda_val = 0.51
    @parameters:
    - N_list: lista, contém os valores de N
    @output:
    -
    '''
    _1a(T = 1, lambda_list = [0.51], N_list = N_list)
