################################
######## NUMERICO - EP1 ########
################################
# Joao Rodrigo Windisch Olenscki
# NUSP 10773224
# Luca Rodrigues Miguel
# NUSP 10705655

# Bibliotecas
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import os
import sys
import time
import datetime
import math

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
            print("Folder", "{}".format(folder_name), "criado")
        except FileExistsError:
            print("Folder {} already exists in this directory".format(folder_name))
        except TypeError:
            print("TypeError, path = {}, folder_name = {}".format(path, folder_name))
            
def get_M_parameter(T, lambda_val, N, method):
    '''
    funcao para calcular o parametro M
    @parameters:
    - T: float, constante de tempo T
    - lambda_val: float, constante do problema
    - N: inteiro, numero de divisoes feitas na barra
    - method: string, metodo empregado
    -- example: 'euler', 'implicit_euler', 'crank_nicolson'
    @output:
    - M: inteiro, numero de divisoes no tempo
    '''
    if method == 'euler':
        M =T*(N**2)/lambda_val
    else:
        M = T*N
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
    u = np.zeros((M+1, N+1))
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
    time_array = np.linspace(0, T, num = M+1)
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
    space_array = np.linspace(0, 1, num = N+1)
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

def get_lambda_val(lambda_val, N, method):
    '''
    funcao para modularizar as funcoes de iteracao, uma vez que para os meto-
    dos implicitos o valor de lambda deixa de ser uma variavel do problema
    @parameters:
    - lambda_val: float, constante do problema
    - N: inteiro, numero de divisoes feitas na barra
    - method: string, metodo empregado
    @output:
    -~ lambda_val: float, constante do problema, para o metodo de euler
    -~ N: inteiro, numero de divisoes feitas na barra, para os metodos implicitos
    '''
    if method == 'euler':
        return lambda_val
    else:
        return N

def format_method(method):
    '''
    funcao simples que formata a string de metodo, deixando-a com primeiras
    letras maiusculas para cada palavra
    @parameters:
    - method: string, metodo empregado
    -- example: 'implicit_euler'
    @output:
    - method: string, nome do metodo modificado
    -- example: 'Implicit Euler'
    '''
    method = (' ').join(method.split('_')).title()
    return method
    
def plot_temperatures(T, lambda_val, N, delta_time, space_array, temperature_matrix, title, path, filename, method, test):
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
    - method: string, metodo empregado
    -- example: 'euler', 'implicit_euler', 'crank_nicolson'
    - test: string, teste a ser feito
    -- example: 'a', 'b', 'c'
    @output:
    - ax: axis (objeto de eixo do mpl)
    '''    
    L = temperature_matrix.shape[0]
    for time_step in range(11):
        temperature_array = temperature_matrix[(L//10)*time_step]
        plt.plot(space_array, temperature_array, label = r'${} s$'.format(time_step/10))
        
    ax = plt.gca()
    title_string = r'{} em função da posição para certas séries temporais pelo método de {}'.format(title, format_method(method))
    if method == 'euler':
        subtitle_string = r'$função = {}, \;T = {},\; \lambda = {},\; N = {},\;$Tempo de execução$:\; {}$ segundos'.format(test, T, lambda_val, N, delta_time)
    else:
        subtitle_string = r'$função = {}, \;T = {},\; N = {},\;$Tempo de execução$:\; {}$ segundos'.format(test, T, N, delta_time)
    plt.suptitle(title_string, y=1.0, fontsize = 18)
    ax.set_title(subtitle_string, fontsize = 14)
    ax.set_xlabel(r'Posição na barra ($x$)')
    ax.set_ylabel(r'{}'.format(title))
    ax.legend(loc='right', bbox_to_anchor=(1.25, 0.5))
    savedir = os.path.join(path, filename + '.png')
    plt.savefig(savedir, dpi = 300, bbox_inches="tight")
    plt.close()
    return ax
    
def plot_error_array(T, lambda_val, N, delta_time, space_array, error_array, path, filename, method, test):
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
    - max_error: float, valor absoluto do maior erro contido na array error_matrix
    '''
    plt.plot(space_array, error_array)
    
    ax = plt.gca()
    d_time = round(delta_time, 3)
    max_error = '{:.2e}'.format(np.absolute(error_array).max())
    title_string = r'Erro em função da posição para o instante $t = T$ no método de {}'.format(format_method(method))
    if method == 'euler':
        subtitle_string = r'$função = {}, \;T = {},\; \lambda = {},\; N = {},\;$Tempo de execução$:\; {}$ segundos, Erro máximo:$\; {}$'.format(test, T, lambda_val, N, d_time, max_error.replace('e-', '$e-$'))
    else:
        subtitle_string = r'$função = {}, \;T = {},\; N = {},\;$Tempo de execução$:\; {}$ segundos, Erro máximo:$\; {}$'.format(test, T, N, d_time, max_error.replace('e-', '$e-$'))
    plt.suptitle(title_string, y=1.0, fontsize = 18)
    ax.set_title(subtitle_string, fontsize = 14)
    ax.set_xlabel(r'Posição na barra ($x$)')
    ax.set_ylabel(r'Erro')
    savedir = os.path.join(path, filename + '.png')
    plt.savefig(savedir, dpi = 300, bbox_inches="tight")
    plt.close()
    return ax, max_error

def plot_convergence_order(method, test, error_list):
    '''
    funcao que executa o plot dos erros maximos para cada valor de N
    para a analise de convergencia de cada metodo
    @parameters:
    - method: string, metodo empregado
    -- example: 'euler', 'implicit_euler', 'crank_nicolson'
    - test: string, teste a ser feito
    -- example: 'a', 'b', 'c'
    - error_list: list, lista de inteiros (ou lista de listas de inteiros)
                  que contem os erros maximos para cada teste
    '''
    max_val = 0
    is_list_list = False
    
    _dir = os.path.join(os.getcwd(), 'figures')
    create_folder(['error_analysis'], path = _dir)
    path = os.path.join(_dir, 'error_analysis')
    if any(isinstance(el, list) for el in error_list): #caso a, em que temos 2 lambdas
        is_list_list = True
        N_list = [10*(2**i) for i in range(len(error_list[0]))]
        for i in range(len(error_list)):
            plt_list = [float(el) for el in error_list[i]]
            max_val = max(max_val, max(plt_list))
            plt.scatter(N_list, plt_list, label = r'${} $'.format(0.25*(i+1)))
    else:
        N_list = [10*(2**i) for i in range(len(error_list))]
        plt_list = [float(el) for el in error_list]
        max_val = max(plt_list)
        plt.scatter(N_list, plt_list)
    ax = plt.gca()
    title_string = r'Ordem de convergência do método {} para a função ${}$'.format(format_method(method), test)
    plt.suptitle(title_string, y=1.0, fontsize = 18)
    ax.set_xlabel(r'$N$')
    ax.set_ylabel(r'Módulo do máximo erro')
    
    if is_list_list:
        leg = ax.legend(loc='right', bbox_to_anchor=(1.25, 0.5))
        leg.set_title(r'Valor de $\lambda$', prop = {'size': 14})
    ax.set_ylim([0,1.05*max_val])
    ax.set_xscale('log', basex = 2)
    
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    
    ax.xaxis.set_major_formatter(formatter)
    plt.xticks(N_list)
    filename = method+ '_' + test
    savedir = os.path.join(path, filename + '.png')
    plt.savefig(savedir, dpi = 300, bbox_inches="tight")
    plt.close()
    return ax
    
def plot_heatmap(T, lambda_val, N, delta_time, space_array, time_array, temperature_matrix, title, path, filename, method, test):
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
    - method: string, metodo empregado
    -- example: 'euler', 'implicit_euler', 'crank_nicolson'
    - test: string, teste a ser feito
    -- example: 'a', 'b', 'c'
    @output:
    - ax: axis (objeto de eixo do mpl)
    '''
    title_string = r'Mapa de {}'.format(title) + r' para a barra inteira em todos os $\mathit{ticks}$ de tempo ' + 'no método de {}'.format(format_method(method))
    if method == 'euler':
        subtitle_string = r'$função = {}, \;T = {},\; \lambda = {},\; N = {},\;$Tempo de execução$:\; {}$ segundos'.format(test, T, lambda_val, N, delta_time)
    else:
        subtitle_string = r'$função = {}, \;T = {},\; N = {},\;$Tempo de execução$:\; {}$ segundos'.format(test, T, N, delta_time)
        
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
    final_array = np.zeros(1)
    final_array = np.concatenate((final_array, array))
    final_array = np.concatenate((final_array, np.zeros(1)))
    return final_array

def get_A_matrix(N, lambda_val, method):
    '''
    funcao para obter a matriz tri-diagonal A a partir do valor de lambda
    @parameters:
    - N: inteiro, numero de divisoes na barra
    - lambda_val: float, constante do problema
    - method: string, metodo empregado
    @output:
    -~ A_matrix: matrix ((M+1)x(N+1)), matriz tridiagonal com bordas nulas, para o metodo
                 de euler
    -~ A_matrix: matrix ((N-1)x(N-1)), matriz tridiagonal para o metodo de euler implicito
                 e para o metodo de crank-nicolson
    '''
    if method == 'euler':
        a = np.diagflat(lambda_val * np.ones(N), 1)
        b = np.diagflat((1-2*lambda_val) * np.ones(N+1))
        c = np.diagflat(lambda_val * np.ones(N), -1)
        A_matrix = np.matrix(a+b+c)
        A_matrix[:,0] = np.zeros((N+1, 1))
        A_matrix[:,-1] = np.zeros((N+1, 1))
    elif method == 'implicit_euler':
        a = np.diagflat(- lambda_val * np.ones(N-2), 1)
        b = np.diagflat((1+2*lambda_val) * np.ones(N-1))
        c = np.diagflat(- lambda_val * np.ones(N-2), -1)
        A_matrix = np.matrix(a+b+c)
    elif method == 'crank_nicolson':
        a = np.diagflat(- lambda_val/2 * np.ones(N-2), 1)
        b = np.diagflat((1+lambda_val) * np.ones(N-1))
        c = np.diagflat(- lambda_val/2 * np.ones(N-2), -1)
        A_matrix = np.matrix(a+b+c)
    return A_matrix

def _1a_e(space_array, k, T, M):
    '''
    funcao e associada a solucao exata para a f(x,t) do item a)
    @parameters:
    - space_array: array (1x(N+1)), contem todas as posicoes da barra
    - k: inteiro, indice da linha (ou seja, o instante de tempo) em que estamos calculado f
    - T: float, constante de tempo T
    - M: inteiro, numero de divisoes no tempo
    @output:
    - e_array: array (1x(N+1)), contem os valores de temperatura calculados 
                em todos os instantes de tempo para todas as posicoes da barra
    '''
    # e = 10*(k*T/M)*(space_array**2)*(space_array - 1) # deprecated since EP1_v0.8
    def e(space_array, k, T, M):
        t = k*T/M
        x = space_array
        return (1+np.sin(10*t))*(x**2)*((1-x)**2)
    
    e_array = np.apply_along_axis(e, 0, space_array, k, T, M)
    return e_array

def _1b_e(space_array, k, T, M):
    '''
    funcao e associada a solucao exata para a f(x,t) do item b)
    @parameters:
    - space_array: array (1x(N+1)), contem todas as posicoes da barra
    - k: inteiro, indice da linha (ou seja, o instante de tempo) em que estamos calculado f
    - T: float, constante de tempo T
    - M: inteiro, numero de divisoes no tempo
    @output:
    - e_array: array (1x(N+1)), contem os valores de temperatura calculados 
                em todos os instantes de tempo para todas as posicoes da barra
    '''
    def e(space_array, k, T, M):
        t = k*T/M
        x = space_array
        return np.exp(t-x)*np.cos(5*t*x)
    
    e_array = np.apply_along_axis(e, 0, space_array, k, T, M)
    return e_array    
    
def apply_exact_solution(T, lambda_val, exact_matrix, space_array, f_function):
    '''
    funcao para criar uma array com os valores da solucao exata
    @parameters:
    - T: float, constante de tempo T
    - lambda_val: float, constante do problema
    - exact_matrix: array ((M+1)x(N+1)), matriz das solucoes exatas (suposta nula neste momento)
    - space_array: array (1x(N+1)), contem todas as posicoes da barra
    - f_function: funcao, funcao do python passada como argumento para identificar o teste
    @output:
    - exact_matrix: array ((M+1)x(N+1)), matriz das solucoes exatas preenchida segundo a solucao exata
    '''
    M = exact_matrix.shape[0] - 1 
    N = exact_matrix.shape[1] - 1
    for k in range(M+1):
        if f_function.__name__ == '_1a_f':
            exact_matrix[k] = _1a_e(space_array, k, T, M)
        elif f_function.__name__ == '_1b_f':
            exact_matrix[k] = _1b_e(space_array, k, T, M)
            
    return exact_matrix
    
def _1a_f(space_array, k, T, M):
    '''
    funcao f associada a eq 11 e ao item a)
    @parameters:
    - space_array: array (1x(N+1)), contem todas as posicoes da barra
    - k: inteiro, indice da linha (ou seja, o instante de tempo) em que estamos calculado f
    - T: float, constante de tempo T
    - M: inteiro, numero de divisoes no tempo
    @output:
    - f_matrix: matrix (1x(N+1)), contem os valores de f calculados para um instante especifico k 
                para as posicoes da barra exceto as extremas, estas sao substituidas por zeros
    '''
    # f = 10*(space_array**2)*(space_array - 1) - 60*space_array*((k*T)/M) + 20*((k*T)/M) # deprecated since EP1_v0.8
    def f(space_array, k, T, M):
        t = k*T/M
        x = space_array
        return (10*(np.cos(10*t))*(x**2)*((1-x)**2) -
                 (1 + np.sin(10*t))*(12*(x**2) - 12*x + 2))
    
    f_array = np.apply_along_axis(f, 0, space_array[1:-1], k, T, M)
    f_matrix = np.matrix(add_initial_ending_zeros(f_array))
    return f_matrix

def _1b_f(space_array, k, T, M):
    '''
    funcao f associada a eq 11 e ao item b)
    @parameters:
    - space_array: array (1x(N+1)), contem todas as posicoes da barra
    - k: inteiro, indice da linha (ou seja, o instante de tempo) em que estamos calculado f
    - T: float, constante de tempo T
    - M: inteiro, numero de divisoes no tempo
    @output:
    - f_matrix: matrix (1x(N+1)), contem os valores de f calculados para um instante especifico k 
                para as posicoes da barra exceto as extremas, estas sao substituidas por zeros
    '''
    def f(space_array, k, T, M):
        t = k*T/M
        x = space_array
        return ((25*(t**2)*np.cos(5*t*x)) - 5*(x + 2*t)*np.sin(5*t*x))*np.exp(t - x)
    f_array = np.apply_along_axis(f, 0, space_array[1:-1], k, T, M)
    f_matrix = np.matrix(add_initial_ending_zeros(f_array))
    return f_matrix
    
def _1c_f(space_array, k, T, M, p = 0.25):
    '''
    funcao que define a f(x,t) para o item c) da primeira tarefa
    @parameters:
    - space_array: array (1x(N+1)), contem todas as posicoes da barra
    - k: inteiro, indice da linha (ou seja, o instante de tempo) em que estamos calculado f
    - T: float, constante de tempo T
    - M: inteiro, numero de divisoes no tempo
    - p: float, 0 < p < 1 posicao da fonte de calor
    -- default_value: 0.25
    @output:
    - f_matrix: matrix (1x(N+1)), contem os valores de f calculados para um instante especifico k 
                para as posicoes da barra exceto as extremas, estas sao substituidas por zeros
    '''
    N = space_array.shape[0] - 1
    h = 1/N
    t_delta = 1e-10 # Pequeno delta a ser adicionado aos intervalos de forma a
                    # alarga-los e permitir que os erros de truncamento do python
                    # e do np sejam equiparados.
    t = k*T/M
    x = space_array
    inf = p - h/2 - t_delta
    sup = p + h/2 + t_delta
    g_array = np.piecewise(x,
                              [x < inf, (x >= p - h/2) & (x <= p + h/2), x > sup],
                              [   0   ,               1/h              ,   0    ]
                             )
    tal = 10000*(1-2*(t**2))
    f_array = g_array*tal
    f_matrix = np.matrix(add_initial_ending_zeros(f_array[1:-1]))
    return f_matrix

def get_u0_array(space_array, f_function):
    '''
    funcao que obtem a condicao de contorno u0, array das temperaturas da barra no tempo
    inicial
    @parameters:
    - space_array: array (1x(N+1)), contem todas as posicoes da barra
    - f_function: funcao, funcao do python passada como argumento para identificar o teste
    @output:
    - u0: array (1x(N+1)), array que indica a temperatura da barra no instante t = 0
    '''
    if f_function.__name__ == '_1a_f':
        bondary = lambda x: (x**2)*((1-x)**2)
    elif f_function.__name__ == '_1b_f':
        bondary = lambda x: np.exp(-x)
    else:
        bondary = lambda x: 0
    
    u0 = np.array([bondary(xi) for xi in space_array])
    return u0

def get_g1_array(time_array, f_function):
    '''
    funcao para obter a condicao de contorno g1, array das temperaturas da barra na
    posicao x = 0 para todos os instantes do tempo
    @parameters:
    - time_array: array (1x(M+1)), contem todos os instantes de tempo
    - f_function: funcao, funcao do python passada como argumento para identificar o teste
    @output:
    - g1: array (1x(N+1)), array que indica a temperatura da barra na posicao x = 0
          para todos os instantes de tempo
    '''
    if f_function.__name__ == '_1a_f':
        bondary = lambda t: 0
    elif f_function.__name__ == '_1b_f':
        bondary = lambda t: np.exp(t)
    else:
        bondary = lambda x: 0
        
    g1 = np.array([bondary(ti) for ti in time_array]) 
    return g1

def get_g2_array(time_array, f_function):
    '''
    funcao para obter a condicao de contorno g2, array das temperaturas da barra na
    posicao x = 1 para todos os instantes do tempo
    @parameters:
    - time_array: array (1x(M+1)), contem todos os instantes de tempo
    - f_function: funcao, funcao do python passada como argumento para identificar o teste
    @output:
    - g2: array (1x(N+1)), array que indica a temperatura da barra na posicao x = 0
          para todos os instantes de tempo
    '''
    if f_function.__name__ == '_1a_f':
        bondary = lambda t: 0
    elif f_function.__name__ == '_1b_f':
        bondary = lambda t: np.exp(t-1)*np.cos(5*t)
    else:
        bondary = lambda x: 0
    
    g2 = np.array([bondary(ti) for ti in time_array]) 
    return g2

def apply_estimated_solution(T, lambda_val, u, space_array, f_function, method):
    '''
    funcao iterativa que realiza as integracoes numericas
    @parameters:
    - T: float, constante de tempo T
    - lambda_val: float, constante do problema
    - u: array ((M+1)x(N+1)), matriz de temperaturas (uma matriz quase nula com bordas ajustadas 
         pelas condicoes de contorno)
    - space_array: array (1x(N+1)), contem todas as posicoes da barra
    - f_function: funcao, funcao do python
    - method: string, metodo empregado
    @output:
    - u: array ((M+1)x(N+1)), matriz de temperaturas com seus valores calculados segundo a equacao 11
    '''
    M = u.shape[0] - 1
    N = u.shape[1] - 1
    A = get_A_matrix(N, lambda_val, method)
    delta_t = T/M
    if method == 'euler':
        for k,_ in enumerate(u[1:], start = 1):
            f_array =f_function(space_array, k, T, M)
            u[k, 1:N] = np.asarray(u[k-1].dot(A) + delta_t*(f_array))[0,1:N].reshape(N-1,)
            
    elif method == 'implicit_euler':
        for k,_ in enumerate(u[1:], start = 1):
            f_array = np.asarray(f_function(space_array, k, T, M)).ravel()
            upper_element = np.array([
                                        u[k-1, 1] + 
                                        delta_t*f_array[1] + 
                                        lambda_val*u[k, 0]
                                        ])
            mid_elements = np.asarray(
                                        u[k-1, 2:N-1] + 
                                        delta_t*f_array[2:N-1]
                                        ).ravel()
            lower_element = np.array([
                                        u[k-1, N-1] + 
                                        delta_t*f_array[N-1] + 
                                        lambda_val*u[k, -1]
                                        ])
            linsys_asw = np.concatenate((upper_element, mid_elements, lower_element))
            u[k, 1:N] = solve_linear_system(A, linsys_asw)
            
    elif method == 'crank_nicolson':
        f_array_anterior = np.asarray(f_function(space_array, 0, T, M)).ravel()
        for k,_ in enumerate(u[1:], start = 1):
            f_array_atual = np.asarray(f_function(space_array, k, T, M)).ravel()
            f_array_mean = (delta_t/2)*(f_array_anterior + f_array_atual)
            f_array_anterior = f_array_atual
            upper_element = np.array([
                                        (1 - lambda_val)*u[k-1, 1] + 
                                        (lambda_val/2)*(u[k-1, 2] + u[k-1, 0] + u[k, 0]) + 
                                        f_array_mean[1]
                                        ])
            mid_elements = np.asarray(
                                        (1 - lambda_val) *u[k-1, 2:N-1] +
                                        (lambda_val/2)*(u[k-1, 1:N-2] + u[k-1, 3:N]) +
                                        f_array_mean[2:N-1]
                                        ).ravel()
            lower_element = np.array([
                                        (1 - lambda_val)*u[k-1, N-1] +
                                        (lambda_val/2)*(u[k-1, N-2] + u[k-1, -1] + u[k, -1]) +
                                        f_array_mean[N-1]
                                        ])
            linsys_asw = np.concatenate((upper_element, mid_elements, lower_element))
            u[k, 1:N] = solve_linear_system(A, linsys_asw)
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
    error_matrix = np.subtract(u, e)
    return error_matrix

def run_vectorized(T, lambda_val, N, f_function, exact = False, method = 'euler'):
    '''
    funcao que define a rotina de execucao do programa para determinados valores de T, lambda_val e N
    para o metodo de euler
    @parameters:
    - T: float, constante de tempo T
    - lambda_val: float, constante do problema
    - N: inteiro, numero de divisoes feitas na barra
    - f_function, funcao f(x,t) para o teste
    - exact: bool, indicador se calcularemos a equacao exata ou a aproximacao
    -- default_value: False
    - method: string, representa qual o metodo empregado na resolucao da integracao numerica
    -- default_value: 'euler'
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
    M = get_M_parameter(T, lambda_val, N, method)
    lambda_val = get_lambda_val(lambda_val, N, method)
    zeros = create_m_n_matrix(M,N)

    time_array = get_time_array(M, T)
    space_array = get_space_array(N)
    
    if exact:
        exact_matrix = np.copy(zeros)
        zeros = None
        exact_matrix = apply_exact_solution(T, lambda_val, exact_matrix, space_array, f_function)
        last_line = exact_matrix[-1]
        delta_time = round(time.time() - start_time, 3)
        return delta_time, exact_matrix, last_line
    else:
        u0 = get_u0_array(space_array, f_function)
        g1 = get_g1_array(time_array, f_function)
        g2 = get_g2_array(time_array, f_function)
        u = apply_boundary_conditions(zeros, u0, g1, g2)
        zeros = None
        u = apply_estimated_solution(T, lambda_val, u, space_array, f_function, method)
        last_line = u[-1]
        delta_time = round(time.time() - start_time, 3)
        return M, delta_time, time_array, space_array, u, last_line

def perform_ldlt_transformation(a, b):
    '''
    funcao que aplica a decomposicao de uma matriz tridiagonal simetrica
    na multiplicacao de 3 outras, de forma que A = L.D.L'
    @parameters:
    - a: array, array unidimensional generica que representa os valores da 
         diagonal principal
    - b: array, array unidimensional generica que representa os valores da 
         diagonal secundaria, seu primeiro valor e nulo (b[0] = 0)
    -> a e b possuem a mesma dimensao
    @output:
    - d: array, array unidimensional de mesma dimensao que a e b e que repre-
         senta a diagonal principal da matriz D
    - l: array, array unidimensional de mesma dimensao que a e b e que repre-
         senta a diagonal secundaria inferior da matriz L e também a diagonal 
         secundaria superior de L', seu primeiro valor e nulo (l[0] = 0)
    '''
    N = a.shape[0] + 1
    d = np.ones(N-1)
    l = np.ones(N-1)
    d[0] = a[0]
    for i in range(N-2):
        l[i+1] = b[i+1]/d[i]
        d[i+1] = a[i+1] - b[i+1]*l[i+1]
    return d, l

def solve_linear_system(A, u):
    '''
    funcao que resolve um sistema linear da forma Ax = u,
    retornando o vetor x
    @parameters:
    - A: array ((N-1)x(N-1)), matriz quadrada de entrada do sistema
    - u: array ((N-1)x1), vetor de saida da equacao
    @output:
    - x: array ((N-1)x1), vetor de incognitas
    '''
    # A.x = L.D.Lt.x
    a = np.asarray(A.diagonal(0)).ravel()
    N = a.shape[0] + 1
    b = np.concatenate((np.array([0]), np.asarray(A.diagonal(1)).ravel()))
    d, l = perform_ldlt_transformation(a, b)
    
    # Como explicado no relatorio, resolvemos o problema atraves da reso-
    # lucao de 3 loops: primeiro resolvemos em z, depois em y e ai sim em
    # x
    #Loop em z:
    z = np.zeros(N-1)
    z[0] =  u[0]
    for i in range(1, N-1):
        z[i] = u[i] - z[i-1]*l[i]
    #Loop em y:
    y = np.zeros(N-1)
    for i in range(N-1):
        y[i] = z[i]/d[i]
    #Loop em x:
    x = np.zeros(N-1)
    x[-1] = y[-1]
    for i in range(N-3, -1, -1):
        x[i] = y[i] - x[i+1]*l[i+1]
    return x

def generate_plots(T, lambda_val, N, _dir, method, test):
    '''
    funcao que gera os graficos de um teste
    @parameters:
    - T: float, constante de tempo T
    - lambda_val: float, constante do problema
    - N: inteiro, numero de divisoes feitas na barra
    - _dir: string, caminho no computador onde os graficos serao salvos
    -- example: 'C:/users/rodri/POLI/NUMERICO/EP1'
    - method: string, representa qual o metodo empregado na resolucao da inte-
              gracao numerica
    - test: string, representa qual a funcao empregada bem como as condicoes 
            de contorno
    @output:
    -~ max_error: float, valor absoluto do maior erro contido na array error_matrix
    '''
    f_function = tests_dic[test]['f_function']
    max_error = 0 # definindo valor padrao de max_error para o caso (c) em que ele nao e aplicavel
    
    exact = False
    print(' '*18 + ' - Solucao', 'aproximada', 'local_time = {}'.format(time.strftime('%H:%M:%S', time.localtime())))
    M, delta_time_a, time_array, space_array, temperature_matrix, last_line_a = run_vectorized(T, lambda_val, N, f_function, exact = exact, method = method)
    plot_temperatures(T, lambda_val, N, delta_time_a, space_array, temperature_matrix, 'Temperatura', _dir, 'time_series', method = method, test = test)
    plot_heatmap(T, lambda_val, N, delta_time_a, space_array, time_array, temperature_matrix, 'Temperatura', _dir, 'heatmap', method = method, test = test)
    temperature_matrix = None
    
    if f_function.__name__ != '_1c_f':
        exact = True
        print(' '*18 + ' - Solucao ', 'exata', 'local_time = {}'.format(time.strftime('%H:%M:%S', time.localtime())))
        delta_time_e, exact_matrix, last_line_e = run_vectorized(T, lambda_val, N, f_function, exact = exact, method = method)
        plot_temperatures(T, lambda_val, N, delta_time_e, space_array, exact_matrix, 'Solução exata', _dir, 'exact_time_series', method = method, test = test)
        plot_heatmap(T, lambda_val, N, delta_time_e, space_array, time_array, exact_matrix, 'Solução exata', _dir, 'exact_heatmap', method = method, test = test)
        exact_matrix = None

        print(' '*18 + ' -', 'Erro associado', 'local_time = {}'.format(time.strftime('%H:%M:%S', time.localtime())))
        error_array = get_error(last_line_a, last_line_e)
        delta_time_total = delta_time_a + delta_time_e
        ax, max_error = plot_error_array(T, lambda_val, N, delta_time_total, space_array, error_array, _dir, 'error_series', method = method, test = test)
        print(' '*18 + ' -','Finalizado', 'local_time = {}'.format(time.strftime('%H:%M:%S', time.localtime())))
    
    return max_error
        
        
def run_set_of_tests(T = 1, lambda_list = [0.25, 0.5], N_list = [10, 20, 40, 80, 160, 320], method = 'euler', test = 'a'):
    '''
    funcao que organiza os testes, rodando em sequencia varios deles para
    certo metodo e teste
    @parameters:
    - T: float, constante de tempo T
    -- default_value: 1
    - lambda_list: list, lista dos valores de lambda a serem testados
    -- default_value: [0.25, 0.5]
    - N_list: list, lista dos valores de N a serem testados
    -- default_value: [10, 20, 40, 80, 160, 320]
    - method: string, representa qual o metodo empregado na resolucao da inte-
              gracao numerica
    -- default_value: 'euler'
    - test: string, representa qual a funcao empregada bem como as condicoes 
            de contorno
    -- default_value: 'a'
    @output:
    - error_list: list, lista que guarda o maior valor de erro para
                 cada valor de N (e lambda_val) para a analise da ordem de 
                 convergencia
    '''
    error_list = []
    _dir = os.getcwd()
    create_folder(['figures'], path = _dir)
    _dir = os.path.join(_dir, 'figures')
    create_folder([method], path = _dir)
    _dir = os.path.join(_dir, method)
    create_folder([test], path = _dir)
    _dir = os.path.join(_dir, test)
    if method != 'euler':
        create_folder(N_list, path = _dir)
        for N in N_list:
            lambda_val = N
            N_dir = os.path.join(_dir, str(N))
            print('Iniciando execucao -',
                  'método = {}'.format(method), 
                  'funcao = {}'.format(test), 
                  'N = {}'.format(N),
                  'local_time = {}'.format(time.strftime('%H:%M:%S', time.localtime()))
                 )
            max_error = generate_plots(T, lambda_val, N, N_dir, method, test)
            error_list.append(max_error)
            
    else:
        create_folder(lambda_list, path = _dir)
        for lambda_val in lambda_list:
            lambda_dir = os.path.join(_dir, str(lambda_val))
            create_folder(N_list, path = lambda_dir)
            _list = []
            for N in N_list:
                N_dir = os.path.join(lambda_dir, str(N))
                print('Iniciando execucao -',
                  'método = {}'.format(method), 
                  'funcao = {}'.format(test), 
                  'N = {}'.format(N),
                  'lambda_val = {}'.format(lambda_val),
                  'local_time = {}'.format(time.strftime('%H:%M:%S', time.localtime()))
                 )
                max_error = generate_plots(T, lambda_val, N, N_dir, method, test)
                _list.append(max_error)
            error_list.append(_list)
            
    return error_list
            
# Lista com os nomes dos metodos
methods_list = ['euler', 'crank_nicolson', 'implicit_euler']
# Dicionario com as funcoes f de cada teste
tests_dic = {'a' : {'f_function': _1a_f}, 'b' : {'f_function': _1b_f}, 'c' : {'f_function': _1c_f}}
# Lista com as keys do dicionario acima
tests_list = list(tests_dic.keys())

                
def main():
    '''
    funcao main() do ep, roda todos os testes requisitados
    @parameters:
    - None
    @output:
    - None
    '''
    error_dic = {}
    for method in methods_list:
        error_dic[method] = {}
        for test in tests_list:
            error_list = run_set_of_tests(T = 1, lambda_list = [0.25, 0.5], N_list = [10, 20, 40, 80, 160, 320], method = method, test = test)
            error_dic[method][test] = error_list
            if test != 'c':
                plot_convergence_order(method, test, error_list)

    run_set_of_tests(T = 1, lambda_list = [0.51], N_list = [10, 20, 40], method = 'euler', test = 'a')