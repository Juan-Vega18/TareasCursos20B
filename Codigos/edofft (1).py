import numpy as np
from scipy.fft import fft
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Definimos las constantes
m = 1.0  # masa
m_1 = 1.0 #masa 1
m_2 = 1.0 #masa_2

g = 9.81  # gravedad
k = 1.0  # constante del resorte
a = 1.0 #longitud natural del resorte

b = 1.0  # longitud del péndulo 1
c = 1.0 # longitud del pendulo 2

pequeño1 = 0.3
pequeño2 = 0.2
#condicones iniciales

angulo1 = np.pi /4
angulo2 = np.pi /5

# Definimos la ecuación diferencial
def sistema_ecuaciones(t, y):
    theta1, dtheta1_dt, theta2, dtheta2_dt = y
    
    # Ecuación para el primer péndulo
    
    d2tetha1_dt2 = ((-1)*m*g*b*(np.sin(theta1))-(0.5)*k*(-2*a*b*(np.cos(theta1))-2*(b**2)*(np.sin(theta2-theta1))-a*((2*(b**2) + 2*a*b*(np.sin(theta2)-np.sin(theta1)) - 2*(b**2)*(np.cos(theta2-theta1))+(a**2))**(-0.5))*(-2*a*b*np.cos(theta1)- 2*(b**2)*np.sin(theta2-theta1))))/(m*b*b)
    d2tetha2_dt2 = ((-1)*m*g*b*(np.sin(theta2))-(0.5)*k*(2*a*b*(np.cos(theta2))+2*(b**2)*(np.sin(theta2-theta1))-a*((2*(b**2) + 2*a*b*(np.sin(theta2)-np.sin(theta1)) - 2*(b**2)*(np.cos(theta2-theta1))+(a**2))**(-0.5))*(2*a*b*np.cos(theta2)+ 2*(b**2)*np.sin(theta2-theta1))))/(m*b*b)

    return [dtheta1_dt, d2tetha1_dt2, dtheta2_dt, d2tetha2_dt2]


def sistema_ecuaciones_1(t, y): #Para longitudes de varillas y masas distintas
    theta1_1, dtheta1_1_dt, theta2_1, dtheta2_1_dt = y
    
    
    d2tetha1_1_dt2 = ((-1)*m_1*g*b*(np.sin(theta1_1))-(0.5)*k*(-2*b*a*(np.cos(theta1_1))-2*(b)*c*(np.sin(theta2_1-theta1_1))-a*(((a**2)+ (c**2) + (b**2) + 2*a*((c*np.sin(theta2_1))-((b)*np.sin(theta1_1))) - 2*b*c*(np.cos(theta2_1-theta1_1)))**(-0.5))*(-2*a*b*np.cos(theta1_1)- 2*b*c*np.sin(theta2_1-theta1_1))))/(m_1*b*b)
    d2tetha2_1_dt2 = ((-1)*m_1*g*b*(np.sin(theta2_1))-(0.5)*k*(2*b*a*(np.cos(theta2_1))+2*(b)*c*(np.sin(theta2_1-theta1_1))-a*(((a**2)+ (c**2) + (b**2) + 2*a*((c*np.sin(theta2_1))-((b)*np.sin(theta1_1))) - 2*b*c*(np.cos(theta2_1-theta1_1)))**(-0.5))*(2*a*c*np.cos(theta2_1)+ 2*b*c*np.sin(theta2_1-theta1_1))))/(m_2*c*c)

    return [dtheta1_1_dt, d2tetha1_1_dt2, dtheta2_1_dt, d2tetha2_1_dt2]

def sistema_ecuaciones_2(t, y): #Para cuando el resorte está colocado en la mitad del las barillas
    theta1_2, dtheta1_2_dt, theta2_2, dtheta2_2_dt = y
    
    
    d2tetha1_2_dt2 = ((-1)*m*g*b*(np.sin(theta1_2))-(0.5)*k*(-a*b*(np.cos(theta1_2))-(b**2)*(0.5)*(np.sin(theta2_2-theta1_2))-a*(((0.5)*(b**2) + a*b*(np.sin(theta2_2)-np.sin(theta1_2)) - (0.5)*(b**2)*(np.cos(theta2_2-theta1_2))+(a**2))**(-0.5))*(-a*b*np.cos(theta1_2)- (0.5)*(b**2)*np.sin(theta2_2-theta1_2))))/(m*b*b)
    d2tetha2_2_dt2 = ((-1)*m*g*b*(np.sin(theta2_2))-(0.5)*k*(a*b*(np.cos(theta2_2))+(b**2)*(0.5)*(np.sin(theta2_2-theta1_2))-a*(((0.5)*(b**2) + a*b*(np.sin(theta2_2)-np.sin(theta1_2)) - (0.5)*(b**2)*(np.cos(theta2_2-theta1_2))+(a**2))**(-0.5))*(a*b*np.cos(theta2_2)+ (0.5)*(b**2)*np.sin(theta2_2-theta1_2))))/(m*b*b)

    return [dtheta1_2_dt, d2tetha1_2_dt2, dtheta2_2_dt, d2tetha2_2_dt2]

def sistema_ecuaciones_3(t, y): #Para angulos pequeños

    theta1_3, dtheta1_3_dt, theta2_3, dtheta2_3_dt = y
    
    # Ecuación para el primer péndulo
    
    d2tetha1_3_dt2 = ((-1)*m*g*b*((theta1_3))-(0.5)*k*(-2*a*b*(1)-2*(b**2)*((theta2_3-theta1_3))-a*((2*(b**2) + 2*a*b*((theta2_3)-(theta1_3)) - 2*(b**2)*(1)+(a**2))**(-0.5))*(-2*a*b*1- 2*(b**2)*(theta2_3-theta1_3))))/(m*b*b)
    d2tetha2_3_dt2 = ((-1)*m*g*b*((theta2_3))-(0.5)*k*(2*a*b*(1)+2*(b**2)*((theta2_3-theta1_3))-a*((2*(b**2) + 2*a*b*((theta2_3)-(theta1_3)) - 2*(b**2)*(1)+(a**2))**(-0.5))*(2*a*b*1+ 2*(b**2)*(theta2_3-theta1_3))))/(m*b*b)

    return [dtheta1_3_dt, d2tetha1_3_dt2, dtheta2_3_dt, d2tetha2_3_dt2]
condiciones_iniciales = [angulo1, 0, angulo2, 0]

# Intervalo de tiempo para la simulación
t_span = (0, 2500)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

#Se resuelven los sistemas de ecuaciones diferenciales
solucion = solve_ivp(sistema_ecuaciones, t_span, condiciones_iniciales, t_eval=t_eval)
solucion1 = solve_ivp(sistema_ecuaciones_1, t_span, condiciones_iniciales, t_eval=t_eval)
solucion2 = solve_ivp(sistema_ecuaciones_2, t_span, condiciones_iniciales, t_eval=t_eval)
solucion3 = solve_ivp(sistema_ecuaciones_3, t_span, condiciones_iniciales, t_eval=t_eval)


#solución con condiciones ligeramente modificadas.
condiciones_iniciales_modificadas = [angulo1 + 0.0001, 0, angulo2 + 0.0001, 0]
solucion_condiciones_modificadas = solve_ivp(sistema_ecuaciones, t_span, condiciones_iniciales_modificadas, t_eval=t_eval)

# Se obtienen los thetas para el primer SED
theta1 = solucion.y[0]
theta2 = solucion.y[2]
# Se obtienen los thetas para el segundo SED
theta1_1 = solucion1.y[0]
theta2_1 = solucion1.y[2]
# Se obtienen los thetas para el tercer SED
theta1_2 = solucion2.y[0]
theta2_2 = solucion2.y[2]
# Se obtienen los thetas para el tercer SED
theta1_3 = solucion3.y[0]
theta2_3 = solucion3.y[2]

theta1_condicones_modificadas = solucion_condiciones_modificadas.y[0]
theta2_condicones_modificadas = solucion_condiciones_modificadas.y[2]

#resta de soluciones condicones iniciales menos condiciones ligeramente modificadas

resta1 = theta1_condicones_modificadas - theta1
resta2 = theta1_condicones_modificadas - theta2

# Aplicar la FFT para analizar las frecuencias de theta1 y theta2
fft_theta1 = fft(theta1)
fft_theta2 = fft(theta2)
frecuencias = np.fft.fftfreq(len(t_eval), d=t_eval[1] - t_eval[0])

# Aplicar la FFT para analizar las frecuencias de theta1_1 y theta2_1
fft_theta1_1 = fft(theta1_1)
fft_theta2_1 = fft(theta2_1)
frecuencias = np.fft.fftfreq(len(t_eval), d=t_eval[1] - t_eval[0])

# Aplicar la FFT para analizar las frecuencias de theta1_2 y theta2_2
fft_theta1_2 = fft(theta1_2)
fft_theta2_2 = fft(theta2_2)
frecuencias = np.fft.fftfreq(len(t_eval), d=t_eval[1] - t_eval[0])
# Aplicar la FFT para analizar las frecuencias de theta1_3 y theta2_3
fft_theta1_3 = fft(theta1_3)
fft_theta2_3 = fft(theta2_3)
frecuencias = np.fft.fftfreq(len(t_eval), d=t_eval[1] - t_eval[0])

# Graficar los resultados
plt.figure(figsize=(12, 6))
#grafica de theta1 y theta2
plt.subplot(2, 1, 1)
plt.plot(t_eval, theta1, label='θ1(t)')
plt.plot(t_eval, theta2, label='θ2(t)')
plt.title('Solución de las ecuaciones en el dominio del tiempo con θ1(0) = {} rad y θ2(0) = {} rad'.format(round(angulo1, 4), round(angulo2, 4)))
plt.xlabel('Tiempo [s]')
plt.ylabel('Ángulo [rad]')
plt.legend()

#grafica de theta1_1 y theta2_2
plt.subplot(2, 1, 2)
plt.plot(t_eval, theta1_1, label='θ1(t)')
plt.plot(t_eval, theta2_1, label='θ2(t)')
plt.title('Solución de las ecuaciones en el dominio del tiempo con θ1_1(0) = {} rad y θ2_1(0) = {} rad'.format(round(angulo1, 4), round(angulo2, 4)))
plt.xlabel('Tiempo [s]')
plt.ylabel('Ángulo [rad]')
plt.legend()

plt.tight_layout()
plt.show()

#segunda grafica: theta1_2 y theta2_2 cuando el resorte está en la mitad de la varilla 
plt.figure(figsize=(12, 6))
#grafica de theta1 y theta2
plt.subplot(2, 1, 1)
plt.plot(t_eval, theta1, label='θ1(t)')
plt.plot(t_eval, theta2, label='θ2(t)')
plt.title('Solución de las ecuaciones en el dominio del tiempo con θ1(0) = {} rad y θ2(0) = {} rad'.format(round(angulo1, 4), round(angulo2, 4)))
plt.xlabel('Tiempo [s]')
plt.ylabel('Ángulo [rad]')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t_eval, theta1_2, label='θ1(t)')
plt.plot(t_eval, theta2_2, label='θ2(t)')
plt.title('Solución de las ecuaciones cuando el resorte está a la mitad de las varillas con θ1_2(0) = {} rad y θ2_2(0) = {} rad'.format(round(angulo1, 4), round(angulo2, 4)))
plt.xlabel('Tiempo [s]')
plt.ylabel('Ángulo [rad]')
plt.legend()

plt.tight_layout()
plt.show()

#tercer grafica
plt.figure(figsize=(12, 6))
#grafica de theta1 y theta2
plt.subplot(2, 1, 1)
plt.plot(t_eval, theta1, label='θ1(t)')
plt.plot(t_eval, theta2, label='θ2(t)')
plt.title('Solución de las ecuaciones en el dominio del tiempo con θ1(0) = {} rad y θ2(0) = {} rad'.format(pequeño1, pequeño2))
plt.xlabel('Tiempo [s]')
plt.ylabel('Ángulo [rad]')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t_eval, theta1_3, label='θ1(t)')
plt.plot(t_eval, theta2_3, label='θ2(t)')
plt.title('Solución de las ecuaciones para angulos pequeños θ1_3(0) = {} rad y θ2_3(0) = {} rad'.format(pequeño1,pequeño2))
plt.xlabel('Tiempo [s]')
plt.ylabel('Ángulo [rad]')
plt.legend()

plt.tight_layout()
plt.show()

#grafica que puede llegar a evidenciar el caos
plt.figure(figsize=(12, 6))
#grafica de theta1 y theta2
plt.subplot(2, 1, 1)
plt.plot(t_eval, theta1, label='θ1(t)')
plt.plot(t_eval, theta1_condicones_modificadas, label='θ1_modified(t)')
plt.title('Comparacion de las soluciones para una variacion de condiciones iniciales de 0.0001 con θ1(0) = {} rad y θ1_modified(0) = {} rad'.format(round(angulo1,5), round(angulo1 + 0.0001, 5)))
plt.xlabel('Tiempo [s]')
plt.ylabel('Ángulo [rad]')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t_eval, resta1, label='θ1_modified(t)-θ1(t)')
plt.title('Resta de las soluciones para una variacion de condiciones iniciales de 0.0001, θ(0) = {} rad y θ(0) = {} rad'.format(round(angulo1,5), round(angulo1 + 0.0001, 5)))
plt.xlabel('Tiempo [s]')
plt.ylabel('Ángulo [rad]')
plt.legend()

plt.tight_layout()
plt.show()

#apartado 2: gráficas de transformada de furier
plt.figure(figsize=(12, 6))
#Graficar la FFT de theta1 y theta2
plt.subplot(2, 1, 1)
plt.plot(frecuencias, np.abs(fft_theta1), label='FFT de θ1')
plt.plot(frecuencias, np.abs(fft_theta2), label='FFT de θ2')
plt.title('Transformada Rápida de Fourier (FFT)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud')
plt.legend()
3
#Graficar la FFT de theta1_1 y theta2_1
plt.subplot(2, 1, 2)
plt.plot(frecuencias, np.abs(fft_theta1_1), label='FFT de θ1_1')
plt.plot(frecuencias, np.abs(fft_theta2_1), label='FFT de θ2_1')
plt.title('Transformada Rápida de Fourier  (FFT) para el caso de masas y longitudes de varilla distintas')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud')
plt.legend()

plt.tight_layout()
plt.show()

#grafica 
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(frecuencias, np.abs(fft_theta1_2), label='FFT de θ1_2')
plt.plot(frecuencias, np.abs(fft_theta2_2), label='FFT de θ2_2')
plt.title('Transformada Rápida de Fourier (FFT) para el caso de resorte a la mitad de las varillas')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud')
plt.legend()

plt.subplot(2, 1,2)
plt.plot(frecuencias, np.abs(fft_theta1_3), label='FFT de θ1_3')
plt.plot(frecuencias, np.abs(fft_theta2_3), label='FFT de θ2_3')
plt.title('Transformada Rápida de Fourier (FFT) para la aproximacion de angulos pequeños')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud')
plt.legend()

plt.tight_layout()
plt.show()

# cuarta: Espectrogramas
plt.figure(figsize=(12, 6))

# Graficar el espectrograma para θ1
plt.subplot(2, 1, 1)
plt.specgram(theta1, Fs=1/(t_eval[1] - t_eval[0]), NFFT=128, noverlap=64, cmap='viridis')
plt.title('Espectrograma de θ1')
plt.xlabel('Tiempo [s]')
plt.ylabel('Frecuencia [Hz]')

# Graficar el espectrograma para θ2
plt.subplot(2, 1, 2)
plt.specgram(theta2, Fs=1/(t_eval[1] - t_eval[0]), NFFT=128, noverlap=64, cmap='viridis')
plt.title('Espectrograma de θ2')
plt.xlabel('Tiempo [s]')
plt.ylabel('Frecuencia [Hz]')



#Agregar el espectrogtrama para los demas........
plt.tight_layout()
plt.show()