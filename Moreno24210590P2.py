"""
Práctica 2: Sistema cardiovascular

Departamento de Ingeniería Eléctrica y Electrónica, Ingeniería Biomédica
Tecnológico Nacional de México [TecNM - Tijuana]
Blvd. Alberto Limón Padilla s/n, C.P. 22454, Tijuana, B.C., México

Nombre del alumno: Leonardo Zaid Moreno Zavala
Número de control: 24210590
Correo institucional: l24210590@tectijuana.edu.mx

Asignatura: Modelado de Sistemas Fisiológicos
Docente: Dr. Paul Antonio Valle Trujillo; paul.valle@tectijuana.edu.mx
"""

# librerias
import numpy as np
import math as m
import control as ctrl
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal

# datos generales de la simulacion
u = np.array(pd.read_excel('signal.xlsx',header=None)).flatten()

x0,t0,tend,dt,w,h = 0,0,15,1e-2,10,5
N = round((tend-t0)/dt) + 1
t = np.linspace(t0,tend,N)

u = np.interp(t, np.linspace(t0,tend,len(u)), u)

signals = ["Lazo abierto","Hipotenso","Hipertenso"]

def caso(Z,L,R,C):
    num=[L*R,R*Z]
    den=[C*L*R*Z,L*R+L*Z,R*Z]
    sys = ctrl.tf(num,den)
    L = np.roots(den)
    return sys,L

#Normotenso
Z,L,R,C = 0.010,0.001,1.00,1.000
sysnormo,L = caso(Z,L,R,C)
print(f'Funcion de transferencia del normotenso:{sysnormo}')
print(f'Polos del sistema : L1 = {L[0]:.3e}, L2 = {L[1]:.3e}/n')

#Hipertenso
Z,L,R,C = 0.010,0.001,1.00,1.500
syshiper,L = caso(Z,L,R,C)
print(f'Funcion de transferencia del Hipertenso:{syshiper}')
print(f'Polos del sistema : L1 = {L[0]:.3e}, L2 = {L[1]:.3e}\n')

#Hipotenso
Z,L,R,C = 0.010,0.001,1.00,3.500
syshipo,L = caso(Z,L,R,C)
print(f'Funcion de transferencia del hipotenso:{syshipo}')
print(f'Polos del sistema : L1 = {L[0]:3e}, L2 = {L[1]:3e}\n')

#Funciones
def controlador(kP,kI,kD,sys):
    Cr = 1e-6
    Re = 1/(kI*Cr)
    Rr = kP*Re
    Ce = kD/Rr
    numPID = [Re*Rr*Ce*Cr,(Re*Ce + Rr*Cr),1]
    denPID = [Re*Cr,0]
    PID = ctrl.tf(numPID,denPID)
    sysPID = ctrl.feedback(ctrl.series(PID,sys),1,sign=-1)
    return sysPID

hipoPID = controlador(458.483,1112094.736,0.00627,syshipo)
print(f'Funcion de transferencia del hipotenso en el lazo cerrado: {hipoPID}')
hiperPID = controlador(126.595,293327.703,0.00140,syshiper)
print(f'Funcion de transferencia del hipertenso en el lazo cerrado: {hiperPID}')

#Colores
clr1 = np.array([255, 0, 0])/255
clr2 = np.array([0, 0, 255])/255
clr3 = np.array([0, 255, 0])/255

#Respuestas en lazo abierto
_,PpN = ctrl.forced_response(sysnormo,t,u,x0)
_,PpL = ctrl.forced_response(syshipo,t,u,x0)
_,PpH = ctrl.forced_response(syshiper,t,u,x0)

#Respuestas en lazo cerrado
_,PIDL = ctrl.forced_response(hipoPID,t,PpN,x0)
_,PIDH = ctrl.forced_response(hiperPID,t,PpN,x0)

# Respuestas: Simulaciones numéricas
#Lazo abierto
fg1 = plt.figure()
fg1.set_size_inches(w,h)
plt.plot(t,PpH,'-',linewidth=1,color=clr1,label='Pp(t): Hipertenso')
plt.plot(t,PpN,'--',linewidth=1,color=clr2,label='Pp(t): Normotenso')
plt.plot(t,PpL,':',linewidth=1.5,color=clr3,label='Pp(t): Hipotenso')
plt.grid(False)

plt.title('Lazo abierto')
plt.xlim(0,15); plt.xticks(np.arange(0,16,1))
plt.ylim(-0.4,1.2)
plt.xlabel('t [s]')
plt.ylabel('Pp(t)')
plt.legend(bbox_to_anchor=(0.5,-0.25),loc='center',ncol=3,frameon=False)
plt.subplots_adjust(bottom=0.3)
fg1.savefig('Lazo_abierto_python.pdf',bbox_inches='tight')
plt.show()

#Hipertenso
fg2 = plt.figure()
fg2.set_size_inches(w,h)
plt.plot(t,PpH, '-', linewidth=1,color=clr1,label='Pp(t): Hipertenso')
plt.plot(t,PpN, '--',linewidth=1,  color=clr2,label='Pp(t): Normotenso')
plt.plot(t,PIDH,':',linewidth=1.5,  color=clr3,label='PID(t): Hipertenso')
plt.grid(False)

plt.title('Hipertenso con controlador')
plt.xlim(0,15); plt.xticks(np.arange(0,16,1))
plt.ylim(-0.4,1.2)
plt.xlabel('t [s]')
plt.ylabel('Pp(t)')
plt.legend(bbox_to_anchor=(0.5,-0.25),loc='center',ncol=3,frameon=False)
plt.subplots_adjust(bottom=0.3)
fg2.savefig('Hipertenso_python.pdf',bbox_inches='tight')
plt.show()

#Hipotenso
fg3 = plt.figure()
fg3.set_size_inches(w,h)
plt.plot(t,PpL, '-', linewidth=1,color=clr1,label='Pp(t): Hipotenso')
plt.plot(t,PpN, '--',linewidth=1,  color=clr2,label='Pp(t): Normotenso')
plt.plot(t,PIDL,':',linewidth=1.5,  color=clr3,label='PID(t): Hipotenso')
plt.grid(False)

plt.title('Hipotenso con controlador')
plt.xlim(0,15); plt.xticks(np.arange(0,16,1))
plt.ylim(-0.4,1.2)
plt.xlabel('t [s]')
plt.ylabel('Pp(t)')
plt.legend(bbox_to_anchor=(0.5,-0.25),loc='center',ncol=3,frameon=False)
plt.subplots_adjust(bottom=0.3)
fg3.savefig('Hipotenso_python.pdf',bbox_inches='tight')
plt.show()
