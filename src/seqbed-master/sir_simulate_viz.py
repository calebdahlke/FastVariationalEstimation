'''
  SIR_SIMULATE_VIZ - Simulate SIR model for visualization in paper.  This is
    purely visualizatoin code.
'''

import numpy as np
import matplotlib.pyplot as plt

## PARAMETERS ##
#----------------#
dt = 0.01         # Time delta
d = 3.0           # Total time to simulate
N = 50            # Population size
S0 = N-1          # Initial susceptible
I0 = 1            # Initial infected
R0 = 0            # Initial recovered
beta = 0.14        # Beta parameter
gamma = 0.01       # Gamma parameter
#----------------#

# beta = np.random.uniform(0,0.5)
# gamma = np.random.uniform(0,0.5)
# print('beta: {0} \tgamma: {1}'.format(beta, gamma))

times = np.arange(0 + dt, d + dt, dt)

# initialize  stuff
S = np.zeros(len(times), dtype='int')
I = np.zeros(len(times), dtype='int')
R = np.zeros(len(times), dtype='int')

# simulate
for t, _ in enumerate(times):

  # initial conditions
  if t == 0:
    S[t] = S0
    I[t] = I0
    R[t] = R0
    continue

  # general case
  pinf = beta * I[t-1] / N
  dI = np.random.binomial(S[t-1], pinf)  
  dR = np.random.binomial(I[t-1], gamma)

  S[t] = S[t-1] - dI
  I[t] = I[t-1] + dI - dR
  R[t] = R[t-1] + dR

# plot stuff
plt.figure()
plt.plot(times, S, label='S(t)')
plt.plot(times, I, label='I(t)')
plt.plot(times, R, label='R(t)')
plt.xlabel('Time (s)', fontsize=12)
plt.title('SIR Simulation', fontsize=14)
plt.legend(fontsize=12)
plt.show()

