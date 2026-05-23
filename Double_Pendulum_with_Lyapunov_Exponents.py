import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import animation

# ======================
# PARAMETERS
# ======================
g = 9.81
L1, L2 = 1.0, 1.0
m1, m2 = 1.0, 1.0

t_span = (0, 15)
t_eval = np.linspace(0, 15, 2000)

# ======================
# EQUATIONS OF MOTION
# ======================
def pendulum(t, state):
    th1, w1, th2, w2 = state
    d = th1 - th2

    den = (2*m1 + m2 - m2*np.cos(2*th1 - 2*th2))

    dw1 = (
        -g*(2*m1 + m2)*np.sin(th1)
        - m2*g*np.sin(th1 - 2*th2)
        - 2*np.sin(d)*m2*(w2**2*L2 + w1**2*L1*np.cos(d))
    ) / (L1 * den)

    dw2 = (
        2*np.sin(d) *
        (w1**2*L1*(m1+m2) + g*(m1+m2)*np.cos(th1) + w2**2*L2*m2*np.cos(d))
    ) / (L2 * den)

    return [w1, dw1, w2, dw2]

# ======================
# INITIAL CONDITIONS
# ======================
state0 = [np.pi/2, 0, np.pi/2, 0]
state0_p = [np.pi/2 + 1e-6, 0, np.pi/2, 0]

sol = solve_ivp(pendulum, t_span, state0, t_eval=t_eval)
sol_p = solve_ivp(pendulum, t_span, state0_p, t_eval=t_eval)

# ======================
# LYAPUNOV EXPONENT (IMPROVED)
# ======================
dx = sol.y - sol_p.y
dist = np.sqrt(np.sum(dx**2, axis=0))
dist = np.where(dist < 1e-12, 1e-12, dist)

lyap = np.mean(np.log(dist[1:] / dist[:-1])) / (t_eval[1] - t_eval[0])

# ======================
# ENERGY FUNCTION
# ======================
def energy(th1, w1, th2, w2):
    V = -(m1+m2)*g*L1*np.cos(th1) - m2*g*L2*np.cos(th2)
    K = 0.5*m1*(L1*w1)**2 + 0.5*m2*((L1*w1)**2 + (L2*w2)**2 +
        2*L1*L2*w1*w2*np.cos(th1-th2))
    return K + V

E = energy(sol.y[0], sol.y[1], sol.y[2], sol.y[3])

# ======================
# PLOTS
# ======================
plt.figure(figsize=(10,5))
plt.plot(E)
plt.title("Energy vs Time (should be roughly conserved)")
plt.xlabel("Time step")
plt.ylabel("Energy")
plt.grid()
plt.show()

plt.figure(figsize=(8,6))
plt.plot(sol.y[0], sol.y[1], label="Pendulum 1")
plt.plot(sol.y[2], sol.y[3], label="Pendulum 2")
plt.title(f"Phase Space (Lyapunov ≈ {lyap:.4f})")
plt.xlabel("Angle")
plt.ylabel("Angular velocity")
plt.legend()
plt.grid()
plt.show()

# ======================
# ANIMATION SETUP
# ======================
x1 = L1*np.sin(sol.y[0])
y1 = -L1*np.cos(sol.y[0])
x2 = x1 + L2*np.sin(sol.y[2])
y2 = y1 - L2*np.cos(sol.y[2])

fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_title("Double Pendulum Animation")

line, = ax.plot([], [], 'o-', lw=2)

def update(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]
    line.set_data(thisx, thisy)
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(t_eval), interval=15, blit=True)

plt.show()

# ======================
# OUTPUT
# ======================
print("Lyapunov Exponent ≈", lyap)