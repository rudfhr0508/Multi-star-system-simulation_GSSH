import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# ------------------------------------------------------------------- #

m1 = 1.0  
m2 = 1.0 

# 초기 위치
p1 = np.array([0.5, 0.0, 0.0])  
p2 = np.array([-0.5, 0.0, 0.0]) 

# 초기 속도 
v1 = np.array([0.0, 0.5, 0.0])   
v2 = np.array([0.0, -0.5, 0.0])  

initial_conditions = np.array([p1, p2, v1, v2]).ravel()

# ------------------------------------------------------------------- #

def system_odes(t, S, m1, m2):
    p1, p2 = S[0:3], S[3:6]
    dp1_dt, dp2_dt = S[6:9], S[9:12]

    f1, f2 = dp1_dt, dp2_dt

    r12 = np.linalg.norm(p2 - p1)
    df1_dt = m2 * (p2 - p1) / r12**3  
    df2_dt = m1 * (p1 - p2) / r12**3  

    return np.array([f1, f2, df1_dt, df2_dt]).ravel()

# ------------------------------------------------------------------- #

# 시간 설정
time_s, time_e = 0, 20
t_points = np.linspace(time_s, time_e, 2001)

# 수치 적분
solution = solve_ivp(
    fun=system_odes,
    t_span=(time_s, time_e),
    y0=initial_conditions,
    t_eval=t_points,
    args=(m1, m2)
)

# 결과 추출
t_sol = solution.t
p1x_sol = solution.y[0]
p1y_sol = solution.y[1]
p1z_sol = solution.y[2]

p2x_sol = solution.y[3]
p2y_sol = solution.y[4]
p2z_sol = solution.y[5]

# ------------------------------------------------------------------- #

# 3D 그래프 생성
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# 궤적 플롯
planet1_plt, = ax.plot(p1x_sol, p1y_sol, p1z_sol, 'green', label='Planet 1', linewidth=1)
planet2_plt, = ax.plot(p2x_sol, p2y_sol, p2z_sol, 'red', label='Planet 2', linewidth=1)

# 현재 위치 점
planet1_dot, = ax.plot([p1x_sol[-1]], [p1y_sol[-1]], [p1z_sol[-1]], 'o', color='green', markersize=6)
planet2_dot, = ax.plot([p2x_sol[-1]], [p2y_sol[-1]], [p2z_sol[-1]], 'o', color='red', markersize=6)

# 그래프 설정
ax.set_title("쌍성계 운동 시뮬레이션")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.grid()
plt.legend()

# ------------------------------------------------------------------- #

# 애니메이션 함수
from matplotlib.animation import FuncAnimation

def update(frame):
    lower_lim = max(0, frame - 100)
    print(f"Progress: {(frame+1)/len(t_points):.1%} | 100.0 %", end='\r')

    x_current_1 = p1x_sol[lower_lim:frame+1]
    y_current_1 = p1y_sol[lower_lim:frame+1]
    z_current_1 = p1z_sol[lower_lim:frame+1]

    planet1_plt.set_data(x_current_1, y_current_1)
    planet1_plt.set_3d_properties(z_current_1)
    planet1_dot.set_data([x_current_1[-1]], [y_current_1[-1]])
    planet1_dot.set_3d_properties([z_current_1[-1]])

    x_current_2 = p2x_sol[lower_lim:frame+1]
    y_current_2 = p2y_sol[lower_lim:frame+1]
    z_current_2 = p2z_sol[lower_lim:frame+1]

    planet2_plt.set_data(x_current_2, y_current_2)
    planet2_plt.set_3d_properties(z_current_2)
    planet2_dot.set_data([x_current_2[-1]], [y_current_2[-1]])
    planet2_dot.set_3d_properties([z_current_2[-1]])

    return planet1_plt, planet1_dot, planet2_plt, planet2_dot

# 애니메이션 생성
animation = FuncAnimation(fig, update, frames=range(0, len(t_points), 2), interval=10, blit=True)
plt.show()