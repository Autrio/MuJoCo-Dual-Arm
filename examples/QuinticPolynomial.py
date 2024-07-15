# import numpy as np

# class QuinticPolynomial:

#     def __init__(self, init_pos, init_vel, init_accel, final_pos, final_vel, final_accel, dist):

#         # Derived coefficients
#         self.a_0 = init_pos
#         self.a_1 = init_vel
#         self.a_2 = init_accel / 2.0

#         # Solve the linear equation (Ax = B)
#         A = np.array([[dist ** 3,      dist ** 4,        dist ** 5], 
#                      [3 * dist ** 2,   4 * dist ** 3,    5 * dist ** 4],
#                      [6 * dist,        12 * dist ** 2,   20 * dist ** 3]])

#         B = np.array([final_pos - self.a_0 - self.a_1 * dist - self.a_2 * dist ** 2, final_vel - self.a_1 - init_accel * dist, final_accel - init_accel])

#         x = np.linalg.solve(A, B)

#         self.a_3 = x[0]
#         self.a_4 = x[1]
#         self.a_5 = x[2]

#     def calc_point(self, s):

#         xs = self.a_0 + self.a_1 * s + self.a_2 * s ** 2 + self.a_3 * s ** 3 + self.a_4 * s ** 4 + self.a_5 * s ** 5

#         return xs

#     def calc_first_derivative(self, s):

#         xs = self.a_1 + 2 * self.a_2 * s + 3 * self.a_3 * s ** 2 + 4 * self.a_4 * s ** 3 + 5 * self.a_5 * s ** 4

#         return xs

#     def calc_second_derivative(self, s):

#         xs = 2 * self.a_2 + 6 * self.a_3 * s + 12 * self.a_4 * s ** 2 + 20 * self.a_5 * s ** 3

#         return xs

#     def calc_third_derivative(self, s):

#         xs = 6 * self.a_3 + 24 * self.a_4 * s + 60 * self.a_5 * s ** 2

#         return xs

# def quintic_polynomial_planner(x_i, y_i, yaw_i, v_i, a_i, x_f, y_f, yaw_f, v_f, a_f, max_accel, max_jerk, ds):
    
#     # Calculate the velocity boundary conditions based on vehicle's orientation
#     v_xi = v_i * np.cos(yaw_i)
#     v_xf = v_f * np.cos(yaw_f)
#     v_yi = v_i * np.sin(yaw_i)
#     v_yf = v_f * np.sin(yaw_f)

#     # Calculate the acceleration boundary conditions based on vehicle's orientation
#     a_xi = a_i * np.cos(yaw_i)
#     a_xf = a_f * np.cos(yaw_f)
#     a_yi = a_i * np.sin(yaw_i)
#     a_yf = a_f * np.sin(yaw_f)

#     for S in np.arange(1.0, 1000.0, 1.0):
#         # Initialise the class
#         xqp = QuinticPolynomial(x_i, v_xi, a_xi, x_f, v_xf, a_xf, S)
#         yqp = QuinticPolynomial(y_i, v_yi, a_yi, y_f, v_yf, a_yf, S)

#         # Instantiate/clear the arrays
#         x = []
#         y = []
#         v = []
#         a = []
#         j = []
#         yaw = []

#         for s in np.arange(0.0, S + ds, ds):
#             # Solve for position (m)
#             x.append(xqp.calc_point(s))
#             y.append(yqp.calc_point(s))

#             # Solve for velocity (m/s)
#             v_x = xqp.calc_first_derivative(s)
#             v_y = yqp.calc_first_derivative(s)
#             v.append(np.hypot(v_x, v_y))

#             # Solve for orientation (rad)
#             yaw.append(np.arctan2(v_y, v_x))

#             # Solve for acceleration (m/s^2)
#             a_x = xqp.calc_second_derivative(s)
#             a_y = yqp.calc_second_derivative(s)

#             if len(v) >= 2 and v[-1] - v[-2] < 0.0:
#                 a.append(-1.0 * np.hypot(a_x, a_y))

#             else:
#                 a.append(np.hypot(a_x, a_y))
        
#             # Solve for jerk (m/s^3)
#             j_x = xqp.calc_third_derivative(s)
#             j_y = yqp.calc_third_derivative(s)
            
#             if len(a) >= 2 and a[-1] - a[-2] < 0.0:
#                 j.append(-1 * np.hypot(j_x, j_y))

#             else:
#                 j.append(np.hypot(j_x, j_y))

#         if max([abs(i) for i in a]) <= max_accel and max([abs(i) for i in j]) <= max_jerk:
#             break

#     return x, y, v, a, j, yaw


"""

Quintic Polynomials Planner

author: Atsushi Sakai (@Atsushi_twi)

Ref:

- [Local Path planning And Motion Control For Agv In Positioning](http://ieeexplore.ieee.org/document/637936/)

"""

import math

import matplotlib.pyplot as plt
import numpy as np

# parameter
MAX_T = 100.0  # maximum time to the goal [s]
MIN_T = 5.0  # minimum time to the goal[s]

show_animation = True


class QuinticPolynomial:

    def __init__(self, xs, vxs, axs, xe, vxe, axe, time):
        # calc coefficient of quintic polynomial
        # See jupyter notebook document for derivation of this equation.
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[time ** 3, time ** 4, time ** 5],
                      [3 * time ** 2, 4 * time ** 3, 5 * time ** 4],
                      [6 * time, 12 * time ** 2, 20 * time ** 3]])
        b = np.array([xe - self.a0 - self.a1 * time - self.a2 * time ** 2,
                      vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2

        return xt


def quintic_polynomials_planner(sx, sy, syaw, sv, sa, gx, gy, gyaw, gv, ga, max_accel, max_jerk, dt):
    """
    quintic polynomial planner

    input
        s_x: start x position [m]
        s_y: start y position [m]
        s_yaw: start yaw angle [rad]
        sa: start accel [m/ss]
        gx: goal x position [m]
        gy: goal y position [m]
        gyaw: goal yaw angle [rad]
        ga: goal accel [m/ss]
        max_accel: maximum accel [m/ss]
        max_jerk: maximum jerk [m/sss]
        dt: time tick [s]

    return
        time: time result
        rx: x position result list
        ry: y position result list
        ryaw: yaw angle result list
        rv: velocity result list
        ra: accel result list

    """

    vxs = sv * math.cos(syaw)
    vys = sv * math.sin(syaw)
    vxg = gv * math.cos(gyaw)
    vyg = gv * math.sin(gyaw)

    axs = sa * math.cos(syaw)
    ays = sa * math.sin(syaw)
    axg = ga * math.cos(gyaw)
    ayg = ga * math.sin(gyaw)

    time, rx, ry, ryaw, rv, ra, rj = [], [], [], [], [], [], []

    for T in np.arange(MIN_T, MAX_T, MIN_T):
        xqp = QuinticPolynomial(sx, vxs, axs, gx, vxg, axg, T)
        yqp = QuinticPolynomial(sy, vys, ays, gy, vyg, ayg, T)

        time, rx, ry, ryaw, rv, ra, rj = [], [], [], [], [], [], []

        for t in np.arange(0.0, T + dt, dt):
            time.append(t)
            rx.append(xqp.calc_point(t))
            ry.append(yqp.calc_point(t))

            vx = xqp.calc_first_derivative(t)
            vy = yqp.calc_first_derivative(t)
            v = np.hypot(vx, vy)
            yaw = math.atan2(vy, vx)
            rv.append(v)
            ryaw.append(yaw)

            ax = xqp.calc_second_derivative(t)
            ay = yqp.calc_second_derivative(t)
            a = np.hypot(ax, ay)
            if len(rv) >= 2 and rv[-1] - rv[-2] < 0.0:
                a *= -1
            ra.append(a)

            jx = xqp.calc_third_derivative(t)
            jy = yqp.calc_third_derivative(t)
            j = np.hypot(jx, jy)
            if len(ra) >= 2 and ra[-1] - ra[-2] < 0.0:
                j *= -1
            rj.append(j)

        if max([abs(i) for i in ra]) <= max_accel and max([abs(i) for i in rj]) <= max_jerk:
            print("find path!!")
            break

    if show_animation:  # pragma: no cover
        for i, _ in enumerate(time):
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])
            plt.grid(True)
            plt.axis("equal")
            plot_arrow(sx, sy, syaw)
            plot_arrow(gx, gy, gyaw)
            plot_arrow(rx[i], ry[i], ryaw[i])
            plt.title("Time[s]:" + str(time[i])[0:4] +
                      " v[m/s]:" + str(rv[i])[0:4] +
                      " a[m/ss]:" + str(ra[i])[0:4] +
                      " jerk[m/sss]:" + str(rj[i])[0:4],
                      )
            plt.pause(0.001)

    return time, rx, ry, ryaw, rv, ra, rj


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):  # pragma: no cover
    """
    Plot arrow
    """

    if not isinstance(x, float):
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)


def main():
    print(__file__ + " start!!")

    sx = 10.0  # start x position [m]
    sy = 10.0  # start y position [m]
    syaw = np.deg2rad(10.0)  # start yaw angle [rad]
    sv = 1.0  # start speed [m/s]
    sa = 0.1  # start accel [m/ss]
    gx = 30.0  # goal x position [m]
    gy = -10.0  # goal y position [m]
    gyaw = np.deg2rad(20.0)  # goal yaw angle [rad]
    gv = 1.0  # goal speed [m/s]
    ga = 0.1  # goal accel [m/ss]
    max_accel = 1.0  # max accel [m/ss]
    max_jerk = 0.5  # max jerk [m/sss]
    dt = 0.1  # time tick [s]

    time, x, y, yaw, v, a, j = quintic_polynomials_planner(
        sx, sy, syaw, sv, sa, gx, gy, gyaw, gv, ga, max_accel, max_jerk, dt)

    if show_animation:  # pragma: no cover
        plt.plot(x, y, "-r")

        plt.subplots()
        plt.plot(time, [np.rad2deg(i) for i in yaw], "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("Yaw[deg]")
        plt.grid(True)

        plt.subplots()
        plt.plot(time, v, "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("Speed[m/s]")
        plt.grid(True)

        plt.subplots()
        plt.plot(time, a, "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("accel[m/ss]")
        plt.grid(True)

        plt.subplots()
        plt.plot(time, j, "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("jerk[m/sss]")
        plt.grid(True)

        plt.show()


if __name__ == '__main__':
    main()