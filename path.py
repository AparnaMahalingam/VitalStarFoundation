import numpy as np
import matplotlib.pyplot as plt
from scipy.special import fresnel
from numpy import linspace


class Path(object):
    def __init__(self, path_coordinates):
        self.a = path_coordinates
        self.velocity, self.acceleration, self.t_component, self.n_component\
            , self.curvature, self.speed, self.d2x_dt2, self.d2y_dt2 = self.get_acceleration()

    def plot_curvature(self):

        #plt.axes().set_aspect("equal")

        fig = plt.figure()
        fig.suptitle('curvature', fontsize=14, fontweight='bold')
        plt.scatter(self.a[:, 0], self.a[:, 1])
        # annotate with the curvature at each sample point
        for i, txt in enumerate(self.curvature):
            plt.annotate(str(txt)[0:5], (self.a[i, 0], self.a[i, 1]))
        plt.show()

    def plot_speed(self):
        fig = plt.figure()
        fig.suptitle('speed', fontsize=14, fontweight='bold')
        plt.scatter(self.a[:, 0], self.a[:, 1])

        for i, txt in enumerate(self.speed):
            plt.annotate(str(txt)[0:5], (self.a[i, 0], self.a[i, 1]))
        plt.show()

    def plot_IMU_simulation(self, co_var=0.05):
        x_prime_vec, y_prime_vec = self.get_simulated_path_from_acceleration(Dt=1, co_var=co_var)

        fig = plt.figure()
        fig.suptitle('IMU dead reckoning', fontsize=14, fontweight='bold')
        plt.scatter(self.a[:, 0], self.a[:, 1], color='g')
        plt.scatter(x_prime_vec, y_prime_vec, color='r')
        # plt.scatter(x_pred_vec, y_pred_vec, color='b')
        plt.show()

        #plotting the errors with the path
        error_vec_x = np.array(self.a[:, 0] - np.array(x_prime_vec))
        error_vec_y = np.array(self.a[:, 1] - np.array(y_prime_vec))
        error_vec_a = np.sqrt(error_vec_x * error_vec_x + error_vec_y * error_vec_y)
        number_of_samples = self.a.shape[0]

        fig = plt.figure()
        fig.suptitle('IMU dead reckoning errors along the path', fontsize=14, fontweight='bold')
        plt.scatter(range(0, number_of_samples), error_vec_a)
        plt.show()

    def plot_IMU_errors(self, Monte_Carlo = 100, co_var=0.05):

        max_errors_2 = []
        mc_times = Monte_Carlo
        for i in range(1, mc_times):
            x_prime_vec, y_prime_vec = self.get_simulated_path_from_acceleration(Dt=1, co_var=co_var)
            error_vec_x = np.array(self.a[:, 0] - np.array(x_prime_vec))
            error_vec_y = np.array(self.a[:, 1] - np.array(y_prime_vec))
            error_vec_a = np.sqrt(error_vec_x * error_vec_x + error_vec_y * error_vec_y)
            max_errors_2.append(max(error_vec_a))

        fig = plt.figure()
        fig.suptitle('IMU dead reckoning positioning errors (in meters)', fontsize=14, fontweight='bold')
        plt.scatter(range(1, mc_times), max_errors_2)
        plt.show()

        return max_errors_2

    def get_acceleration(self):
        '''
        input: a is the x,y coordinates of a numpy array of a path sampled with a given time interval
        for example:
        array([[  0.  ,   0.  ],
       [  0.3 ,   0.  ],
       [  1.25,  -0.1 ],
       [  2.1 ,  -0.9 ],
       [  2.85,  -2.3 ],
       [  3.8 ,  -3.95],
       [  5.  ,  -5.75],
       [  6.4 ,  -7.8 ],
       [  8.05,  -9.9 ],
       [  9.9 , -11.6 ],
       [ 12.05, -12.85],
       [ 14.25, -13.7 ],
       [ 16.5 , -13.8 ],
       [ 19.25, -13.35],
       [ 21.3 , -12.2 ],
       [ 22.8 , -10.5 ],
       [ 23.55,  -8.15],
       [ 22.95,  -6.1 ],
       [ 21.35,  -3.95],
       [ 19.1 ,  -1.9 ]])
        :return:
        '''
        a = self.a
        dx_dt = np.gradient(a[:, 0])
        dy_dt = np.gradient(a[:, 1])
        velocity = np.array([ [dx_dt[i], dy_dt[i]] for i in range(dx_dt.size)])
        ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)

        speed = ds_dt
        # tangent is the direction of the velocity vector, while speed is the magnitude of the velocity vector

        tangent = np.array([1 / ds_dt] * 2).transpose() * velocity
        tangent_x = tangent[:, 0]
        tangent_y = tangent[:, 1]

        deriv_tangent_x = np.gradient(tangent_x)
        deriv_tangent_y = np.gradient(tangent_y)

        dT_dt = np.array([[deriv_tangent_x[i], deriv_tangent_y[i]] for i in range(deriv_tangent_x.size)])

        length_dT_dt = np.sqrt(deriv_tangent_x * deriv_tangent_x + deriv_tangent_y * deriv_tangent_y)

        normal = np.array([1 / length_dT_dt] * 2).transpose() * dT_dt

        #when traveling on a linear path, normal is zero
        normal1 = np.nan_to_num(normal)


        d2s_dt2 = np.gradient(ds_dt)
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)

        curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
        t_component = np.array([d2s_dt2] * 2).transpose()
        n_component = np.array([curvature * ds_dt * ds_dt] * 2).transpose()

        acceleration = t_component * tangent + n_component * normal1
        return velocity, acceleration, t_component, n_component, curvature, speed, d2x_dt2, d2y_dt2

    def get_simulated_path_from_acceleration(self, Dt = 1, co_var = 0.05):
        '''

        :param co_var:
        :return: x_prime_vec: the vector of x_prime coordidate, predicted x position based on acceleration meas
                 y_prime_vec: the vector of y_prime coordinate, predicted y position based on acceleration meas
        '''
        # main simulation loop
        # initial condition
        v_x_0 = self.velocity[0, 0]
        v_y_0 = self.velocity[0, 1]
        x_0 = self.a[0, 0]
        y_0 = self.a[0, 1]
        alpha_x_0 = self.acceleration[0, 0]
        alpha_y_0 = self.acceleration[0, 1]

        # error condition, assuming Gaussian error on alpha_x and alpha_y with co_var = sd/|alpha| = 0.05
        # therefore sd = co_var * |alpha|
        # then take error = Gaussian(0, sd^2)
        # then the measured alpha_m = alpha + error

        sd_x_0 = abs(alpha_x_0) * co_var
        error_x = np.random.normal(loc=0, scale=sd_x_0, size=1)
        alpha_x_0_m = alpha_x_0 + error_x
        sd_y_0 = abs(alpha_y_0) * co_var
        error_y = np.random.normal(loc=0, scale=sd_y_0, size=1)
        alpha_y_0_m = alpha_y_0 + error_y

        # main loop to create all positions
        # x_prime are the x vector of predicted positions
        # y_prime are the y vector of predicted positions

        x_prime_vec = [x_0]
        y_prime_vec = [y_0]
        x_pred_vec = [x_0]
        y_pred_vec = [y_0]

        # point to move from
        x_pos = x_0
        y_pos = y_0
        v_x_pos = v_x_0
        v_y_pos = v_y_0
        alpha_x_pos = alpha_x_0_m
        alpha_y_pos = alpha_y_0_m

        for i in range(1, len(self.a)):
            x_prime = x_pos + 0.5 * alpha_x_pos * Dt * Dt + v_x_pos * Dt
            y_prime = y_pos + 0.5 * alpha_y_pos * Dt * Dt + v_y_pos * Dt
            v_x_1_prime = v_x_pos + alpha_x_pos * Dt
            v_y_1_prime = v_y_pos + alpha_y_pos * Dt
            x_prime_vec.append(x_prime)
            y_prime_vec.append(y_prime)

            x_pred = self.a[i, 0] + 0.5 * alpha_x_pos * Dt * Dt + v_x_pos * Dt
            y_pred = self.a[i, 1] + 0.5 * alpha_y_pos * Dt * Dt + v_y_pos * Dt
            x_pred_vec.append(x_pred)
            y_pred_vec.append(y_pred)

            # reset the point to move from to be the next predicted point and its velocity
            x_pos = x_prime
            y_pos = y_prime
            v_x_pos = v_x_1_prime
            v_y_pos = v_y_1_prime
            # reset the alpha by taking another noisy measurement
            alpha_x_i = self.acceleration[i, 0]
            alpha_y_i = self.acceleration[i, 1]
            # add noise to x acceleration
            sd_x_i = abs(alpha_x_i) * co_var
            error_x = np.random.normal(loc=0, scale=sd_x_i, size=1)
            alpha_x_pos = alpha_x_i + error_x
            # add noise to y acceleration
            sd_y_i = abs(alpha_y_i) * co_var
            error_y = np.random.normal(loc=0, scale=sd_y_i, size=1)
            alpha_y_pos = alpha_y_i + error_y

        return x_prime_vec, y_prime_vec


class Clothoid(Path):
    def __init__(self, length=1000, radius_at_end=300, sampling_points=50):
        L_s = length
        R_c = radius_at_end

        scaling_factor_a = 1 / np.sqrt(L_s * R_c)
        t = linspace(0, length, sampling_points)
        t_scaled = t * scaling_factor_a
        x, y = fresnel(t_scaled)
        x_scaled = x / scaling_factor_a
        y_scaled = y / scaling_factor_a

        a = np.column_stack((x_scaled, y_scaled))
        Path.__init__(self, a)


def main():
    a = np.array([[0., 0.], [0.3, 0.], [1.25, -0.1], [2.1, -0.9], [2.85, -2.3], [3.8, -3.95], [5., -5.75], [6.4, -7.8],
                  [8.05, -9.9], [9.9, -11.6], [12.05, -12.85], [14.25, -13.7], [16.5, -13.8], [19.25, -13.35],
                  [21.3, -12.2], [22.8, -10.5], [23.55, -8.15], [22.95, -6.1], [21.35, -3.95], [19.1, -1.9]])

    a_2 = np.array([[0., 0.], [0.10, 0], [0.3, 0.], [0.75, -0.05], [1.25, -0.1], [1.75, -0.5], [2.1, -0.9], [2.4, -1.8],
                    [2.85, -2.3], [3.4, -3.2], [3.8, -3.95], [4.6, -4.8], [5., -5.75], [5.7, -6.8], [6.4, -7.8],
                    [7.2, -8.8], [8.05, -9.9], [9.0, -10.7], [9.9, -11.6], [11.0, -12.2], [12.05, -12.85],
                    [13.0, -13.2], [14.25, -13.7], [15.25, -13.75], [16.5, -13.8], [18.0, -13.5], [19.25, -13.35],
                    [20.25, -12.9], [21.3, -12.2], [22, -11.4], [22.8, -10.5], [23.15, -9], [23.55, -8.15], [23.25, -7],
                    [22.95, -6.1], [22.1, -5], [21.35, -3.95], [20.1, -3], [19.1, -1.9]])
    #plt.scatter(a_2[:, 0], a_2[:, 1])

    path2 = Path(a_2)
    x_prime_vec, y_prime_vec = path2.get_simulated_path_from_acceleration(Dt=0.5, co_var=0.05)

    plt.scatter(a_2[:, 0], a_2[:, 1], color='g')
    plt.scatter(x_prime_vec, y_prime_vec, color='r')
    # plt.scatter(x_pred_vec, y_pred_vec, color='b')
    plt.show()

if __name__ == '__main__':
    main()
