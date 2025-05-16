import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class ReducedOrderModel:
    def __init__(self):
        pass

    def preprocessing(self, split = 0.3):

        forces = np.loadtxt(r"data\forces.txt", delimiter = " ").T

        X = np.loadtxt(r"data\X.txt", delimiter = " ")

        not_nan_list = []

        for i in range(100):
            if np.any(np.isnan(forces[:,i])):
                pass
            else:
                not_nan_list.append(i)

        reduced_forces = forces[:,not_nan_list]

        u, s, v = svd(reduced_forces)

        conv = 0

        i = 0

        while conv < 0.9999**2:

            i += 1

            conv = np.sum(s[0:i]**2) / np.sum(s**2) 

        L = i

        V = u[:,0:L]

        Y_train = V[:,:].T @ reduced_forces[:,:]

        X_train = X[not_nan_list,:]

        X_train, X_test, Y_train, Y_test, uh_train, uh_test = train_test_split(X_train, Y_train.T, reduced_forces.T, test_size = split)

        self.uh_train = uh_train
        self.uh_test = uh_test
        self.X_train = X_train  
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.X_test = X_test
        self.V = V
        self.L = L
        self.reduced_forces = reduced_forces

    def train(self, X_train, Y_train):

        V = rom.V
        L = rom.L

        self.scaler = MinMaxScaler()

        X_scaled = self.scaler.fit_transform(X_train)

        gaussian_processes = []

        for i in range(L):
            kernel = 1.0 * Matern(length_scale=[1.0, 1.0], nu=2.5, length_scale_bounds=(1e-9, 100)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-9, 1000))
            gp = GaussianProcessRegressor(kernel=kernel, alpha = 1e-10, n_restarts_optimizer=20, normalize_y=True)
            gaussian_processes.append(gp)
            gp.fit(X_scaled, Y_train[:,i])

        self.gaussian_processes = gaussian_processes

    def predict(self, nbr_pred_points = 1, X_test = None):
        
        if X_test is not None:

            X_test = self.scaler.fit_transform(X_test)

        else:

            X_test = np.random.uniform(0,1,size=(nbr_pred_points,2))

        L = self.L
        V = self.V

        P_pred_arr = np.zeros((nbr_pred_points,301))
        mean_list = np.zeros((nbr_pred_points, L))
        std_list = np.zeros((nbr_pred_points, L))

        for j in range(nbr_pred_points):
            mean_L = np.zeros((L,))
            std_L = np.zeros((L,))
            for i in range(L):
                gp = self.gaussian_processes[i]
                mean, std = gp.predict(X_test[None, j,:], return_std = True)
                mean_L[i] = mean.item()
                std_L[i] = std.item()

            P_pred = V @ mean_L.T

            mean_list[j,:] = mean_L
            std_list[j,:] = std_L

            P_pred_arr[j,:] = P_pred

        return P_pred_arr, mean_list, std_list
    
    def nu(self, V, std):

        return np.sqrt(np.sum(V**2 @ std.T**2, axis = 0))
    
    def estimate_error(self):

        P_pred_arr, mean_list, std_list = self.predict(nbr_pred_points = self.X_test.shape[0], X_test = self.X_test)

        reduced_uh = self.V @ mean_list.T

        error = np.sum(np.linalg.norm(self.uh_test.T - self.V @ mean_list.T, axis = 0) / (np.linalg.norm(self.uh_test.T, axis = 0))) / self.uh_test.shape[0]

        return error
    
    def estimate_proj_error(self):

        V = self.V

        err = np.sum(np.linalg.norm(self.reduced_forces - V @ V.T @ self.reduced_forces, axis = 0) / (np.linalg.norm(self.reduced_forces, axis = 0))) / self.reduced_forces.shape[0]
        
        return err
    
    def plot_error(self, err_list):

        proj_err = self.estimate_proj_error()

        fig, ax = plt.subplots(1,1, figsize = (5,5), dpi=300)
        ax.plot(np.array(err_list), '.--', label = "")
        ax.plot(np.linspace(proj_err, proj_err, 68))
        ax.set_xlabel("No. of included training points")
        ax.set_ylabel("Relative error")
        ax.legend([r"$\varepsilon_t$", r"$\varepsilon_{\mathbf{V}}$"])
        fig.savefig("plots\\error.pdf")

    def plot_test_curves(self, n = 10):

        disp = np.linspace(0, 3.43, 301)

        P_pred_arr, mean_list, std_list = rom.predict(nbr_pred_points = n, X_test = rom.X_test)

        fig, ax = plt.subplots(1,1, dpi = 300)
        ax.plot(disp, P_pred_arr.T[:,:], '--', label="Reduced order model", color = "tab:red")
        ax.plot(disp, rom.uh_test.T[:,0:n], '-', label="Full order model", color = "tab:blue", alpha = 0.5)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([handles[0], handles[n]], [labels[0], labels[n]])
        ax.set_xlim(0,3.43)
        ax.set_ylim(0,50)
        ax.set_xlabel("Displacement [mm]")
        ax.set_ylabel("Force [N]")
        fig.savefig("plots\\test_curves.pdf")
    

    def plot_active_selection(self, X_current):

        fig, ax = plt.subplots(1,1, figsize = (5,5), dpi=300)
        ax.plot(X_current[:,0], X_current[:,1], 'k.', label = "Data pool", clip_on = False)
        ax.plot(rom.X_test[:,0], rom.X_test[:,1], 'k.', label=None, clip_on = False)
        ax.plot(X_current[0:20,0], X_current[0:20,1], 'bo', markersize = 6, markerfacecolor="None", label="first 20", clip_on = False)
        ax.plot(X_current[20:40,0], X_current[20:40,1], 'ro', markersize = 10, markerfacecolor="None", label="21-40", clip_on = False)
        ax.plot(X_current[40:60,0], X_current[40:60,1], 'go', markersize = 14, markerfacecolor="None", label="41-60", clip_on = False)
        ax.set_xlim(1.5,4.5)
        ax.set_ylim(1.5,4.5)
        ax.set_ylabel(r"$G_f$")
        ax.set_xlabel(r"$f_t$")
        ax.legend()
        fig.savefig("plots\\active_selection.pdf")

    def plot_components(self, X_current):

        X_scaled = rom.scaler.fit_transform(X_current)

        for i in range(rom.L):
            fig, axs = plt.subplots(1, 1, subplot_kw={'projection': '3d'}, dpi = 250)
            axs.scatter(X_current[:,0], X_current[:,1], Y_current[:,i], linewidth = 1e-5, color = "tab:red")
            mu, sigma = rom.gaussian_processes[i].predict(X_scaled, return_std = True)
            axs.plot_trisurf(X_current[:,0], X_current[:,1], mu, linewidth = 0, alpha = 0.4)
            axs.plot_trisurf(X_current[:,0], X_current[:,1], mu - sigma, linewidth = 0, color = "tab:blue", alpha = 0.1)
            axs.plot_trisurf(X_current[:,0], X_current[:,1], mu + sigma, linewidth = 0, color = "tab:blue", alpha = 0.1)
            axs.set_xlabel(r"$\mu_1$")
            axs.set_ylabel(r"$\mu_2$")
            axs.set_zlabel(r"$ \mathbf{V}^T \mathbf{u} $")
            fig.savefig("plots\\entry%s.pdf"%i)
    

if __name__ == "__main__":
    
    rom = ReducedOrderModel()
    rom.preprocessing(split = 0.3)
    proj_err = rom.estimate_proj_error()

    X_current = rom.X_train[0:2,:]
    Y_current = rom.Y_train[0:2,:]

    rom.X_train = np.delete(rom.X_train, 0, axis = 0)
    rom.X_train = np.delete(rom.X_train, 0, axis = 0)

    rom.Y_train = np.delete(rom.Y_train, 0, axis = 0)
    rom.Y_train = np.delete(rom.Y_train, 0, axis = 0)

    err_list = []

    for i in range(67):

        rom.train(X_current, Y_current)

        P_pred_arr, mean_list, std_list = rom.predict(nbr_pred_points = rom.X_train.shape[0], X_test = rom.X_train)

        err = rom.nu(rom.V, std_list)

        a = np.argmax(err)

        X_current = np.append(X_current, rom.X_train[None, a,:], axis = 0)
        Y_current = np.append(Y_current, rom.Y_train[None, a,:], axis = 0)

        rom.X_train = np.delete(rom.X_train, a, axis = 0)
        rom.Y_train = np.delete(rom.Y_train, a, axis = 0)

        error = rom.estimate_error()

        err_list.append(error)

    rom.plot_error(err_list)
    rom.plot_active_selection(X_current)
    rom.plot_components(X_current)
    rom.plot_test_curves(n = 30)

    #########################
    ## Test on unseen data ##
    #########################

    n = 30
    X_new = np.loadtxt("data\\X_new.txt")
    X_new = X_new[0:n,:]

    P_pred_arr, mean_list, std_list = rom.predict(nbr_pred_points = n, X_test = X_new)

    forces = np.loadtxt(r"data\\forces_new.txt")
    forces = forces[0:n,:]

    fig, ax = plt.subplots(1,1, dpi = 300)

    ax.plot(np.linspace(0,3.43,301), P_pred_arr.T[:,:], '--', label = "Reduced order model", color = "tab:red", alpha = 1)
    ax.plot(np.linspace(0,3.43,301), forces.T[:,:], '-', label = "Full order model", color = "tab:blue", alpha = 0.5)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[0], handles[n]], [labels[0], labels[n]])
    ax.set_xlim(0,3.43)
    ax.set_ylim(0,50)
    ax.set_xlabel("Displacement [mm]")
    ax.set_ylabel("Force [N]")
    fig.savefig(r"plots\\overfitting_test.pdf")
    error = np.sum(np.linalg.norm(forces.T - rom.V @ mean_list.T, axis = 0) / (np.linalg.norm(forces.T, axis = 0))) / rom.uh_test.shape[0]








