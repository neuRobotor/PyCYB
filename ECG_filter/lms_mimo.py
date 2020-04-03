import numpy as np


def lms_stack(x, order):
    x = np.squeeze(x)
    X = np.zeros((order, len(x)-order+1))
    for i in range(order):
        X[i] = x[order-i-1:len(x)-i]
    return X


class LMS:
    def __init__(self, x_in, d_in, mu, gamma=0., def_weigths=0.):
        """
        :type d_in: np.ndarray
        :type x_in: np.ndarray
        :type mu: Union[ndarray, float, int]
        :type gamma: Union[ndarray, float, int]
        :type def_weigths: Union[ndarray, float, int]
        """
        self.X = np.array(x_in, ndmin=2)
        self.D = np.array(d_in, ndmin=2)
        self.mu = np.zeros(self.D.shape[0])
        self.mu[:] = mu
        self.gamma = gamma
        self.W = np.zeros((self.X.shape[0], self.D.shape[0], self.D.shape[1] + 1))
        self.W[:, :, 0] = def_weigths
        self.e = np.zeros_like(self.D)
        self.Y = np.zeros_like(self.D)
        self.round = 0
        self.e_mul = np.outer if len(self.mu) is not 1 else np.multiply

    def get_range(self, rounds):
        if rounds < 1:
            n_range = range(self.round, self.X.shape[1] + rounds)
        else:
            n_range = range(self.round, min(self.X.shape[1], self.round + rounds))
        self.round = n_range[-1]
        return n_range

    def run(self, rounds=0):
        for n in self.get_range(rounds):
            self.Y[:, n] = self.W[:, :, n].T @ self.X[:, n]
            self.e[:, n] = self.D[:, n] - self.Y[:, n]
            self.W[:, :, n + 1] = (1 - self.mu * self.gamma) * self.W[:, :, n] + \
                                  self.mu * self.e_mul(self.X[:, n], self.e[:, n])


class CLMS(LMS):
    def __init__(self, x_in, d_in, mu, alg='reg', gamma=0, def_weigths=0, def_a_weights=0):
        super(CLMS, self).__init__(x_in=x_in, d_in=d_in, mu=mu, gamma=gamma, def_weigths=def_weigths)
        self.alg = alg
        self.G = np.zeros_like(self.W)
        self.G[:, :, 0] = def_weigths

    def reg(self, rounds=0):
        for n in self.get_range(rounds):
            self.Y[:, n] = np.conj(self.W[:, :, n].T) @ self.X[:, n]
            self.e[:, n] = self.D[:, n] - self.Y[:, n]
            self.W[:, :, n + 1] = (1 - self.mu * self.gamma) * self.W[:, :, n] + \
                                  self.mu * self.e_mul(self.X[:, n], np.conj(self.e[:, n]))

    def aug(self, rounds=0):
        for n in self.get_range(rounds):
            self.Y[:, n] = np.conj(self.W[:, :, n].T) @ self.X[:, n] + \
                           np.conj(self.G[:, :, n].T) @ np.conj(self.X[:, n])
            self.e[:, n] = self.D[:, n] - self.Y[:, n]
            self.W[:, :, n + 1] = (1 - self.mu * self.gamma) * self.W[:, :, n] + \
                                  self.mu * self.e_mul(self.X[:, n], np.conj(self.e[:, n]))
            self.G[:, :, n + 1] = (1 - self.mu * self.gamma) * self.G[:, :, n] + \
                                  self.mu * self.e_mul(np.conj(self.X[:, n]), np.conj(self.e[:, n]))

    def run(self, rounds=0):
        def alg_select(argument):
            switcher = {
                'reg': self.reg,
                'aug': self.aug}
            return switcher.get(argument, lambda: Exception('Not a valid algorithm option!'))

        c_alg = alg_select(self.alg)
        c_alg(rounds)


class GNGD(LMS):
    def __init__(self, x_in, d_in, mu, eps=0., rho=0.15, gamma=0., def_weigths=0, complex=False):
        """
        :type rho: Union[float, int]
        :type eps: Union[np.ndarray, float, int]
        """
        super(GNGD, self).__init__(x_in=x_in, d_in=d_in, mu=mu, gamma=gamma, def_weigths=def_weigths)
        self.rho = rho
        self.eps = np.zeros((self.Y.shape[0], self.W.shape[-1]))
        if isinstance(eps, int) or isinstance(eps, float):
            self.eps[:, 0] = eps
        else:
            self.eps = eps

    def run(self, rounds=0):
        if self.round == 0:
            self.Y[:, 0] = self.W[:, :, 0].T @ self.X[:, 0]
            self.e[:, 0] = self.D[:, 0] - self.Y[:, 0]
            eta = self.mu / (self.X[:, 0] @ self.X[:, 0] + self.eps[:, 0])

            self.W[:, :, 1] = (1 - self.mu * self.gamma) * self.W[:, :, 0] + \
                              eta * self.e_mul(self.X[:, 0], self.e[:, 0])
            self.eps[:, 1] = self.eps[:, 0] - \
                             self.rho * self.mu * self.e[:, 0] * self.e[:, 0] * (self.X[:, 0] @ self.X[:, 0]) / \
                             (self.eps[:, 0] + self.X[:, 0] @ self.X[:, 0].T) ** 2
            self.round = 1
            if rounds:
                rounds -= 1

        for n in self.get_range(rounds):
            self.Y[:, n] = self.W[:, :, n].T @ self.X[:, n]
            self.e[:, n] = self.D[:, n] - self.Y[:, n]
            eta = self.mu / (self.X[:, n] @ self.X[:, n] + self.eps[:, n])
            self.W[:, :, n + 1] = (1 - eta * self.gamma) * self.W[:, :, n] + \
                                  eta * self.e_mul(self.X[:, n], self.e[:, n])
            self.eps[:, n + 1] = self.eps[:, n] - \
                                 self.rho * self.mu * self.e[:, n] * self.e[:, n - 1] * (
                                         self.X[:, n] @ self.X[:, n - 1]) / \
                                 (self.eps[:, n - 1] + self.X[:, n] @ self.X[:, n]) ** 2


class AdLMS(LMS):
    def __init__(self, x_in, d_in, mu, rho=0., psi=0., alph=0., alg='M&X', gamma=0, def_weigths=0):
        """
        :type psi: Union[ndarray, float, int]
        :type rho: Union[float, int]
        :type alph: Union[float, int]
        """
        super(AdLMS, self).__init__(x_in=x_in, d_in=d_in, mu=mu, gamma=gamma, def_weigths=def_weigths)
        self.rho = rho
        self.alg = alg
        self.psi = np.zeros_like(self.W[:, :, 0])
        self.psi[:] = psi
        self.alph = alph

    def run(self, rounds=0):
        def alg_select(argument):
            switcher = {
                'Benv': lambda mu1, x1, psi1, e1: (np.eye(x1.shape[0]) - mu1 * np.outer(x1, x1)) @
                                                  psi1 + self.e_mul(x1, e1),
                'A&F': lambda mu1, x1, psi1, e1: self.alph * psi1 + self.e_mul(x1, e1),
                'M&X': lambda mu1, x1, psi1, e1: self.e_mul(x1, e1)}
            return switcher.get(argument, lambda mu1, x1, psi1, e1: Exception('Not a valid algorithm option!'))

        psi_update = alg_select(self.alg)

        for n in self.get_range(rounds):
            self.Y[:, n] = self.W[:, :, n].T @ self.X[:, n]
            self.e[:, n] = self.D[:, n] - self.Y[:, n]
            self.W[:, :, n + 1] = self.W[:, :, n] + self.mu * self.e_mul(self.X[:, n], self.e[:, n])
            mu_old = self.mu
            self.mu = self.mu + self.rho * self.e[:, n] * (self.X[:, n].T @ self.psi)
            self.psi = psi_update(mu_old, self.X[:, n], self.psi, self.e[:, n])


def main():
    import scipy.signal as sp
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('darkgrid')
    s = np.genfromtxt(r'C:\Users\hbkm9\Documents\Projects\CYB\PyCYB\ECG_filter\noise.csv', delimiter=',')
    #s = sp.lfilter([1, 0.5], [1], n)
    X = np.stack((s[0:-1]), axis=0)
    Y = np.stack((s[1:], 2*s[1:]), axis=0)

    F = AdLMS(X, Y, mu=1e-05, rho=1e-08, alg='Benv')
    F.run()
    plt.plot(F.W[:, 0, :].T)
    plt.show()
    return


if __name__ == "__main__":
    main()
