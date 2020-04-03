import numpy as np


def lms_stack(x, order):
    x = np.squeeze(x)
    X = np.zeros((order, len(x) - order + 1))
    for i in range(order):
        X[i] = x[order - i - 1:len(x) - i]
    return X


class LMS:
    def __init__(self, x_in, d_in, mu, gamma=0., def_weigths=0., sign=False, act='lin'):
        """
        :type d_in: np.ndarray
        :type x_in: np.ndarray
        :type mu: Union[ndarray, float, int]
        :type gamma: Union[ndarray, float, int]
        :type def_weigths: Union[ndarray, float, int]
        """
        self.X = np.array(x_in, ndmin=2)
        self.D = np.array(d_in)
        if self.D.ndim > 1:
            Exception("Multidimensional target signal supplied. Use MIMO algorithm instead.")
        self.mu = mu
        self.gamma = gamma
        self.W = np.zeros((self.X.shape[0], len(self.D) + 1))
        self.W[:, 0] = def_weigths
        self.e = np.zeros_like(self.D)
        self.Y = np.zeros_like(self.D)
        self.round = 0
        self.sign = sign

        if sign:
            self.weight_update = self.weight_update_sign
            self.run = self.run_symbolic
        if act is 'tanh':
            self.act = np.tanh
            self.d_act = lambda x: 1-np.tanh(x)**2
            self.run = self.run_symbolic
        elif act is 'lin':
            self.act = lambda x: x
            self.d_act = lambda x: 1

    def get_range(self, rounds):
        if rounds < 1:
            n_range = range(self.round, self.X.shape[1] + rounds)
        else:
            n_range = range(self.round, min(self.X.shape[1], self.round + rounds))
        self.round = n_range[-1]
        return n_range

    def run(self, rounds=0):
        for n in self.get_range(rounds):
            self.Y[n] = self.W[:, n] @ self.X[:, n]
            self.e[n] = self.D[n] - self.Y[n]
            self.W[:, n + 1] = (1 - self.mu * self.gamma) * self.W[:, n] + \
                               self.mu * self.X[:, n] * self.e[n]

    def run_symbolic(self, rounds=0):
        for n in self.get_range(rounds):
            self.predict(n)
            self.error(n)
            self.weight_update(n, self.mu)

    def predict(self, round_in):
        self.Y[round_in] = self.act(self.W[:, round_in] @ self.X[:, round_in])

    def error(self, round_in):
        self.e[round_in] = self.D[round_in] - self.Y[round_in]

    def weight_update(self, round_in, stepsize):
        self.W[:, round_in+1] = (1 - stepsize * self.gamma) * self.W[:, round_in] + \
                                stepsize * self.d_act(self.W[:, round_in] @ self.X[:, round_in]) * \
                                self.X[:, round_in] * self.e[round_in]

    def weight_update_sign(self, round_in, stepsize):
        self.W[:, round_in + 1] = (1 - stepsize * self.gamma) * self.W[:, round_in] + \
                                  stepsize * self.d_act(self.W[:, round_in] @ self.X[:, round_in]) * \
                                  np.sign(self.X[:, round_in]) * np.sign(self.e[round_in])

class CLMS(LMS):
    def __init__(self, x_in, d_in, mu, alg='reg', def_a_weights=0, **kwargs):
        super(CLMS, self).__init__(x_in=x_in, d_in=d_in, mu=mu, **kwargs)
        self.Y = np.array(self.Y, dtype=complex)
        self.W = np.array(self.W, dtype=complex)
        self.e = np.array(self.e, dtype=complex)
        self.alg = alg
        self.G = np.zeros_like(self.W)
        self.G[:, 0] = def_a_weights

    def reg(self, rounds=0):
        for n in self.get_range(rounds):
            self.Y[n] = np.conj(self.W[:, n]) @ self.X[:, n]
            self.e[n] = self.D[n] - self.Y[n]
            self.W[:, n + 1] = (1 - self.mu * self.gamma) * self.W[:, n] + \
                               self.mu * self.X[:, n] * np.conj(self.e[n])

    def aug(self, rounds=0):
        for n in self.get_range(rounds):
            self.Y[n] = np.conj(self.W[:, n]) @ self.X[:, n] + \
                        np.conj(self.G[:, n]) @ np.conj(self.X[:, n])
            self.e[n] = self.D[n] - self.Y[n]
            self.W[:, n + 1] = (1 - self.mu * self.gamma) * self.W[:, n] + \
                               self.mu * self.X[:, n] * np.conj(self.e[n])
            self.G[:, n + 1] = (1 - self.mu * self.gamma) * self.G[:, n] + \
                               self.mu * np.conj(self.X[:, n]) * np.conj(self.e[n])

    def run(self, rounds=0):
        def alg_select(argument):
            switcher = {
                'reg': self.reg,
                'aug': self.aug}
            return switcher.get(argument, lambda: Exception('Not a valid algorithm option!'))

        c_alg = alg_select(self.alg)
        c_alg(rounds)


class GNGD(LMS):
    def __init__(self, x_in, d_in, mu, eps=0., rho=0.15, **kwargs):
        """
        :type rho: Union[float, int]
        :type eps: Union[np.ndarray, float, int]
        """
        super(GNGD, self).__init__(x_in=x_in, d_in=d_in, mu=mu, **kwargs)
        self.rho = rho
        self.eps = np.zeros(self.W.shape[-1])
        self.eps[0] = eps

    def run(self, rounds=0):
        if self.round == 0:
            self.Y[0] = self.W[:, 0] @ self.X[:, 0]
            self.e[0] = self.D[0] - self.Y[0]
            eta = self.mu / (self.X[:, 0] @ self.X[:, 0] + self.eps[0])

            self.W[:, 1] = (1 - eta * self.gamma) * self.W[:, 0] + \
                           eta * self.X[:, 0] * self.e[0]
            self.eps[1] = self.eps[0] - \
                             self.rho * self.mu * self.e[0] * self.e[0] * (self.X[:, 0] @ self.X[:, 0]) / \
                             (self.X[:, 0] @ self.X[:, 0] + self.eps[0]) ** 2
            self.round = 1
            if rounds:
                rounds -= 1

        for n in self.get_range(rounds):
            self.Y[n] = self.W[:, n] @ self.X[:, n]
            self.e[n] = self.D[n] - self.Y[n]
            eta = self.mu / (self.X[:, n] @ self.X[:, n] + self.eps[n])
            self.W[:, n + 1] = (1 - eta * self.gamma) * self.W[:, n] + eta * self.X[:, n] * self.e[n]
            self.eps[n + 1] = self.eps[n] - \
                                 self.rho * self.mu * self.e[n] * self.e[n - 1] * (self.X[:, n] @ self.X[:, n - 1]) / \
                                 (self.X[:, n] @ self.X[:, n] + self.eps[n - 1]) ** 2

    def run_symbolic(self, rounds=0):
        if self.round == 0:
            self.predict(0)
            self.error(0)
            eta = self.mu / (self.X[:, 0] @ self.X[:, 0] + self.eps[0])
            self.weight_update(0, eta)
            self.eps[1] = self.eps[0] - \
                             self.rho * self.mu * self.e[0] * self.e[0] * np.sign(self.X[:, 0] @ self.X[:, 0]) / \
                          (self.X[:, 0] @ self.X[:, 0] + self.eps[0]) ** 2
            self.round = 1
            if rounds:
                rounds -= 1

        for n in self.get_range(rounds):
            self.predict(n)
            self.error(n)
            eta = self.mu / (self.X[:, n] @ self.X[:, n] + self.eps[n])
            self.weight_update(n, eta)
            self.eps[n + 1] = self.eps[n] - \
                                 self.rho * self.mu * self.e[n] * self.e[n - 1] * np.sign(self.X[:, n] @ self.X[:, n - 1]) / \
                                 (self.X[:, n] @ self.X[:, n] + self.eps[n - 1]) ** 2


class GASS(LMS):
    def __init__(self, x_in, d_in, mu, rho=0., psi=0., alph=0.99, alg='M&X', **kwargs):
        """
        :type psi: Union[ndarray, float, int]
        :type rho: Union[float, int]
        :type alph: Union[float, int]
        """
        super(GASS, self).__init__(x_in=x_in, d_in=d_in, mu=mu, **kwargs)
        self.rho = rho
        self.alg = alg
        self.psi = np.zeros_like(self.W[:, 0])
        self.psi[:] = psi
        self.alph = alph

        def alg_select(argument):
            switcher = {
                'Benv': lambda mu1, x1, psi1, e1: (np.eye(x1.shape[0]) - mu1 * np.outer(x1, x1)) @ psi1 + x1 * e1,
                'A&F': lambda mu1, x1, psi1, e1: self.alph * psi1 + x1 * e1,
                'M&X': lambda mu1, x1, psi1, e1: x1 * e1}
            return switcher.get(argument, lambda mu1, x1, psi1, e1: Exception('Not a valid algorithm option!'))

        self.psi_update = alg_select(self.alg)
        if self.sign:
            def psi_update_sym(mu1, x1, psi1, e1):
                return self.psi_update(mu1, np.sign(x1), psi1, np.sign(e1))
            self.psi_update_sym = psi_update_sym
        else:
            self.psi_update_sym = self.psi_update


    def run(self, rounds=0):
        for n in self.get_range(rounds):
            self.Y[n] = self.W[:, n] @ self.X[:, n]
            self.e[n] = self.D[n] - self.Y[n]
            self.W[:, n + 1] = self.W[:, n] + self.mu * np.sign(self.X[:, n]) * self.e[n]
            mu_old = self.mu
            self.mu = self.mu + self.rho * self.e[n] * self.X[:, n] @ self.psi
            self.psi = self.psi_update(mu_old, np.sign(self.X[:, n]), self.psi, np.sign(self.e[n]))

    def run_symbolic(self, rounds=0):
        for n in self.get_range(rounds):
            self.predict(n)
            self.error(n)
            self.weight_update(n, self.mu)
            mu_old = self.mu
            self.mu = self.mu + self.rho * self.e[n] * self.X[:, n] @ self.psi
            self.psi = self.psi_update_sym(mu_old, self.X[:, n], self.psi, self.e[n])



def main():
    import scipy.signal as sp
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('darkgrid')
    s = np.genfromtxt(r'C:\Users\hbkm9\Documents\Projects\CYB\PyCYB\ECG_filter\noise.csv', delimiter=',')
    # s = sp.lfilter([1, 0.5], [1], n)
    # s[5000] = 1000
    s = 3*s
    X = np.stack((s[0:-1]), axis=0)
    Y = np.stack(np.tanh((s[1:])), axis=0)

    F = GASS(X, Y, mu=0.001, alg='M&X', rho=0,  act='tanh')
    F.run()
    plt.plot(F.e)
    plt.xlabel('Sample n')
    plt.ylabel('Weight estimate')
    plt.hlines([0.4], 0, 10000, colors='r', ls='--')
    plt.title("GASS sign sign algorithm + outlier")
    plt.show()

    return


if __name__ == "__main__":
    main()
