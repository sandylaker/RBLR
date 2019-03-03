import numpy as np

class Bisquare:
    def __init__(self, clipping=4.685):
        self.clipping = clipping

    def rho(self, array):
        def unit_rho(element):
            element = float(element)
            if abs(element) <= self.clipping:
                return ((self.clipping ** 2.0) / 6.0) * \
                       (1 - (1 - (element / self.clipping) ** 2) ** 3)
            else:
                return (self.clipping ** 2) / 6.0

        vfunc = np.vectorize(unit_rho)
        return vfunc(array)

    def psi(self, array):
        def unit_psi(element):
            element = float(element)
            if abs(element) <= self.clipping:
                return element * ((1 - (element / self.clipping) ** 2) ** 2)
            else:
                return 0.0

        vfunc = np.vectorize(unit_psi)
        return vfunc(array)

    def m_weights(self, array):
        def unit_operation(element):
            if element == 0:
                return 1.0
            else:
                return self.psi(element) / (2 * element)

        vfunc = np.vectorize(unit_operation)
        return vfunc(array)


if __name__ == '__main__':
    pass

