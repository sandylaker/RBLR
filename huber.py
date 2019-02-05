import numpy as np

class Huber:

    def __init__(self, clipping=1.345):
        self.clipping = clipping

    def rho(self, array):
        def unit_rho(element):
            element = float(element)
            if abs(element) <= self.clipping:
                return element ** 2 / 2.0
            else:
                return self.clipping * (abs(element) - self.clipping / 2.0)

        vfunc = np.vectorize(unit_rho)
        return vfunc(array)

    def psi(self, array):
        def unit_psi(element):
            element = float(element)
            if abs(element) >= self.clipping:
                return self.clipping * np.sign(element)
            else:
                return element

        vfunc = np.vectorize(unit_psi)
        return vfunc(array)

    def m_weights(self, matrix):
        def unit_operation(element):
            if element == 0:
                return 1.0
            else:
                return self.psi(element) / (2 * element)
        vfunc = np.vectorize(unit_operation)
        return vfunc(matrix)


if __name__ == '__main__':
    h = Huber()
    print('weight: \n', h.m_weights(np.random.randn(3,4)*5))