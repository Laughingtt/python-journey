cdef list prime(int nb_primes):
    cdef list p = []
    cdef int n = 2
    while len(p) < nb_primes:
        # Is n prime?
        for i in p:
            if n % i == 0:
                break

        # If no break occurred in the loop
        else:
            p.append(n)
        n += 1
    return p

def prime2(int nb_primes):
    cdef list p = []
    cdef int n = 2
    while len(p) < nb_primes:
        # Is n prime?
        for i in p:
            if n % i == 0:
                break

        # If no break occurred in the loop
        else:
            p.append(n)
        n += 1
    return p

def prime3(int nb_primes):
    cdef list p = []
    cdef int n = 2
    while len(p) < nb_primes:
        # Is n prime?
        for i in p:
            if n % i == 0:
                break

        # If no break occurred in the loop
        else:
            p.append(n)
        n += 1
    return p

cdef class Animal():
    def __init__(self, w, h):
        self.width = w
        self.height = h
