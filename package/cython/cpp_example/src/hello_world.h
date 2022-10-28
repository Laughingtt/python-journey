#ifndef HELLO_H
#define HELLO_H
#include "vector"

using namespace std;

namespace test {
    class Hello {
        public:
           Hello();
           vector<int> primesc(unsigned int nb_primes);
    };
}

#endif