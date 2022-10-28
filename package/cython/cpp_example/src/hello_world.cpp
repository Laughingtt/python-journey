#include <iostream>
#include "hello_world.h"

namespace test{
Hello::Hello () {}

vector<int> Hello::primesc(unsigned int nb_primes)
{
    int n;
    vector<int> p;
    p.reserve(nb_primes);
    n = 2;

    while (p.size() < nb_primes)
    {
        bool s = false;
        for (size_t i = 0; i < p.size(); i++)
        {
            if (n % p[i] == 0)
            {
                s = true;
                break;
            }
        }
        if (!s)
        {
            p.push_back(n);
        }
        n++;
    }
    return p;
}
}
