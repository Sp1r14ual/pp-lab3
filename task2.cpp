#include <omp.h>
#include <iostream>
#include <math.h>

using namespace std;

int f(int &b)
{

    int result = 0;
    for (int i = 0; i < 10000000; ++i)
    {
        result += log(b) + log(i);
    }
    return result;
}

int main()
{

    int a[100], b[100];
    // Инициализация массива b
    for (int i = 0; i < 100; ++i)
        b[i] = i;
    // Директива OpenMP для распараллеливания цикла

#pragma omp parallel for
    for (int i = 0; i < 100; ++i)
    {
        a[i] = f(b[i]);
        b[i] = 2 * a[i];
    }

    int result = 0;
// Далее значения a[i] и b[i] используются, например, так:
#pragma omp parallel for reduction(+ : result)
    for (int i = 0; i < 100; ++i)
        result += (a[i] + b[i]);

    cout << "Result = " << result << endl;

    return 0;
}