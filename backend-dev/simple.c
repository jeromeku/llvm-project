// foo.c
int foo(int *a, int n) {
    int sum = 0;
    for (int i = 0; i < n; ++i)
        sum += a[i] * 3;
    return sum;
}
