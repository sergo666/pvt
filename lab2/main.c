#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <sys/time.h>
#include <inttypes.h>
#include <xmalloc/xmalloc.h>

double wtime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

void matrix_vector_product_omp(const double* a, const double* b, double* c, const uint64_t** m, const uint64_t** n) {
#pragma omp parallel for
    for (size_t i = 0; i < **m; ++i) {
        *(c + i) = 0.;
        for (size_t j = 0; j < **n; ++j) {
            *(c + i) += *(a + i * **n + j) * *(b + j);
        }
    }
}

void run_parallel(const uint64_t* m, const uint64_t* n) {
    double* a = xmalloc(sizeof(*a) * *n * *m);
    double* b = xmalloc(sizeof(*b) * *n);
    double* c = xmalloc(sizeof(*c) * *m);
    double t = wtime();
#pragma omp parallel for
    for (size_t i = 0; i < *m; ++i) {
        for (size_t j = 0; j < *n; ++j) {
            *(a + i * *n + j)= i + j;
        }
        *(c + i) = 0.;
    }
    for (size_t j = 0; j < *n; ++j) {
        *(b + j) = j;
    }
    //double t = wtime();
    matrix_vector_product_omp(a, b, c, &m, &n);
    t = wtime() - t;
    printf("Elapsed time (parallel): %.7f sec.\n", t);
    xfree(a);
    xfree(b);
    xfree(c);
}
/*void run_parallel(const uint64_t* m, const uint64_t* n) {
    double* a = xmalloc(sizeof(*a) * *n * *m);
    double* b = xmalloc(sizeof(*b) * *n);
    double* c = xmalloc(sizeof(*c) * *m);

    for (size_t i = 0; i < *m; ++i) {
        for (size_t j = 0; j < *n; ++j) {
            *(a + i * *n + j)= i + j;
        }
    }
    for (size_t j = 0; j < *n; ++j) {
        *(b + j) = j;
    }

    double t = wtime();
    matrix_vector_product_omp(a, b, c, &m, &n);
    t = wtime() - t;
    printf("Elapsed time (parallel): %.7f sec.\n", t);
    xfree(a);
    xfree(b);
    xfree(c);
}*/

void matrix_vector_product(const double* a, const double* b, double* c, const uint64_t** m, const uint64_t** n) {
    for (size_t i = 0; i < **m; ++i) {
        *(c + i) = 0.;
        for (size_t j = 0; j < **n; ++j) {
            *(c + i) = *(c + i) + *(a + i * **n + j) * *(b + j);
        }
    }
}

void run_serial(const uint64_t* m, const uint64_t* n) {
    double* a = xmalloc(sizeof(*a) * *n * *m);
    double* b = xmalloc(sizeof(*b) * *n);
    double* c = xmalloc(sizeof(*c) * *m);
    double t = wtime();
    for (size_t i = 0; i < *m; ++i) {
        for (size_t j = 0; j < *n; ++j) {
            *(a + i * *n + j)= i + j;
        }
    }
    for (size_t j = 0; j < *n; ++j) {
        *(b + j) = j;
    }

    //double t = wtime();
    matrix_vector_product(a, b, c, &m, &n);
    t = wtime() - t;
    printf("Elapsed time (serial): %.7f sec.\n", t);
    xfree(a);
    xfree(b);
    xfree(c);
}

int main(uint32_t argc, uint8_t **argv) {
    if (argc < 1) {
        perror("No arguments!");
    }

    uint8_t* temp;
    uint64_t m = strtol(argv[1], &temp, 10);
    uint64_t n = strtol(argv[2], &temp, 10);
    printf("MATRIX = %"PRId64"\n", m * n);
    printf("Matrix-vector product (c[m] = a[m, n] * b[n]; m = %"PRId64", n = %"PRId64")\n", m, n);
    printf("Memory used: %" PRIu64 " MiB\n", ((m * n + m + n) * sizeof(double)) >> 20);
    run_serial(&m, &n);
    run_parallel(&m, &n);
    return 0;
}
