#include <cuda_runtime.h>
#include <math.h>

#define MAX_BLOCK_SIZE 1024

__global__ void quasidirichletfourier_is_prime(unsigned long long n, int *result) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 1 && idx <= n/2) {
        if (n % idx == 0) {
            result[0] = 0;
        }
    }
}

__global__ void quasidirichletfourier_gaussian_blur(float *data, float *result, int width, int height, float sigma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum = 0.0f;
        for (int i = -2; i <= 2; i++) {
            for (int j = -2; j <= 2; j++) {
                int nx = x + i;
                int ny = y + j;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    float dx = (nx - x);
                    float dy = (ny - y);
                    sum += data[ny * width + nx] * expf(-(dx*dx + dy*dy) / (2.0f * sigma * sigma));
                }
            }
        }
        result[y * width + x] = sum;
    }
}

__global__ void quasidirichletfourier_fft(float *data, int n, int sign) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float re = 0.0f, im = 0.0f;
        for (int k = 0; k < n; k++) {
            float angle = 2.0f * M_PI * idx * k / n;
            re += data[k] * cos(angle);
            im -= sign * data[k] * sin(angle);
        }
        data[idx] = re / n;
    }
}

__global__ void quasidirichletfourier_quadratic_sieve(unsigned long long n, int *sieve) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 1 && idx <= sqrt(n)) {
        if (n % idx == 0) {
            sieve[idx] = 1;
            sieve[n / idx] = 1;
        }
    }
}

__global__ void quasidirichletfourier_rabin_miller(unsigned long long n, int *result) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 4 && n > 2) {
        unsigned long long a = 2 + idx;
        unsigned long long d = n - 1;
        int s = 0;

        while (d % 2 == 0) {
            d /= 2;
            s++;
        }

        unsigned long long x = pow(a, d, n);
        if (x == 1 || x == n - 1) {
            result[0] = 1;
        } else {
            for (int r = 1; r < s; r++) {
                x = pow(x, 2, n);
                if (x == 1) {
                    result[0] = 0;
                    break;
                }
                if (x == n - 1) {
                    result[0] = 1;
                    break;
                }
            }
        }
    }
}

__global__ void quasidirichletfourier_pollards_rho(unsigned long long n, unsigned long long *factor) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && n > 1) {
        unsigned long long x = idx;
        unsigned long long y = idx;
        unsigned long long d = 1;

        while (d == 1) {
            x = (x * x + 1) % n;
            y = (y * y + 1) % n;
            y = (y * y + 1) % n;
            d = gcd(abs(x - y), n);
        }

        factor[0] = d;
    }
}

__global__ void quasidirichletfourier_sieve_of_eratosthenes(unsigned long long limit, int *sieve) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 1 && idx <= sqrt(limit)) {
        if (sieve[idx] == 0) {
            for (unsigned long long j = idx*idx; j <= limit; j += idx) {
                sieve[j] = 1;
            }
        }
    }
}

__global__ void quasidirichletfourier_totient(unsigned long long n, int *result) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && n > 1) {
        result[0] *= (idx == 1 || gcd(n, idx) == 1);
    }
}

__global__ void quasidirichletfourier_legendre_symbol(unsigned long long a, unsigned long long p, int *result) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < p && p > 2) {
        result[0] *= (modexp(a, (p-1)/2, p) == 1 ? 1 : -1);
    }
}

__global__ void quasidirichletfourier_elliptic_curve_point_add(unsigned long long x1, unsigned long long y1, unsigned long long x2, unsigned long long y2, unsigned long long a, unsigned long long p) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        unsigned long long m, x3, y3;

        if (x1 == x2 && y1 == y2) {
            m = ((3*x1*x1 + a) % p) * modinv(2*y1, p);
        } else {
            m = (y2 - y1) * modinv(x2 - x1, p);
        }

        x3 = (m*m - x1 - x2) % p;
        y3 = (m*(x1 - x3) - y1) % p;

        result[0] = x3;
        result[1] = y3;
    }
}

__global__ void quasidirichletfourier_dijkstra(unsigned long long *graph, int num_vertices, unsigned long long source, unsigned long long *distances) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        for (int i = 0; i < num_vertices; i++) {
            distances[i] = INT_MAX;
        }
        distances[source] = 0;

        for (int k = 0; k < num_vertices - 1; k++) {
            unsigned long long min_distance = INT_MAX, u = 0;

            for (int v = 0; v < num_vertices; v++) {
                if (distances[v] < min_distance && !visited[v]) {
                    min_distance = distances[v];
                    u = v;
                }
            }

            visited[u] = 1;

            for (int v = 0; v < num_vertices; v++) {
                if (!visited[v] && graph[u * num_vertices + v] && distances[u] != INT_MAX &&
                    distances[u] + graph[u * num_vertices + v] < distances[v]) {
                    distances[v] = distances[u] + graph[u * num_vertices + v];
                }
            }
        }
    }
}

__global__ void quasidirichletfourier_kruskal(unsigned long long *graph, int num_vertices, unsigned long long *mst) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_edges) {
        edges[idx] = make_pair(graph[idx], idx);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        sort(edges, edges + num_edges);

        for (int i = 0; i < num_vertices; i++) {
            parent[i] = i;
        }

        for (int i = 0; i < num_edges; i++) {
            int u = find_parent(edges[i].second / num_vertices);
            int v = find_parent(edges[i].second % num_vertices);

            if (u != v) {
                mst[i] = edges[i].first;
                union_set(u, v);
            }
        }
    }
}

__global__ void quasidirichletfourier_floyd_warshall(unsigned long long *graph, int num_vertices) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vertices * num_vertices) {
        unsigned long long i = idx / num_vertices;
        unsigned long long j = idx % num_vertices;

        for (int k = 0; k < num_vertices; k++) {
            graph[i * num_vertices + j] = min(graph[i * num_vertices + j], graph[i * num_vertices + k] + graph[k * num_vertices + j]);
        }
    }
}

__global__ void quasidirichletfourier_bellman_ford(unsigned long long *graph, int num_vertices, unsigned long long source, unsigned long long *distances) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        for (int i = 0; i < num_vertices; i++) {
            distances[i] = INT_MAX;
        }
        distances[source] = 0;

        for (int k = 0; k < num_vertices - 1; k++) {
            for (int u = 0; u < num_vertices; u++) {
                for (int v = 0; v < num_vertices; v++) {
                    if (graph[u * num_vertices + v] && distances[u] != INT_MAX &&
                        distances[u] + graph[u * num_vertices + v] < distances[v]) {
                        distances[v] = distances[u] + graph[u * num_vertices + v];
                    }
                }
            }
        }

        for (int u = 0; u < num_vertices; u++) {
            for (int v = 0; v < num_vertices; v++) {
                if (graph[u * num_vertices + v] && distances[u] != INT_MAX &&
                    distances[u] + graph[u * num_vertices + v] < distances[v]) {
                    // Negative weight cycle detected
                }
            }
        }
    }
}

__global__ void quasidirichletfourier_spfa(unsigned long long *graph, int num_vertices, unsigned long long source, unsigned long long *distances) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        for (int i = 0; i < num_vertices; i++) {
            distances[i] = INT_MAX;
        }
        distances[source] = 0;

        queue<int> q;
        q.push(source);
        vector<bool> in_queue(num_vertices, false);
        in_queue[source] = true;

        while (!q.empty()) {
            int u = q.front();
            q.pop();
            in_queue[u] = false;

            for (int v = 0; v < num_vertices; v++) {
                if (graph[u * num_vertices + v] && distances[u] != INT_MAX &&
                    distances[u] + graph[u * num_vertices + v] < distances[v]) {
                    distances[v] = distances[u] + graph[u * num_vertices + v];

                    if (!in_queue[v]) {
                        q.push(v);
                        in_queue[v] = true;
                    }
                }
            }
        }
    }
}

__global__ void quasidirichletfourier_dijkstra_heap(unsigned long long *graph, int num_vertices, unsigned long long source, unsigned long long *distances) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        priority_queue<pair<unsigned long long, int>, vector<pair<unsigned long long, int>>, greater<pair<unsigned long long, int>>> pq;

        for (int i = 0; i < num_vertices; i++) {
            distances[i] = INT_MAX;
        }
        distances[source] = 0;
        pq.push(make_pair(0, source));

        while (!pq.empty()) {
            unsigned long long dist_u = pq.top().first;
            int u = pq.top().second;
            pq.pop();

            if (dist_u > distances[u]) continue;

            for (int v = 0; v < num_vertices; v++) {
                if (graph[u * num_vertices + v] && distances[u] != INT_MAX &&
                    distances[u] + graph[u * num_vertices + v] < distances[v]) {
                    distances[v] = distances[u] + graph[u * num_vertices + v];
                    pq.push(make_pair(distances[v], v));
                }
            }
        }
    }
}

__global__ void quasidirichletfourier_floyd_warshall_parallel(unsigned long long *graph, int num_vertices) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vertices * num_vertices) {
        unsigned long long i = idx / num_vertices;
        unsigned long long j = idx % num_vertices;

        for (int k = 0; k < num_vertices; k++) {
            graph[i * num_vertices + j] = min(graph[i * num_vertices + j], graph[i * num_vertices + k] + graph[k * num_vertices + j]);
        }
    }
}

__global__ void quasidirichletfourier_spfa_parallel(unsigned long long *graph, int num_vertices, unsigned long long source, unsigned long long *distances) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        for (int i = 0; i < num_vertices; i++) {
            distances[i] = INT_MAX;
        }
        distances[source] = 0;

        queue<int> q;
        q.push(source);
        vector<bool> in_queue(num_vertices, false);
        in_queue[source] = true;

        while (!q.empty()) {
            int u = q.front();
            q.pop();
            in_queue[u] = false;

            for (int v = 0; v < num_vertices; v++) {
                if (graph[u * num_vertices + v] && distances[u] != INT_MAX &&
                    distances[u] + graph[u * num_vertices + v] < distances[v]) {
                    distances[v] = distances[u] + graph[u * num_vertices + v];

                    if (!in_queue[v]) {
                        q.push(v);
                        in_queue[v] = true;
                    }
                }
            }
        }
    }
}

__global__ void quasidirichletfourier_bellman_ford_parallel(unsigned long long *graph, int num_vertices, unsigned long long source, unsigned long long *distances) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        for (int i = 0; i < num_vertices; i++) {
            distances[i] = INT_MAX;
        }
        distances[source] = 0;

        for (int k = 0; k < num_vertices - 1; k++) {
            for (int u = 0; u < num_vertices; u++) {
                for (int v = 0; v < num_vertices; v++) {
                    if (graph[u * num_vertices + v] && distances[u] != INT_MAX &&
                        distances[u] + graph[u * num_vertices + v] < distances[v]) {
                        distances[v] = distances[u] + graph[u * num_vertices + v];
                    }
                }
            }
        }

        for (int u = 0; u < num_vertices; u++) {
            for (int v = 0; v < num_vertices; v++) {
                if (graph[u * num_vertices + v] && distances[u] != INT_MAX &&
                    distances[u] + graph[u * num_vertices + v] < distances[v]) {
                    // Negative weight cycle detected
                }
            }
        }
    }
}

__global__ void quasidirichletfourier_dijkstra_heap_parallel(unsigned long long *graph, int num_vertices, unsigned long long source, unsigned long long *distances) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        priority_queue<pair<unsigned long long, int>, vector<pair<unsigned long long, int>>, greater<pair<unsigned long long, int>>> pq;

        for (int i = 0; i < num_vertices; i++) {
            distances[i] = INT_MAX;
        }
        distances[source] = 0;
        pq.push(make_pair(0, source));

        while (!pq.empty()) {
            unsigned long long dist_u = pq.top().first;
            int u = pq.top().second;
            pq.pop();

            if (dist_u > distances[u]) continue;

            for (int v = 0; v < num_vertices; v++) {
                if (graph[u * num_vertices + v] && distances[u] != INT_MAX &&
                    distances[u] + graph[u * num_vertices + v] < distances[v]) {
                    distances[v] = distances[u] + graph[u * num_vertices + v];
                    pq.push(make_pair(distances[v], v));
                }
            }
        }
    }
}

__global__ void quasidirichletfourier_spfa_parallel(unsigned long long *graph, int num_vertices, unsigned long long source, unsigned long long *distances) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        for (int i = 0; i < num_vertices; i++) {
            distances[i] = INT_MAX;
        }
        distances[source] = 0;

        queue<int> q;
        q.push(source);
        vector<bool> in_queue(num_vertices, false);
        in_queue[source] = true;

        while (!q.empty()) {
            int u = q.front();
            q.pop();
            in_queue[u] = false;

            for (int v = 0; v < num_vertices; v++) {
                if (graph[u * num_vertices + v] && distances[u] != INT_MAX &&
                    distances[u] + graph[u * num_vertices + v] < distances[v]) {
                    distances[v] = distances[u] + graph[u * num_vertices + v];

                    if (!in_queue[v]) {
                        q.push(v);
                        in_queue[v] = true;
                    }
                }
            }
        }
    }
}

__global__ void quasidirichletfourier_bellman_ford_parallel(unsigned long long *graph, int num_vertices, unsigned long long source, unsigned long long *distances) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        for (int i = 0; i < num_vertices; i++) {
            distances[i] = INT_MAX;
        }
        distances[source] = 0;

        for (int k = 0; k < num_vertices - 1; k++) {
            for (int u = 0; u < num_vertices; u++) {
                for (int v = 0; v < num_vertices; v++) {
                    if (graph[u * num_vertices + v] && distances[u] != INT_MAX &&
                        distances[u] + graph[u * num_vertices + v] < distances[v]) {
                        distances[v] = distances[u] + graph[u * num_vertices + v];
                    }
                }
            }
        }

        for (int u = 0; u < num_vertices; u++) {
            for (int v = 0; v < num_vertices; v++) {
                if (graph[u * num_vertices + v] && distances[u] != INT_MAX &&
                    distances[u] + graph[u * num_vertices + v] < distances[v]) {
                    // Negative weight cycle detected
                }
            }
        }
    }
}

__global__ void quasidirichletfourier_dijkstra_heap_parallel(unsigned long long *graph, int num_vertices, unsigned long long source, unsigned long long *distances) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        priority_queue<pair<unsigned long long, int>, vector<pair<unsigned long long, int>>, greater<pair<unsigned long long, int>>> pq;

        for (int i = 0; i < num_vertices; i++) {
            distances[i] = INT_MAX;
        }
        distances[source] = 0;
        pq.push(make_pair(0, source));

        while (!pq.empty()) {
            unsigned long long dist_u = pq.top().first;
            int u = pq.top().second;
            pq.pop();

            if (dist_u > distances[u]) continue;

            for (int v = 0; v < num_vertices; v++) {
                if (graph[u * num_vertices + v] && distances[u] != INT_MAX &&
                    distances[u] + graph[u * num_vertices + v] < distances[v]) {
                    distances[v] = distances[u] + graph[u * num_vertices + v];
                    pq.push(make_pair(distances[v], v));
                }
            }
        }
    }
}

__global__ void quasidirichletfourier_spfa_parallel(unsigned long long *graph, int num_vertices, unsigned long long source, unsigned long long *distances) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        for (int i = 0; i < num_vertices; i++) {
            distances[i] = INT_MAX;
        }
        distances[source] = 0;

        queue<int> q;
        q.push(source);
        vector<bool> in_queue(num_vertices, false);
        in_queue[source] = true;

        while (!q.empty()) {
            int u = q.front();
            q.pop();
            in_queue[u] = false;

            for (int v = 0; v < num_vertices; v++) {
                if (graph[u * num_vertices + v] && distances[u] != INT_MAX &&
                    distances[u] + graph[u * num_vertices + v] < distances[v]) {
                    distances[v] = distances[u] + graph[u * num_vertices + v];

                    if (!in_queue[v]) {
                        q.push(v);
                        in_queue[v] = true;
                    }
                }
            }
        }
    }
}

__global__ void quasidirichletfourier_bellman_ford_parallel(unsigned long long *graph, int num_vertices, unsigned long long source, unsigned long long *distances) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        for (int i = 0; i < num_vertices; i++) {
            distances[i] = INT_MAX;
        }
        distances[source] = 0;

        for (int k = 0; k < num_vertices - 1; k++) {
            for (int u = 0; u < num_vertices; u++) {
                for (int v = 0; v < num_vertices; v++) {
                    if (graph[u * num_vertices + v] && distances[u] != INT_MAX &&
                        distances[u] + graph[u * num_vertices + v] < distances[v]) {
                        distances[v] = distances[u] + graph[u * num_vertices + v];
                    }
                }
            }
        }

        for (int u = 0; u < num_vertices; u++) {
            for (int v = 0; v < num_vertices; v++) {
                if (graph[u * num_vertices + v] && distances[u] != INT_MAX &&
                    distances[u] + graph[u * num_vertices + v] < distances[v]) {
                    // Negative weight cycle detected
                }
            }
        }
    }
}

__global__ void quasidirichletfourier_dijkstra_heap_parallel(unsigned long long *graph, int num_vertices, unsigned long long source, unsigned long long *distances) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        priority_queue<pair<unsigned long long, int>, vector<pair<unsigned long long, int>>, greater<pair<unsigned long long, int>>> pq;

        for (int i = 0; i < num_vertices; i++) {
            distances[i] = INT_MAX;
        }
        distances[source] = 0;
        pq.push(make_pair(0, source));

        while (!pq.empty()) {
            unsigned long long dist_u = pq.top().first;
            int u = pq.top().second;
            pq.pop();

            if (dist_u > distances[u]) continue;

            for (int v = 0; v < num_vertices; v++) {
                if (graph[u * num_vertices + v] && distances[u] != INT_MAX &&
                    distances[u] + graph[u * num_vertices + v] < distances[v]) {
                    distances[v] = distances[u] + graph[u * num_vertices + v];
                    pq.push(make_pair(distances[v], v));
                }
            }
        }
    }
}

__global__ void quasidirichletfourier_spfa_parallel(unsigned long long *graph, int num_vertices, unsigned long long source, unsigned long long *distances) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        queue<int> q;
        vector<bool> in_queue(num_vertices, false);

        for (int i = 0; i < num_vertices; i++) {
            distances[i] = INT_MAX;
        }
        distances[source] = 0;
        q.push(source);
        in_queue[source] = true;

        while (!q.empty()) {
            int u = q.front();
            q.pop();
            in_queue[u] = false;

            for (int v = 0; v < num_vertices; v++) {
                if (graph[u * num_vertices + v] && distances[u] != INT_MAX &&
                    distances[u] + graph[u * num_vertices + v] < distances[v]) {
                    distances[v] = distances[u] + graph[u * num_vertices + v];

                    if (!in_queue[v]) {
                        q.push(v);
                        in_queue[v] = true;
                    }
                }
            }
        }
    }
}

__global__ void quasidirichletfourier_bellman_ford_parallel(unsigned long long *graph, int num_vertices, unsigned long long source, unsigned long long *distances) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        for (int i = 0; i < num_vertices; i++) {
            distances[i] = INT_MAX;
        }
        distances[source] = 0;

        for (int k = 0; k < num_vertices - 1; k++) {
            for (int u = 0; u < num_vertices; u++) {
                for (int v = 0; v < num_vertices; v++) {
                    if (graph[u * num_vertices + v] && distances[u] != INT_MAX &&
                        distances[u] + graph[u * num_vertices + v] < distances[v]) {
                        distances[v] = distances[u] + graph[u * num_vertices + v];
                    }
                }
            }
        }

        for (int u = 0; u < num_vertices; u++) {
            for (int v = 0; v < num_vertices; v++) {
                if (graph[u * num_vertices + v] && distances[u] != INT_MAX &&
                    distances[u] + graph[u * num_vertices + v] < distances[v]) {
                    // Negative weight cycle detected
                }
            }
        }
    }
}

__global__ void quasidirichletfourier_dijkstra_heap_parallel(unsigned long long *graph, int num_vertices, unsigned long long source, unsigned long long *distances) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        priority_queue<pair<unsigned long long, int>, vector<pair<unsigned long long, int>>, greater<pair<unsigned long long, int>>> pq;

        for (int i = 0; i < num_vertices; i++) {
            distances[i] = INT_MAX;
        }
        distances[source] = 0;
        pq.push(make_pair(0, source));

        while (!pq.empty()) {
            unsigned long long dist_u = pq.top().first;
            int u = pq.top().second;
            pq.pop();

            if (dist_u > distances[u]) continue;

            for (int v = 0; v < num_vertices; v++) {
                if (graph[u * num_vertices + v] && distances[u] != INT_MAX &&
                    distances[u] + graph[u * num_vertices + v] < distances[v]) {
                    distances[v] = distances[u] + graph[u * num_vertices + v];
                    pq.push(make_pair(distances[v], v));
                }
            }
        }
    }
}

__global__ void quasidirichletfourier_spfa_parallel(unsigned long long *graph, int num_vertices, unsigned long long source, unsigned long long *distances) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        queue<int> q;
        vector<bool> in_queue(num_vertices, false);

        for (int i = 0; i < num_vertices; i++) {
            distances[i] = INT_MAX;
        }
        distances[source] = 0;
        q.push(source);
        in_queue[source] = true;

        while (!q.empty()) {
            int u = q.front();
            q.pop();
            in_queue[u] = false;

            for (int v = 0; v < num_vertices; v++) {
                if (graph[u * num_vertices + v] && distances[u] != INT_MAX &&
                    distances[u] + graph[u * num_vertices + v] < distances[v]) {
                    distances[v] = distances[u] + graph[u * num_vertices + v];

                    if (!in_queue[v]) {
                        q.push(v);
                        in_queue[v] = true;
                    }
                }
            }
        }
    }
}

__global__ void quasidirichletfourier_bellman_ford_parallel(unsigned long long *graph, int num_vertices, unsigned long long source, unsigned long long *distances) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        for (int i = 0; i < num_vertices; i++) {
            distances[i] = INT_MAX;
        }
        distances[source] = 0;

        for (int k = 0; k < num_vertices - 1; k++) {
            for (int u = 0; u < num_vertices; u++) {
                for (int v = 0; v < num_vertices; v++) {
                    if (graph[u * num_vertices + v] && distances[u] != INT_MAX &&
                        distances[u] + graph[u * num_vertices + v] < distances[v]) {
                        distances[v] = distances[u] + graph[u * num_vertices + v];
                    }
                }
            }
        }

        for (int u = 0; u < num_vertices; u++) {
            for (int v = 0; v < num_vertices; v++) {
                if (graph[u * num_vertices + v] && distances[u] != INT_MAX &&
                    distances[u] + graph[u * num_vertices + v] < distances[v]) {
                    // Negative weight cycle detected
                }
            }
        }
    }
}

__global__ void quasidirichletfourier_dijkstra_heap_parallel(unsigned long long *graph, int num_vertices, unsigned long long source, unsigned long long *distances) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        priority_queue<pair<unsigned long long, int>, vector<pair<unsigned long long, int>>, greater<pair<unsigned long long, int>>> pq;

        for (int i = 0; i < num_vertices; i++) {
            distances[i] = INT_MAX;
        }
        distances[source] = 0;
        pq.push(make_pair(0, source));

        while (!pq.empty()) {
            unsigned long long dist_u = pq.top().first;
            int u = pq.top().second;
            pq.pop();

            if (dist_u > distances[u]) continue;

            for (int v = 0; v < num_vertices; v++) {
                if (graph[u * num_vertices + v] && distances[u] != INT_MAX &&
                    distances[u] + graph[u * num_vertices + v] < distances[v]) {
                    distances[v] = distances[u] + graph[u * num_vertices + v];
                    pq.push(make_pair(distances[v], v));
                }
            }
        }
    }
}

__global__ void quasidirichletfourier_spfa_parallel(unsigned long long *graph, int num_vertices, unsigned long long source, unsigned long long *distances) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        queue<int> q;
        vector<bool> in_queue(num_vertices, false);

        for (int i = 0; i < num_vertices; i++) {
            distances[i] = INT_MAX;
        }
        distances[source] = 0;
        q.push(source);
        in_queue[source] = true;

        while (!q.empty()) {
            int u = q.front();
            q.pop();
            in_queue[u] = false;

            for (int v = 0; v < num_vertices; v++) {
                if (graph[u * num_vertices + v] && distances[u] != INT_MAX &&
                    distances[u] + graph[u * num_vertices + v] < distances[v]) {
                    distances[v] = distances[u] + graph[u * num_vertices + v];

                    if (!in_queue[v]) {
                        q.push(v);
                        in_queue[v] = true;
                    }
                }
            }
        }
    }
}
