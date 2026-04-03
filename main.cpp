#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <random>
#include <vector>
#include <string>
#include <algorithm>
#include <sys/stat.h>
#include <omp.h>

// ─────────────────────────────────────────────
// Usage:
//   ./rw_sim [-d dim] [-n steps] [-w walks] [-s seed] [-o outdir]
//
// Defaults:
//   dim   = 1,2,3,4  (all dimensions if -d not given)
//   steps = 100,1000,10000,100000,1000000
//   walks = 1000
//   seed  = time-based
//   outdir = data
// ─────────────────────────────────────────────

struct WalkResult {
    // final position
    std::vector<long long> pos;
    // L1 and L2 distances
    long long l1;
    double    l2;
    // section (quadrant) counts: 2^D sections
    std::vector<long long> section_counts;
    // steps at which particle returned to origin (0-indexed from step 1)
    std::vector<long long> return_steps;
    // 1D specific: n_minus, n_zero, n_plus, m_over_n
    long long n_minus = 0, n_zero = 0, n_plus = 0;
    double m_over_n = 0.0;
};

// Determine which section (象限) a position is in.
// Returns -1 if any coordinate is 0.
static int get_section(const std::vector<long long>& pos, int D) {
    int sec = 0;
    for (int d = 0; d < D; d++) {
        if (pos[d] == 0) return -1;
        if (pos[d] > 0) sec |= (1 << d);
    }
    return sec;
}

static bool is_origin(const std::vector<long long>& pos, int D) {
    for (int d = 0; d < D; d++) {
        if (pos[d] != 0) return false;
    }
    return true;
}

WalkResult simulate_walk(int D, long long n, std::mt19937_64& rng) {
    WalkResult res;
    res.pos.assign(D, 0);
    int num_sections = (1 << D);
    res.section_counts.assign(num_sections, 0);

    std::uniform_int_distribution<int> coin(0, 1);

    for (long long step = 1; step <= n; step++) {
        // move in each dimension independently
        for (int d = 0; d < D; d++) {
            res.pos[d] += coin(rng) ? 1 : -1;
        }

        // section counting
        int sec = get_section(res.pos, D);
        if (sec >= 0) {
            res.section_counts[sec]++;
        }

        // return to origin check
        if (is_origin(res.pos, D)) {
            res.return_steps.push_back(step);
        }

        // 1D specific statistics
        if (D == 1) {
            if (res.pos[0] < 0) res.n_minus++;
            else if (res.pos[0] == 0) res.n_zero++;
            else res.n_plus++;
        }
    }

    // compute distances
    res.l1 = 0;
    res.l2 = 0.0;
    for (int d = 0; d < D; d++) {
        res.l1 += std::abs(res.pos[d]);
        res.l2 += (double)res.pos[d] * res.pos[d];
    }
    res.l2 = std::sqrt(res.l2);

    // 1D m/n
    if (D == 1) {
        double m = 0.5 * res.n_zero + std::max(res.n_minus, res.n_plus);
        res.m_over_n = m / (double)n;
    }

    return res;
}

static void ensure_dir(const std::string& path) {
    mkdir(path.c_str(), 0755);
}

int main(int argc, char* argv[]) {
    // defaults
    std::vector<int> dims = {1, 2, 3, 4};
    std::vector<long long> steps_list = {100, 1000, 10000, 100000, 1000000};
    int num_walks = 1000;
    unsigned long long seed = (unsigned long long)time(nullptr);
    std::string outdir = "data";


    // parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            dims.clear();
            dims.push_back(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            steps_list.clear();
            steps_list.push_back(atoll(argv[++i]));
        } else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
            num_walks = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            seed = strtoull(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            outdir = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [-d dim] [-n steps] [-w walks] [-s seed] [-o outdir]\n", argv[0]);
            printf("  -d dim     Dimension (default: 1,2,3,4 all)\n");
            printf("  -n steps   Number of steps (default: 100..1000000)\n");
            printf("  -w walks   Number of walks (default: 1000)\n");
            printf("  -s seed    Random seed (default: time-based)\n");
            printf("  -o outdir  Output directory (default: data)\n");
            return 0;
        }
    }

    ensure_dir(outdir);
    printf("Random Walk Simulation\n");
    printf("======================\n");
    printf("Seed: %llu\n", seed);
    printf("Walks per (D, n): %d\n", num_walks);
    printf("Dimensions: ");
    for (int d : dims) printf("%d ", d);
    printf("\nSteps: ");
    for (long long s : steps_list) printf("%lld ", s);
    printf("\n\n");

    for (int D : dims) {
        for (long long n : steps_list) {
            printf("[D=%d, n=%lld] Running %d walks...\n", D, n, num_walks);
            fflush(stdout);

            double t0 = omp_get_wtime();

            std::vector<WalkResult> results(num_walks);

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                // each thread gets its own RNG seeded deterministically
                std::mt19937_64 rng(seed + (unsigned long long)D * 1000000ULL
                                         + (unsigned long long)n * 100ULL
                                         + (unsigned long long)tid);

                #pragma omp for schedule(dynamic)
                for (int w = 0; w < num_walks; w++) {
                    // further mix walk index into seed for uniqueness
                    std::mt19937_64 walk_rng(rng() + (unsigned long long)w);
                    results[w] = simulate_walk(D, n, walk_rng);
                }
            }

            double elapsed = omp_get_wtime() - t0;
            printf("  Done in %.2f s\n", elapsed);

            // --- Write distance CSV ---
            {
                char fname[512];
                snprintf(fname, sizeof(fname), "%s/dist_D%d_n%lld.csv",
                         outdir.c_str(), D, n);
                FILE* f = fopen(fname, "w");
                fprintf(f, "walk,l1,l2");
                for (int d = 0; d < D; d++) fprintf(f, ",x%d", d);
                fprintf(f, "\n");
                for (int w = 0; w < num_walks; w++) {
                    fprintf(f, "%d,%lld,%.6f", w, results[w].l1, results[w].l2);
                    for (int d = 0; d < D; d++)
                        fprintf(f, ",%lld", results[w].pos[d]);
                    fprintf(f, "\n");
                }
                fclose(f);
            }

            // --- Write section (quadrant) CSV ---
            {
                char fname[512];
                snprintf(fname, sizeof(fname), "%s/section_D%d_n%lld.csv",
                         outdir.c_str(), D, n);
                FILE* f = fopen(fname, "w");
                int num_sec = (1 << D);
                fprintf(f, "walk");
                for (int s = 0; s < num_sec; s++) fprintf(f, ",sec%d", s);
                fprintf(f, "\n");
                for (int w = 0; w < num_walks; w++) {
                    fprintf(f, "%d", w);
                    for (int s = 0; s < num_sec; s++)
                        fprintf(f, ",%lld", results[w].section_counts[s]);
                    fprintf(f, "\n");
                }
                fclose(f);
            }

            // --- Write return-to-origin CSV ---
            {
                char fname[512];
                snprintf(fname, sizeof(fname), "%s/return_D%d_n%lld.csv",
                         outdir.c_str(), D, n);
                FILE* f = fopen(fname, "w");
                fprintf(f, "walk,num_returns,first_return,return_steps\n");
                for (int w = 0; w < num_walks; w++) {
                    auto& rs = results[w].return_steps;
                    long long first = rs.empty() ? -1 : rs[0];
                    fprintf(f, "%d,%zu,%lld,", w, rs.size(), first);
                    for (size_t i = 0; i < rs.size(); i++) {
                        if (i > 0) fprintf(f, ";");
                        fprintf(f, "%lld", rs[i]);
                    }
                    fprintf(f, "\n");
                }
                fclose(f);
            }

            // --- Write 1D specific CSV ---
            if (D == 1) {
                char fname[512];
                snprintf(fname, sizeof(fname), "%s/onedim_n%lld.csv",
                         outdir.c_str(), n);
                FILE* f = fopen(fname, "w");
                fprintf(f, "walk,n_minus,n_zero,n_plus,m_over_n\n");
                for (int w = 0; w < num_walks; w++) {
                    fprintf(f, "%d,%lld,%lld,%lld,%.10f\n", w,
                            results[w].n_minus, results[w].n_zero,
                            results[w].n_plus, results[w].m_over_n);
                }
                fclose(f);
            }
        }
    }

    printf("\nAll simulations complete. Data written to '%s/'\n", outdir.c_str());
    return 0;
}
