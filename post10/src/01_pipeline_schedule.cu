// 01_pipeline_schedule.cu  --  Post 10: Pipeline Parallelism and MoE
//
// Simulates the two classic pipeline schedules and derives their properties:
//
//   GPipe:  run ALL micro-batch forwards, flush, then all backwards.
//   1F1B:   after a short warmup, each stage alternates one-forward-one-backward.
//
// The surprise this program demonstrates: with equal-cost F and B ops, GPipe
// and 1F1B have the SAME makespan and the SAME bubble fraction:
//
//     bubble = (P-1) / (M + P-1)
//
// 1F1B's win is MEMORY, not bubble: GPipe must hold the activations of ALL M
// micro-batches on every stage until the backward pass drains them. 1F1B caps
// in-flight activations at P-k per stage -- independent of M. That is what
// lets you crank M up (to shrink the bubble) without running out of HBM.
//
// This is a pure scheduling simulation (no GPU work) -- the event-driven
// simulator enforces the real dependencies:
//   F_i at stage k  needs  F_i at stage k-1
//   B_i at stage k  needs  B_i at stage k+1   (B_i at last stage needs F_i)
// and the 1F1B policy: prefer the oldest ready backward; only start a new
// forward if in-flight activations < P-k.
//
// Program 02 then runs a REAL pipeline on the GPU with the same structure.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include "../include/utils.cuh"

struct Op { int start, end; char kind; int mb; };   // one cell of the schedule

// ─────────────────────────────────────────────────────────────────────────────
// Event-driven 1F1B simulator (unit-time ops).
// Returns per-stage op lists; fills peak_inflight[k].
// ─────────────────────────────────────────────────────────────────────────────
static int simulate_1f1b(int P, int M,
                         std::vector<std::vector<Op>>& sched,
                         std::vector<int>& peak_inflight) {
    std::vector<std::vector<int>> f_done(P, std::vector<int>(M, -1));
    std::vector<std::vector<int>> b_done(P, std::vector<int>(M, -1));
    std::vector<int> t_free(P, 0), next_f(P, 0), next_b(P, 0), inflight(P, 0);
    sched.assign(P, {});
    peak_inflight.assign(P, 0);

    int done_ops = 0, total = 2 * M * P;
    while (done_ops < total) {
        int bs = 1 << 30, bk = -1; char bkind = 0;
        for (int k = 0; k < P; k++) {
            int cap = P - k;
            // backward candidate (preferred on ties)
            if (next_b[k] < M) {
                int i = next_b[k];
                int dep = (k == P-1) ? f_done[k][i] : b_done[k+1][i];
                if (dep >= 0) {
                    int s = t_free[k] > dep ? t_free[k] : dep;
                    if (s < bs || (s == bs && bkind == 'F')) { bs = s; bk = k; bkind = 'B'; }
                }
            }
            // forward candidate (respects the in-flight cap)
            if (next_f[k] < M && inflight[k] < cap) {
                int i = next_f[k];
                int dep = (k == 0) ? 0 : f_done[k-1][i];
                if (dep >= 0) {
                    int s = t_free[k] > dep ? t_free[k] : dep;
                    if (s < bs) { bs = s; bk = k; bkind = 'F'; }
                }
            }
        }
        int e = bs + 1;
        if (bkind == 'F') {
            int i = next_f[bk]; f_done[bk][i] = e; next_f[bk]++;
            inflight[bk]++;
            if (inflight[bk] > peak_inflight[bk]) peak_inflight[bk] = inflight[bk];
        } else {
            int i = next_b[bk]; b_done[bk][i] = e; next_b[bk]++; inflight[bk]--;
        }
        sched[bk].push_back({bs, e, bkind, bkind=='F' ? next_f[bk]-1 : next_b[bk]-1});
        t_free[bk] = e;
        done_ops++;
    }
    int makespan = 0;
    for (int k = 0; k < P; k++) makespan = std::max(makespan, t_free[k]);
    return makespan;
}

// ─────────────────────────────────────────────────────────────────────────────
// GPipe simulator: forward wave, flush, backward wave (reverse order).
// Peak in-flight activations per stage = M (all held until backward).
// ─────────────────────────────────────────────────────────────────────────────
static int simulate_gpipe(int P, int M,
                          std::vector<std::vector<Op>>& sched,
                          std::vector<int>& peak_inflight) {
    std::vector<std::vector<int>> f_done(P, std::vector<int>(M, -1));
    std::vector<std::vector<int>> b_done(P, std::vector<int>(M, -1));
    std::vector<int> t_free(P, 0);
    sched.assign(P, {});
    peak_inflight.assign(P, M);   // the defining property of GPipe

    // forward wave
    for (int i = 0; i < M; i++)
        for (int k = 0; k < P; k++) {
            int dep = (k == 0) ? 0 : f_done[k-1][i];
            int s = std::max(t_free[k], dep);
            f_done[k][i] = s + 1; t_free[k] = s + 1;
            sched[k].push_back({s, s+1, 'F', i});
        }
    // backward wave, reverse micro-batch order, flows P-1 -> 0
    for (int i = M-1; i >= 0; i--)
        for (int k = P-1; k >= 0; k--) {
            int dep = (k == P-1) ? f_done[k][i] : b_done[k+1][i];
            int s = std::max(t_free[k], dep);
            b_done[k][i] = s + 1; t_free[k] = s + 1;
            sched[k].push_back({s, s+1, 'B', i});
        }
    int makespan = 0;
    for (int k = 0; k < P; k++) makespan = std::max(makespan, t_free[k]);
    return makespan;
}

// ─────────────────────────────────────────────────────────────────────────────
// ASCII Gantt (one char per slot; F=fwd digit, b=bwd digit, . = idle)
// ─────────────────────────────────────────────────────────────────────────────
static void print_gantt(const char* title,
                        const std::vector<std::vector<Op>>& sched,
                        int P, int makespan) {
    printf("\n  %s\n", title);
    printf("  %-8s", "");
    for (int t = 0; t < makespan; t++) putchar(t % 10 == 0 ? '|' : ' ');
    printf("\n");
    for (int k = 0; k < P; k++) {
        std::vector<char> row(makespan, '.');
        for (const Op& op : sched[k])
            row[op.start] = (op.kind == 'F') ? (char)('0' + op.mb % 10)
                                             : (char)('a' + op.mb % 26);
        printf("  stage %d ", k);
        for (int t = 0; t < makespan; t++) putchar(row[t]);
        printf("\n");
    }
    printf("  (digits = forward micro-batch, letters = backward: a=B0 b=B1 ...)\n");
}

int main(int argc, char** argv) {
    // Pure scheduling analysis -- no GPU required (runs fine without one).
    int P = (argc > 1) ? atoi(argv[1]) : 4;
    int M = (argc > 2) ? atoi(argv[2]) : 6;

    printf("Pipeline schedule simulator: P=%d stages, M=%d micro-batches\n", P, M);
    printf("(unit-cost ops; real backward is ~2x forward but ratios are the same)\n");

    std::vector<std::vector<Op>> s_gpipe, s_1f1b;
    std::vector<int> peak_gpipe, peak_1f1b;

    int mk_g = simulate_gpipe(P, M, s_gpipe, peak_gpipe);
    int mk_o = simulate_1f1b (P, M, s_1f1b, peak_1f1b);

    section("The two schedules, side by side");
    print_gantt("GPipe: all forwards, flush, all backwards", s_gpipe, P, mk_g);
    print_gantt("1F1B: warmup, then alternate", s_1f1b, P, mk_o);

    section("Makespan and bubble");
    double ideal   = 2.0 * M;                        // busy slots per stage
    double bub_g   = 1.0 - ideal / mk_g;
    double bub_o   = 1.0 - ideal / mk_o;
    double formula = (double)(P - 1) / (M + P - 1);
    printf("  %-10s %10s %10s %14s\n", "schedule", "makespan", "bubble", "formula");
    printf("  %-10s %10d %9.1f%% %13.1f%%\n", "GPipe", mk_g, 100*bub_g, 100*formula);
    printf("  %-10s %10d %9.1f%% %13.1f%%\n", "1F1B",  mk_o, 100*bub_o, 100*formula);
    printf("\n  Same makespan, same bubble: (P-1)/(M+P-1). The schedules differ\n");
    printf("  in MEMORY, not speed:\n\n");

    printf("  Peak in-flight activations (micro-batches held per stage):\n");
    printf("  %-8s", "stage");
    for (int k = 0; k < P; k++) printf(" %5d", k);
    printf("\n  %-8s", "GPipe");
    for (int k = 0; k < P; k++) printf(" %5d", peak_gpipe[k]);
    printf("   <- grows with M!\n");
    printf("  %-8s", "1F1B");
    for (int k = 0; k < P; k++) printf(" %5d", peak_1f1b[k]);
    printf("   <- capped at P-k, independent of M\n");

    section("Why the cap matters: shrinking the bubble costs GPipe memory");
    printf("  bubble = (P-1)/(M+P-1) -> to shrink it you raise M. But under\n");
    printf("  GPipe every stage stores M activation sets. Under 1F1B, at most\n");
    printf("  P-k. So 1F1B lets M grow to whatever the bubble target needs:\n\n");
    printf("  %-8s %-10s %-22s %-22s\n", "M", "bubble", "GPipe acts/stage0", "1F1B acts/stage0");
    for (int m : {P, 2*P, 4*P, 8*P, 16*P}) {
        double b = (double)(P - 1) / (m + P - 1);
        printf("  %-8d %8.1f%% %-22d %-22d\n", m, 100*b, m, P);
    }
    printf("\n  Rule of thumb: M >= 4P keeps the bubble under ~20%%.\n");
    printf("  Interleaved schedules (virtual stages) shrink it further by\n");
    printf("  giving each GPU multiple smaller stages -- same ideas, finer grain.\n");
    return 0;
}
