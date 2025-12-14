// qlearning_grid.c
// シンプルな Q-learning の C 実装（5x5 GridWorld）
// コンパイル: gcc qlearning_grid.c -O2 -o qlearn -lm

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define ROWS 5
#define COLS 5
#define N_STATES (ROWS*COLS)
#define N_ACTIONS 4

// actions: 0=UP,1=RIGHT,2=DOWN,3=LEFT
const int dr[N_ACTIONS] = { -1, 0, 1, 0 };
const int dc[N_ACTIONS] = { 0, 1, 0, -1 };

int state_idx(int r, int c) { return r * COLS + c; }
void idx_to_rc(int idx, int* r, int* c) { *r = idx / COLS; *c = idx % COLS; }

// 環境のステップ: s, a -> s', reward, done
void env_step(int s, int a, int* s2, double* reward, int* done) {
    int r, c;
    idx_to_rc(s, &r, &c);
    int nr = r + dr[a];
    int nc = c + dc[a];
    // 壁ははみ出し防止（到達不可なら元に留まる）
    if (nr < 0) nr = 0;
    if (nr >= ROWS) nr = ROWS - 1;
    if (nc < 0) nc = 0;
    if (nc >= COLS) nc = COLS - 1;
    *s2 = state_idx(nr, nc);
    // ゴールは (ROWS-1, COLS-1)
    if (nr == ROWS - 1 && nc == COLS - 1) {
        *reward = 100.0;
        *done = 1;
    }
    else {
        *reward = -1.0;
        *done = 0;
    }
}

// argmax over actions (ties broken randomly)
int argmax_action(double Q[], int s) {
    double best = -1e300;
    int best_count = 0;
    int best_idx = 0;
    for (int a = 0; a < N_ACTIONS; a++) {
        double v = Q[s * N_ACTIONS + a];
        if (v > best + 1e-9) {
            best = v;
            best_count = 1;
            best_idx = a;
        }
        else if (fabs(v - best) <= 1e-9) {
            // tie -> random tie-break
            best_count++;
            if (rand() % best_count == 0) best_idx = a;
        }
    }
    return best_idx;
}

// epsilon-greedy
int select_action(double Q[], int s, double epsilon) {
    if (((double)rand() / RAND_MAX) < epsilon) {
        return rand() % N_ACTIONS;
    }
    else {
        return argmax_action(Q, s);
    }
}

// ユーティリティ: 方策を表示（矢印）
void print_policy(double Q[]) {
    printf("Learned policy (arrows): ^ > v <\n");
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            int s = state_idx(r, c);
            if (r == ROWS - 1 && c == COLS - 1) {
                printf(" G  ");
                continue;
            }
            int a = argmax_action(Q, s);
            char ch = '?';
            if (a == 0) ch = '^';
            if (a == 1) ch = '>';
            if (a == 2) ch = 'v';
            if (a == 3) ch = '<';
            printf(" %c  ", ch);
        }
        printf("\n");
    }
}

// メイン
int main() {
    srand((unsigned)time(NULL));

    // ハイパーパラメータ
    const int episodes = 5000;
    const int max_steps = 200;
    const double alpha = 0.1;    // 学習率
    const double gamma = 0.99;   // 割引率
    const double eps_start = 1.0;
    const double eps_end = 0.05;

    // Q table (N_STATES x N_ACTIONS), 初期値 0
    double* Q = (double*)calloc(N_STATES * N_ACTIONS, sizeof(double));
    if (!Q) { fprintf(stderr, "メモリ確保失敗\n"); return 1; }

    int success_count = 0;

    for (int ep = 1; ep <= episodes; ep++) {
        // epsilon 線形減衰
        double epsilon = eps_end + (eps_start - eps_end) * (1.0 - (double)ep / episodes);
        int s = state_idx(0, 0); // start at (0,0)
        int done = 0;

        for (int step = 0; step < max_steps && !done; step++) {
            int a = select_action(Q, s, epsilon);

            int s2; double reward; int d;
            env_step(s, a, &s2, &reward, &d);

            // Q-learning update
            // Q(s,a) += alpha * [reward + gamma * max_a' Q(s',a') - Q(s,a)]
            int best_a2 = argmax_action(Q, s2);
            double target = reward + (d ? 0.0 : gamma * Q[s2 * N_ACTIONS + best_a2]);
            Q[s * N_ACTIONS + a] += alpha * (target - Q[s * N_ACTIONS + a]);

            s = s2;
            done = d;
        }

        if (done) success_count++;

        // ログ出力（進捗）
        if (ep % 500 == 0 || ep == 1) {
            printf("Episode %4d / %d  epsilon=%.3f  success_rate=%.3f\n",
                ep, episodes, epsilon, (double)success_count / ep);
        }
    }

    printf("\nFinal success rate: %.3f (%d/%d)\n", (double)success_count / episodes, success_count, episodes);
    printf("\n");
    print_policy(Q);

    free(Q);
    return 0;
}
