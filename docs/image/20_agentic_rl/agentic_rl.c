/* ============================================================================
 * Agentic RL - Complete C Implementation (C99/C11)
 * ============================================================================
 * A2C (Advantage Actor-Critic) + LSTM + Adam
 * Pure C. No external libraries except standard C library and math.h.
 *
 * Task: Reach exactly 10 from 1 using actions {+1, +2, x2}
 * Reward: +10 (success), -5 (fail/overshoot), -0.05 per step
 *
 * Build: gcc -std=c11 -O2 -lm -o agentic_rl agentic_rl.c
 * Run:   ./agentic_rl
 * ============================================================================
 */

#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ============================================================================
 * RNG (xorshift64*)
 * ============================================================================ */
typedef struct {
    uint64_t state;
} RNG;

static uint64_t rng_u64(RNG* rng) {
    uint64_t x = rng->state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng->state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static double rng_uniform(RNG* rng) {
    /* [0, 1) */
    return (rng_u64(rng) >> 11) * (1.0 / (1ULL << 53));
}

static double rng_normal(RNG* rng) {
    /* Box-Muller */
    double u1 = rng_uniform(rng);
    double u2 = rng_uniform(rng);
    if (u1 < 1e-10) u1 = 1e-10;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static int rng_choice(RNG* rng, const double* probs, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += probs[i];
    double r = rng_uniform(rng) * sum;
    double cum = 0.0;
    for (int i = 0; i < n; i++) {
        cum += probs[i];
        if (r <= cum) return i;
    }
    return n - 1;
}

static void rng_init(RNG* rng, uint64_t seed) {
    rng->state = seed ? seed : 0x123456789ABCDEFULL;
}

/* ============================================================================
 * Math utilities
 * ============================================================================ */
static inline double relu(double x) { return x > 0.0 ? x : 0.0; }
static inline double relu_deriv(double x) { return x > 0.0 ? 1.0 : 0.0; }

static double sigmoid(double x) {
    if (x >= 0.0) {
        double z = exp(-x);
        return 1.0 / (1.0 + z);
    } else {
        double z = exp(x);
        return z / (1.0 + z);
    }
}

static inline double dsigmoid(double s) { return s * (1.0 - s); }
static inline double dtanh(double t) { return 1.0 - t * t; }

static void softmax(const double* x, double* out, int n) {
    double maxv = x[0];
    for (int i = 1; i < n; i++) if (x[i] > maxv) maxv = x[i];
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        out[i] = exp(x[i] - maxv);
        sum += out[i];
    }
    for (int i = 0; i < n; i++) out[i] /= sum;
}

/* ============================================================================
 * Linear Layer
 * ============================================================================ */
typedef struct {
    int in_features;
    int out_features;
    double* weight;      /* out x in */
    double* bias;        /* out */
    double* grad_w;
    double* grad_b;
    double* m_w;
    double* v_w;
    double* m_b;
    double* v_b;
    /* forward cache */
    double* last_input;
    double* last_pre_act;
} Linear;

static Linear* linear_create(int in_f, int out_f, RNG* rng) {
    Linear* L = (Linear*)calloc(1, sizeof(Linear));
    L->in_features = in_f;
    L->out_features = out_f;
    int wsize = out_f * in_f;
    L->weight = (double*)calloc(wsize, sizeof(double));
    L->bias = (double*)calloc(out_f, sizeof(double));
    L->grad_w = (double*)calloc(wsize, sizeof(double));
    L->grad_b = (double*)calloc(out_f, sizeof(double));
    L->m_w = (double*)calloc(wsize, sizeof(double));
    L->v_w = (double*)calloc(wsize, sizeof(double));
    L->m_b = (double*)calloc(out_f, sizeof(double));
    L->v_b = (double*)calloc(out_f, sizeof(double));
    L->last_input = (double*)calloc(in_f, sizeof(double));
    L->last_pre_act = (double*)calloc(out_f, sizeof(double));

    double scale = sqrt(2.0 / (double)(in_f + out_f));
    for (int i = 0; i < wsize; i++)
        L->weight[i] = rng_normal(rng) * scale;
    return L;
}

static void linear_free(Linear* L) {
    if (!L) return;
    free(L->weight); free(L->bias); free(L->grad_w); free(L->grad_b);
    free(L->m_w); free(L->v_w); free(L->m_b); free(L->v_b);
    free(L->last_input); free(L->last_pre_act);
    free(L);
}

/* weight[i,j] accessor */
#define W(L,i,j) ((L)->weight[(i) * (L)->in_features + (j)])
#define GW(L,i,j) ((L)->grad_w[(i) * (L)->in_features + (j)])
#define MW(L,i,j) ((L)->m_w[(i) * (L)->in_features + (j)])
#define VW(L,i,j) ((L)->v_w[(i) * (L)->in_features + (j)])

static void linear_forward(Linear* L, const double* x, double* out) {
    memcpy(L->last_input, x, L->in_features * sizeof(double));
    for (int i = 0; i < L->out_features; i++) {
        double sum = L->bias[i];
        for (int j = 0; j < L->in_features; j++)
            sum += W(L, i, j) * x[j];
        L->last_pre_act[i] = sum;
        out[i] = sum;
    }
}

/* backward: given grad_out (out_features), accumulate gradients,
   return grad_x (caller must provide buffer) */
static void linear_backward(Linear* L, const double* grad_out, double* grad_x) {
    for (int j = 0; j < L->in_features; j++) grad_x[j] = 0.0;
    for (int i = 0; i < L->out_features; i++) {
        L->grad_b[i] += grad_out[i];
        for (int j = 0; j < L->in_features; j++) {
            GW(L, i, j) += grad_out[i] * L->last_input[j];
            grad_x[j] += grad_out[i] * W(L, i, j);
        }
    }
}

static void linear_zero_grad(Linear* L) {
    memset(L->grad_w, 0, L->out_features * L->in_features * sizeof(double));
    memset(L->grad_b, 0, L->out_features * sizeof(double));
}

/* ============================================================================
 * LSTM Cell
 * ============================================================================ */
typedef struct {
    int input_size;
    int hidden_size;
    /* Weights: W_f, W_i, W_c, W_o  (hidden_size x input_size) */
    double* Wf; double* Wi; double* Wc; double* Wo;
    double* bf; double* bi; double* bc; double* bo;
    /* Gradients */
    double* gWf; double* gWi; double* gWc; double* gWo;
    double* gbf; double* gbi; double* gbc; double* gbo;
    /* Adam states */
    double* mWf; double* mWi; double* mWc; double* mWo;
    double* vWf; double* vWi; double* vWc; double* vWo;
    double* mbf; double* mbi; double* mbc; double* mbo;
    double* vbf; double* vbi; double* vbc; double* vbo;
    /* Forward cache (single step) */
    double* last_x;
    double* last_h_prev;
    double* last_c_prev;
    double* last_f; double* last_i; double* last_c_tilde;
    double* last_o; double* last_c; double* last_h;
} LSTMCell;

#define LW(cell,i,j) ((cell)->Wf[(i) * (cell)->input_size + (j)])
/* helper macros for each gate matrix */
#define Wf_(c,i,j) ((c)->Wf[(i)*(c)->input_size+(j)])
#define Wi_(c,i,j) ((c)->Wi[(i)*(c)->input_size+(j)])
#define Wc_(c,i,j) ((c)->Wc[(i)*(c)->input_size+(j)])
#define Wo_(c,i,j) ((c)->Wo[(i)*(c)->input_size+(j)])
#define gWf_(c,i,j) ((c)->gWf[(i)*(c)->input_size+(j)])
#define gWi_(c,i,j) ((c)->gWi[(i)*(c)->input_size+(j)])
#define gWc_(c,i,j) ((c)->gWc[(i)*(c)->input_size+(j)])
#define gWo_(c,i,j) ((c)->gWo[(i)*(c)->input_size+(j)])
#define mWf_(c,i,j) ((c)->mWf[(i)*(c)->input_size+(j)])
#define mWi_(c,i,j) ((c)->mWi[(i)*(c)->input_size+(j)])
#define mWc_(c,i,j) ((c)->mWc[(i)*(c)->input_size+(j)])
#define mWo_(c,i,j) ((c)->mWo[(i)*(c)->input_size+(j)])
#define vWf_(c,i,j) ((c)->vWf[(i)*(c)->input_size+(j)])
#define vWi_(c,i,j) ((c)->vWi[(i)*(c)->input_size+(j)])
#define vWc_(c,i,j) ((c)->vWc[(i)*(c)->input_size+(j)])
#define vWo_(c,i,j) ((c)->vWo[(i)*(c)->input_size+(j)])

static LSTMCell* lstm_create(int in_sz, int hid_sz, RNG* rng) {
    LSTMCell* cell = (LSTMCell*)calloc(1, sizeof(LSTMCell));
    cell->input_size = in_sz;
    cell->hidden_size = hid_sz;
    int sz = hid_sz * in_sz;
    cell->Wf = (double*)calloc(sz, sizeof(double));
    cell->Wi = (double*)calloc(sz, sizeof(double));
    cell->Wc = (double*)calloc(sz, sizeof(double));
    cell->Wo = (double*)calloc(sz, sizeof(double));
    cell->bf = (double*)calloc(hid_sz, sizeof(double));
    cell->bi = (double*)calloc(hid_sz, sizeof(double));
    cell->bc = (double*)calloc(hid_sz, sizeof(double));
    cell->bo = (double*)calloc(hid_sz, sizeof(double));

    cell->gWf = (double*)calloc(sz, sizeof(double));
    cell->gWi = (double*)calloc(sz, sizeof(double));
    cell->gWc = (double*)calloc(sz, sizeof(double));
    cell->gWo = (double*)calloc(sz, sizeof(double));
    cell->gbf = (double*)calloc(hid_sz, sizeof(double));
    cell->gbi = (double*)calloc(hid_sz, sizeof(double));
    cell->gbc = (double*)calloc(hid_sz, sizeof(double));
    cell->gbo = (double*)calloc(hid_sz, sizeof(double));

    cell->mWf = (double*)calloc(sz, sizeof(double));
    cell->mWi = (double*)calloc(sz, sizeof(double));
    cell->mWc = (double*)calloc(sz, sizeof(double));
    cell->mWo = (double*)calloc(sz, sizeof(double));
    cell->mbf = (double*)calloc(hid_sz, sizeof(double));
    cell->mbi = (double*)calloc(hid_sz, sizeof(double));
    cell->mbc = (double*)calloc(hid_sz, sizeof(double));
    cell->mbo = (double*)calloc(hid_sz, sizeof(double));

    cell->vWf = (double*)calloc(sz, sizeof(double));
    cell->vWi = (double*)calloc(sz, sizeof(double));
    cell->vWc = (double*)calloc(sz, sizeof(double));
    cell->vWo = (double*)calloc(sz, sizeof(double));
    cell->vbf = (double*)calloc(hid_sz, sizeof(double));
    cell->vbi = (double*)calloc(hid_sz, sizeof(double));
    cell->vbc = (double*)calloc(hid_sz, sizeof(double));
    cell->vbo = (double*)calloc(hid_sz, sizeof(double));

    cell->last_x = (double*)calloc(in_sz, sizeof(double));
    cell->last_h_prev = (double*)calloc(hid_sz, sizeof(double));
    cell->last_c_prev = (double*)calloc(hid_sz, sizeof(double));
    cell->last_f = (double*)calloc(hid_sz, sizeof(double));
    cell->last_i = (double*)calloc(hid_sz, sizeof(double));
    cell->last_c_tilde = (double*)calloc(hid_sz, sizeof(double));
    cell->last_o = (double*)calloc(hid_sz, sizeof(double));
    cell->last_c = (double*)calloc(hid_sz, sizeof(double));
    cell->last_h = (double*)calloc(hid_sz, sizeof(double));

    double scale = sqrt(2.0 / (double)(in_sz + hid_sz));
    for (int i = 0; i < sz; i++) {
        cell->Wf[i] = rng_normal(rng) * scale;
        cell->Wi[i] = rng_normal(rng) * scale;
        cell->Wc[i] = rng_normal(rng) * scale;
        cell->Wo[i] = rng_normal(rng) * scale;
    }
    for (int j = 0; j < hid_sz; j++) cell->bf[j] = 1.0; /* forget gate bias */
    return cell;
}

static void lstm_free(LSTMCell* cell) {
    if (!cell) return;
    free(cell->Wf); free(cell->Wi); free(cell->Wc); free(cell->Wo);
    free(cell->bf); free(cell->bi); free(cell->bc); free(cell->bo);
    free(cell->gWf); free(cell->gWi); free(cell->gWc); free(cell->gWo);
    free(cell->gbf); free(cell->gbi); free(cell->gbc); free(cell->gbo);
    free(cell->mWf); free(cell->mWi); free(cell->mWc); free(cell->mWo);
    free(cell->mbf); free(cell->mbi); free(cell->mbc); free(cell->mbo);
    free(cell->vWf); free(cell->vWi); free(cell->vWc); free(cell->vWo);
    free(cell->vbf); free(cell->vbi); free(cell->vbc); free(cell->vbo);
    free(cell->last_x); free(cell->last_h_prev); free(cell->last_c_prev);
    free(cell->last_f); free(cell->last_i); free(cell->last_c_tilde);
    free(cell->last_o); free(cell->last_c); free(cell->last_h);
    free(cell);
}

static void lstm_forward(LSTMCell* cell, const double* x,
                       const double* h_prev, const double* c_prev,
                       double* h_out) {
    memcpy(cell->last_x, x, cell->input_size * sizeof(double));
    memcpy(cell->last_h_prev, h_prev, cell->hidden_size * sizeof(double));
    memcpy(cell->last_c_prev, c_prev, cell->hidden_size * sizeof(double));

    int H = cell->hidden_size;
    int I = cell->input_size;

    for (int j = 0; j < H; j++) {
        double f_sum = cell->bf[j];
        double i_sum = cell->bi[j];
        double c_sum = cell->bc[j];
        double o_sum = cell->bo[j];
        for (int k = 0; k < I; k++) {
            f_sum += Wf_(cell, j, k) * x[k];
            i_sum += Wi_(cell, j, k) * x[k];
            c_sum += Wc_(cell, j, k) * x[k];
            o_sum += Wo_(cell, j, k) * x[k];
        }
        /* Simplified: also multiply by h_prev using same weight matrix */
        for (int k = 0; k < H; k++) {
            /* Use first H columns of W as hidden weights (simplified) */
            if (k < I) {
                f_sum += Wf_(cell, j, k) * h_prev[k];
                i_sum += Wi_(cell, j, k) * h_prev[k];
                c_sum += Wc_(cell, j, k) * h_prev[k];
                o_sum += Wo_(cell, j, k) * h_prev[k];
            }
        }
        cell->last_f[j] = sigmoid(f_sum);
        cell->last_i[j] = sigmoid(i_sum);
        cell->last_c_tilde[j] = tanh(c_sum);
        cell->last_c[j] = cell->last_f[j] * c_prev[j] + cell->last_i[j] * cell->last_c_tilde[j];
        cell->last_o[j] = sigmoid(o_sum);
        cell->last_h[j] = cell->last_o[j] * tanh(cell->last_c[j]);
        h_out[j] = cell->last_h[j];
    }
}

/* Simplified LSTM backward for single step.
   dh_next: gradient w.r.t. h (hidden_size)
   Returns dh_prev through pointer argument. */
static void lstm_backward(LSTMCell* cell, const double* dh_next, double* dh_prev) {
    int H = cell->hidden_size;
    int I = cell->input_size;
    for (int j = 0; j < H; j++) dh_prev[j] = 0.0;

    for (int j = 0; j < H; j++) {
        double tanh_c = tanh(cell->last_c[j]);
        double do_ = dh_next[j] * tanh_c;
        double dc = dh_next[j] * cell->last_o[j] * dtanh(tanh_c);
        double di = dc * cell->last_c_tilde[j];
        double dc_tilde = dc * cell->last_i[j];
        double df = dc * cell->last_c_prev[j];

        double g_do = do_ * dsigmoid(cell->last_o[j]);
        double g_di = di * dsigmoid(cell->last_i[j]);
        double g_df = df * dsigmoid(cell->last_f[j]);
        double g_dc_tilde = dc_tilde * dtanh(cell->last_c_tilde[j]);

        cell->gbf[j] += g_df;
        cell->gbi[j] += g_di;
        cell->gbc[j] += g_dc_tilde;
        cell->gbo[j] += g_do;

        for (int k = 0; k < I; k++) {
            gWf_(cell, j, k) += g_df * cell->last_x[k];
            gWi_(cell, j, k) += g_di * cell->last_x[k];
            gWc_(cell, j, k) += g_dc_tilde * cell->last_x[k];
            gWo_(cell, j, k) += g_do * cell->last_x[k];
        }
        for (int k = 0; k < H && k < I; k++) {
            dh_prev[k] += g_df * Wf_(cell, j, k)
                       + g_di * Wi_(cell, j, k)
                       + g_dc_tilde * Wc_(cell, j, k)
                       + g_do * Wo_(cell, j, k);
        }
    }
}

static void lstm_zero_grad(LSTMCell* cell) {
    int sz = cell->hidden_size * cell->input_size;
    memset(cell->gWf, 0, sz * sizeof(double));
    memset(cell->gWi, 0, sz * sizeof(double));
    memset(cell->gWc, 0, sz * sizeof(double));
    memset(cell->gWo, 0, sz * sizeof(double));
    memset(cell->gbf, 0, cell->hidden_size * sizeof(double));
    memset(cell->gbi, 0, cell->hidden_size * sizeof(double));
    memset(cell->gbc, 0, cell->hidden_size * sizeof(double));
    memset(cell->gbo, 0, cell->hidden_size * sizeof(double));
}

/* ============================================================================
 * Environment
 * ============================================================================ */
typedef struct {
    int state;
    int target;
    int max_steps;
    int current_step;
    int done;
} Environment;

static void env_init(Environment* env, int tgt, int max_s) {
    env->target = tgt;
    env->max_steps = max_s;
    env->state = 1;
    env->current_step = 0;
    env->done = 0;
}

static int env_reset(Environment* env) {
    env->state = 1;
    env->current_step = 0;
    env->done = 0;
    return env->state;
}

/* action: 0=+1, 1=+2, 2=x2
   returns reward, writes done flag */
static double env_step(Environment* env, int action, int* done_out) {
    env->current_step++;
    if (action == 0) env->state += 1;
    else if (action == 1) env->state += 2;
    else if (action == 2) env->state *= 2;

    double reward = -0.05;
    if (env->state == env->target) {
        reward += 10.0;
        env->done = 1;
    } else if (env->state > env->target || env->current_step >= env->max_steps) {
        reward -= 5.0;
        env->done = 1;
    }
    *done_out = env->done;
    return reward;
}

static void env_encode_state(const Environment* env, double* out) {
    out[0] = (double)env->state / 10.0;
    out[1] = (double)(env->target - env->state) / 10.0;
    out[2] = (double)env->current_step / (double)env->max_steps;
    out[3] = env->state < env->target ? 1.0 : 0.0;
    out[4] = env->state > env->target ? 1.0 : 0.0;
}

static const char* env_action_name(int a) {
    if (a == 0) return "+1";
    if (a == 1) return "+2";
    return "x2";
}

/* ============================================================================
 * Agentic Network
 * ============================================================================ */
typedef struct {
    Linear* fc1;
    LSTMCell* lstm;
    Linear* policy_head;
    Linear* value_head;
    double* h_state;
    double* c_state;
    int lstm_size;
} AgenticNet;

static AgenticNet* net_create(int input_sz, int hidden_sz, int lstm_sz, int action_sz, RNG* rng) {
    AgenticNet* net = (AgenticNet*)calloc(1, sizeof(AgenticNet));
    net->fc1 = linear_create(input_sz, hidden_sz, rng);
    net->lstm = lstm_create(hidden_sz, lstm_sz, rng);
    net->policy_head = linear_create(lstm_sz, action_sz, rng);
    net->value_head = linear_create(lstm_sz, 1, rng);
    net->h_state = (double*)calloc(lstm_sz, sizeof(double));
    net->c_state = (double*)calloc(lstm_sz, sizeof(double));
    net->lstm_size = lstm_sz;
    return net;
}

static void net_free(AgenticNet* net) {
    if (!net) return;
    linear_free(net->fc1);
    lstm_free(net->lstm);
    linear_free(net->policy_head);
    linear_free(net->value_head);
    free(net->h_state);
    free(net->c_state);
    free(net);
}

static void net_reset_lstm(AgenticNet* net) {
    memset(net->h_state, 0, net->lstm_size * sizeof(double));
    memset(net->c_state, 0, net->lstm_size * sizeof(double));
}

static void net_zero_grad(AgenticNet* net) {
    linear_zero_grad(net->fc1);
    lstm_zero_grad(net->lstm);
    linear_zero_grad(net->policy_head);
    linear_zero_grad(net->value_head);
}

/* Forward pass: writes policy_probs (action_sz) and returns value */
static double net_forward(AgenticNet* net, const double* x,
                          double* policy_probs, int action_sz,
                          double* h1_out, int h1_sz) {
    /* fc1 */
    linear_forward(net->fc1, x, h1_out);
    for (int i = 0; i < h1_sz; i++) h1_out[i] = relu(h1_out[i]);

    /* lstm */
    lstm_forward(net->lstm, h1_out, net->h_state, net->c_state, net->h_state);
    memcpy(net->c_state, net->lstm->last_c, net->lstm_size * sizeof(double));

    /* policy head */
    double* p_logits = (double*)calloc(action_sz, sizeof(double));
    linear_forward(net->policy_head, net->h_state, p_logits);
    softmax(p_logits, policy_probs, action_sz);
    free(p_logits);

    /* value head */
    double v_out[1];
    linear_forward(net->value_head, net->h_state, v_out);
    return v_out[0];
}

/* ============================================================================
 * Adam Optimizer
 * ============================================================================ */
typedef struct {
    double lr, beta1, beta2, eps;
    long timestep;
} Adam;

static void adam_init(Adam* opt, double lr, double b1, double b2, double eps) {
    opt->lr = lr; opt->beta1 = b1; opt->beta2 = b2; opt->eps = eps; opt->timestep = 0;
}

static void adam_step_linear(Adam* opt, Linear* L) {
    opt->timestep++;
    double b1t = pow(opt->beta1, (double)opt->timestep);
    double b2t = pow(opt->beta2, (double)opt->timestep);
    int wsize = L->out_features * L->in_features;

    for (int i = 0; i < wsize; i++) {
        double g = L->grad_w[i];
        L->m_w[i] = opt->beta1 * L->m_w[i] + (1.0 - opt->beta1) * g;
        L->v_w[i] = opt->beta2 * L->v_w[i] + (1.0 - opt->beta2) * g * g;
        double mhat = L->m_w[i] / (1.0 - b1t);
        double vhat = L->v_w[i] / (1.0 - b2t);
        L->weight[i] -= opt->lr * mhat / (sqrt(vhat) + opt->eps);
    }
    for (int i = 0; i < L->out_features; i++) {
        double g = L->grad_b[i];
        L->m_b[i] = opt->beta1 * L->m_b[i] + (1.0 - opt->beta1) * g;
        L->v_b[i] = opt->beta2 * L->v_b[i] + (1.0 - opt->beta2) * g * g;
        double mhat = L->m_b[i] / (1.0 - b1t);
        double vhat = L->v_b[i] / (1.0 - b2t);
        L->bias[i] -= opt->lr * mhat / (sqrt(vhat) + opt->eps);
    }
}

static void adam_step_lstm(Adam* opt, LSTMCell* cell) {
    int sz = cell->hidden_size * cell->input_size;
    double b1t = pow(opt->beta1, (double)opt->timestep);
    double b2t = pow(opt->beta2, (double)opt->timestep);

    double* mats[4] = {cell->Wf, cell->Wi, cell->Wc, cell->Wo};
    double* grads[4] = {cell->gWf, cell->gWi, cell->gWc, cell->gWo};
    double* ms[4] = {cell->mWf, cell->mWi, cell->mWc, cell->mWo};
    double* vs[4] = {cell->vWf, cell->vWi, cell->vWc, cell->vWo};

    for (int m = 0; m < 4; m++) {
        for (int i = 0; i < sz; i++) {
            double g = grads[m][i];
            ms[m][i] = opt->beta1 * ms[m][i] + (1.0 - opt->beta1) * g;
            vs[m][i] = opt->beta2 * vs[m][i] + (1.0 - opt->beta2) * g * g;
            double mhat = ms[m][i] / (1.0 - b1t);
            double vhat = vs[m][i] / (1.0 - b2t);
            mats[m][i] -= opt->lr * mhat / (sqrt(vhat) + opt->eps);
        }
    }

    double* bs[4] = {cell->bf, cell->bi, cell->bc, cell->bo};
    double* gbs[4] = {cell->gbf, cell->gbi, cell->gbc, cell->gbo};
    double* mbs[4] = {cell->mbf, cell->mbi, cell->mbc, cell->mbo};
    double* vbs[4] = {cell->vbf, cell->vbi, cell->vbc, cell->vbo};

    for (int m = 0; m < 4; m++) {
        for (int i = 0; i < cell->hidden_size; i++) {
            double g = gbs[m][i];
            mbs[m][i] = opt->beta1 * mbs[m][i] + (1.0 - opt->beta1) * g;
            vbs[m][i] = opt->beta2 * vbs[m][i] + (1.0 - opt->beta2) * g * g;
            double mhat = mbs[m][i] / (1.0 - b1t);
            double vhat = vbs[m][i] / (1.0 - b2t);
            bs[m][i] -= opt->lr * mhat / (sqrt(vhat) + opt->eps);
        }
    }
}

static void adam_step(Adam* opt, AgenticNet* net) {
    adam_step_linear(opt, net->fc1);
    adam_step_lstm(opt, net->lstm);
    adam_step_linear(opt, net->policy_head);
    adam_step_linear(opt, net->value_head);
}

/* ============================================================================
 * Episode Record
 * ============================================================================ */
#define MAX_EPISODE_LEN 64

typedef struct {
    double states[MAX_EPISODE_LEN][5];
    int actions[MAX_EPISODE_LEN];
    double rewards[MAX_EPISODE_LEN];
    double values[MAX_EPISODE_LEN];
    double policy_probs[MAX_EPISODE_LEN][3];
    int length;
    double total_reward;
    int success;
} Episode;

/* ============================================================================
 * A2C Trainer
 * ============================================================================ */
typedef struct {
    AgenticNet* net;
    Adam* opt;
    Environment* env;
    RNG* rng;
    double gamma;
    double value_coef;
    double entropy_coef;
    int max_steps;
} Trainer;

static void trainer_init(Trainer* tr, AgenticNet* net, Adam* opt, Environment* env,
                         RNG* rng, double gamma, double vc, double ec, int ms) {
    tr->net = net; tr->opt = opt; tr->env = env; tr->rng = rng;
    tr->gamma = gamma; tr->value_coef = vc; tr->entropy_coef = ec; tr->max_steps = ms;
}

static void run_episode(Trainer* tr, Episode* ep) {
    env_reset(tr->env);
    net_reset_lstm(tr->net);
    ep->length = 0;
    ep->total_reward = 0.0;
    ep->success = 0;

    int input_sz = 5;
    int hidden_sz = tr->net->fc1->out_features;
    int action_sz = tr->net->policy_head->out_features;

    double* h1_buf = (double*)calloc(hidden_sz, sizeof(double));
    double* pbuf = (double*)calloc(action_sz, sizeof(double));

    for (int step = 0; step < tr->max_steps && ep->length < MAX_EPISODE_LEN; step++) {
        env_encode_state(tr->env, ep->states[ep->length]);
        double value = net_forward(tr->net, ep->states[ep->length], pbuf, action_sz, h1_buf, hidden_sz);

        int action = rng_choice(tr->rng, pbuf, action_sz);

        ep->actions[ep->length] = action;
        ep->values[ep->length] = value;
        memcpy(ep->policy_probs[ep->length], pbuf, action_sz * sizeof(double));

        int done = 0;
        double reward = env_step(tr->env, action, &done);
        ep->rewards[ep->length] = reward;
        ep->total_reward += reward;
        ep->length++;

        if (done) {
            if (tr->env->state == tr->env->target) ep->success = 1;
            break;
        }
    }

    free(h1_buf);
    free(pbuf);
}

static void train_episode(Trainer* tr, Episode* ep) {
    net_zero_grad(tr->net);

    int T = ep->length;
    double returns[MAX_EPISODE_LEN];
    double advantages[MAX_EPISODE_LEN];

    /* Compute discounted returns */
    double G = 0.0;
    for (int t = T - 1; t >= 0; t--) {
        G = ep->rewards[t] + tr->gamma * G;
        returns[t] = G;
        advantages[t] = G - ep->values[t];
    }

    int hidden_sz = tr->net->fc1->out_features;
    int action_sz = tr->net->policy_head->out_features;
    int lstm_sz = tr->net->lstm_size;

    double* h1_buf = (double*)calloc(hidden_sz, sizeof(double));
    double* pbuf = (double*)calloc(action_sz, sizeof(double));
    double* grad_lstm = (double*)calloc(lstm_sz, sizeof(double));
    double* grad_h1 = (double*)calloc(hidden_sz, sizeof(double));
    double* grad_x_dummy = (double*)calloc(tr->net->fc1->in_features, sizeof(double));

    net_reset_lstm(tr->net);

    for (int t = 0; t < T; t++) {
        /* Forward */
        linear_forward(tr->net->fc1, ep->states[t], h1_buf);
        for (int i = 0; i < hidden_sz; i++) h1_buf[i] = relu(h1_buf[i]);

        lstm_forward(tr->net->lstm, h1_buf, tr->net->h_state, tr->net->c_state, tr->net->h_state);
        memcpy(tr->net->c_state, tr->net->lstm->last_c, lstm_sz * sizeof(double));

        double p_logits[3];
        linear_forward(tr->net->policy_head, tr->net->h_state, p_logits);
        softmax(p_logits, pbuf, action_sz);

        double v_out[1];
        linear_forward(tr->net->value_head, tr->net->h_state, v_out);
        double value = v_out[0];

        double advantage = returns[t] - value;

        /* Value gradient */
        double grad_v[1] = {-2.0 * advantage * tr->value_coef};
        double grad_lstm_v[128]; /* max size */
        linear_backward(tr->net->value_head, grad_v, grad_lstm_v);

        /* Policy gradient */
        double grad_p[3];
        double entropy = 0.0;
        for (int a = 0; a < action_sz; a++) {
            if (pbuf[a] > 1e-10) entropy -= pbuf[a] * log(pbuf[a]);
        }
        for (int a = 0; a < action_sz; a++) {
            double indicator = (a == ep->actions[t]) ? 1.0 : 0.0;
            grad_p[a] = -(indicator - pbuf[a]) * advantage;
            if (pbuf[a] > 1e-10)
                grad_p[a] += tr->entropy_coef * (log(pbuf[a]) + 1.0);
        }
        double grad_lstm_p[128];
        linear_backward(tr->net->policy_head, grad_p, grad_lstm_p);

        /* Combine LSTM gradients */
        for (int j = 0; j < lstm_sz; j++) {
            grad_lstm[j] = grad_lstm_p[j] + grad_lstm_v[j];
        }

        /* LSTM backward */
        lstm_backward(tr->net->lstm, grad_lstm, grad_h1);
        for (int i = 0; i < hidden_sz; i++) {
            grad_h1[i] *= relu_deriv(tr->net->fc1->last_pre_act[i]);
        }

        /* fc1 backward */
        linear_backward(tr->net->fc1, grad_h1, grad_x_dummy);
    }

    adam_step(tr->opt, tr->net);

    free(h1_buf);
    free(pbuf);
    free(grad_lstm);
    free(grad_h1);
    free(grad_x_dummy);
}

/* ============================================================================
 * Logger
 * ============================================================================ */
static void log_episode(FILE* fp, int episode, const Episode* ep, double avg100, int interval) {
    if (episode % interval == 0 || episode == 1) {
        printf("[Ep %5d] Reward: %7.2f | Steps: %2d | Success: %s | Avg100: %7.2f\n",
               episode, ep->total_reward, ep->length,
               ep->success ? "YES" : "NO ", avg100);
    }
    fprintf(fp, "%d,%.4f,%d,%d,%.4f\n", episode, ep->total_reward, ep->length, ep->success, avg100);
    if (episode % 10 == 0) fflush(fp);
}

/* ============================================================================
 * Main
 * ============================================================================ */
int main(void) {
    const int INPUT_SIZE = 5;
    const int HIDDEN_SIZE = 64;
    const int LSTM_SIZE = 128;
    const int ACTION_SIZE = 3;
    const double LR = 3e-3;
    const double GAMMA = 0.99;
    const double VALUE_COEF = 0.5;
    const double ENTROPY_COEF = 0.01;
    const int MAX_STEPS = 10;
    const int NUM_EPISODES = 5000;
    const int LOG_INTERVAL = 100;
    const int TARGET = 10;
    const uint64_t SEED = 42;

    RNG rng;
    rng_init(&rng, SEED);

    Environment env;
    env_init(&env, TARGET, MAX_STEPS);

    AgenticNet* net = net_create(INPUT_SIZE, HIDDEN_SIZE, LSTM_SIZE, ACTION_SIZE, &rng);

    Adam optimizer;
    adam_init(&optimizer, LR, 0.9, 0.999, 1e-8);

    Trainer trainer;
    trainer_init(&trainer, net, &optimizer, &env, &rng, GAMMA, VALUE_COEF, ENTROPY_COEF, MAX_STEPS);

    FILE* logfp = fopen("training_log.csv", "w");
    fprintf(logfp, "episode,total_reward,length,success,avg_reward_100\n");

    printf("========================================\n");
    printf("  Agentic RL - C Implementation\n");
    printf("========================================\n");
    printf("Task: Reach %d from 1 using {+1, +2, x2}\n", TARGET);
    printf("Episodes: %d\n", NUM_EPISODES);
    printf("----------------------------------------\n");

    double* reward_history = (double*)calloc(NUM_EPISODES, sizeof(double));
    double best_reward = -1e9;

    for (int ep = 1; ep <= NUM_EPISODES; ep++) {
        Episode episode;
        memset(&episode, 0, sizeof(Episode));
        run_episode(&trainer, &episode);
        train_episode(&trainer, &episode);
        reward_history[ep - 1] = episode.total_reward;

        double avg100 = 0.0;
        int start = (ep > 100) ? ep - 100 : 0;
        int count = ep - start;
        for (int i = start; i < ep; i++) avg100 += reward_history[i];
        avg100 /= count;

        if (episode.total_reward > best_reward) best_reward = episode.total_reward;

        log_episode(logfp, ep, &episode, avg100, LOG_INTERVAL);
    }

    printf("----------------------------------------\n");
    printf("Training complete! Best reward: %.2f\n", best_reward);
    printf("Log saved to: training_log.csv\n");

    /* Final evaluation (greedy) */
    printf("\n=== Final Evaluation (Greedy Policy) ===\n");
    int successes = 0;
    for (int i = 0; i < 10; i++) {
        env_reset(&env);
        net_reset_lstm(net);
        double total_r = 0.0;
        const char* acts[10];
        int nacts = 0;

        double h1_buf[64];
        double pbuf[3];

        for (int step = 0; step < MAX_STEPS; step++) {
            double state_enc[5];
            env_encode_state(&env, state_enc);
            net_forward(net, state_enc, pbuf, ACTION_SIZE, h1_buf, HIDDEN_SIZE);

            int best_a = 0;
            double best_p = pbuf[0];
            for (int a = 1; a < ACTION_SIZE; a++) {
                if (pbuf[a] > best_p) { best_p = pbuf[a]; best_a = a; }
            }
            acts[nacts++] = env_action_name(best_a);

            int done = 0;
            double r = env_step(&env, best_a, &done);
            total_r += r;
            if (done) break;
        }

        int ok = (env.state == env.target);
        if (ok) successes++;
        printf("Eval %2d: ", i + 1);
        for (int j = 0; j < nacts; j++) printf("%s ", acts[j]);
        printf("-> state=%d reward=%.2f %s\n", env.state, total_r, ok ? "[OK]" : "[FAIL]");
    }
    printf("Success rate: %d/10\n", successes);

    fclose(logfp);
    free(reward_history);
    net_free(net);

    return 0;
}
