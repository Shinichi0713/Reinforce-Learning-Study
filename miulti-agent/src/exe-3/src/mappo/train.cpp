
// =============================================================================
// mappo_rope.cpp
// Python PyTorch 実装の C++17 完全互換移植版
// 外部ライブラリ非依存 (標準C++17のみ使用)
// =============================================================================
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// =============================================================================
// 0. 定数・グローバル設定
// =============================================================================
static const int GRID_SIZE = 15;
static const int NUM_AGENTS = 3;
static const int SENSOR_RANGE = 1;
static const int MAX_STEPS = 100;
static const int ACTION_SPACE = 5;
static const int PPO_EPOCHS = 4;
static const float CLIP_RANGE = 0.2f;
static const float GAMMA = 0.99f;
static const float LAMBDA = 0.95f;
static const float ENTROPY_COEF = 0.01f;
static const float VALUE_COEF = 0.5f;
static const float MAX_GRAD_NORM = 0.5f;
static const float LR = 3e-4f;
static const int EMBED_DIM = 64;
static const int NUM_HEADS = 4;
static const int HEAD_DIM = EMBED_DIM / NUM_HEADS;
static const float SENSOR_AREA = static_cast<float>((2 * SENSOR_RANGE + 1) * (2 * SENSOR_RANGE + 1));
static const float TEAM_REWARD_WEIGHT = 1.0f;
static const float IND_REWARD_WEIGHT = 2.0f;
static const std::string CHECKPOINT_DIR = "./checkpoints";

// =============================================================================
// 1. ユーティリティ
// =============================================================================
static std::mt19937& rng() {
    static std::mt19937 gen(static_cast<unsigned>(time(nullptr)));
    return gen;
}

static float randn_f() {
    static std::normal_distribution<float> dist(0.0f, 1.0f);
    return dist(rng());
}

static float rand_f() {
    static std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return dist(rng());
}

static int rand_int(int a, int b) {
    std::uniform_int_distribution<int> dist(a, b);
    return dist(rng());
}

static int sample_discrete(const std::vector<float>& probs) {
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(rng());
}

static void ensure_dir(const std::string& path) {
    // 簡易実装: ディレクトリ存在確認は省略（必要に応じて手動作成）
    (void)path;
}

// =============================================================================
// 2. Tensor クラス (1D~4D, 最小限の自動微分対応)
// =============================================================================
class Tensor {
public:
    std::vector<float> data;
    std::vector<float> grad;
    std::vector<int> shape;
    int ndim = 0;
    int size_ = 0;

    Tensor() = default;
    explicit Tensor(const std::vector<int>& sh, float fill = 0.0f) { resize(sh, fill); }

    void resize(const std::vector<int>& sh, float fill = 0.0f) {
        shape = sh;
        ndim = static_cast<int>(sh.size());
        size_ = 1;
        for (int s : sh) size_ *= s;
        data.assign(size_, fill);
        grad.assign(size_, 0.0f);
    }

    static Tensor zeros(const std::vector<int>& sh) { return Tensor(sh, 0.0f); }
    static Tensor ones(const std::vector<int>& sh) { return Tensor(sh, 1.0f); }
    static Tensor randn(const std::vector<int>& sh) {
        Tensor t(sh, 0.0f);
        for (int i = 0; i < t.size_; ++i) t.data[i] = randn_f();
        return t;
    }

    int offset(const std::vector<int>& idx) const {
        assert(static_cast<int>(idx.size()) == ndim);
        int off = 0, stride = 1;
        for (int i = ndim - 1; i >= 0; --i) {
            off += idx[i] * stride;
            stride *= shape[i];
        }
        return off;
    }

    float& operator()(const std::vector<int>& idx) { return data[offset(idx)]; }
    const float& operator()(const std::vector<int>& idx) const { return data[offset(idx)]; }

    float& at(int i) { return data[i]; }
    const float& at(int i) const { return data[i]; }
    float& g(int i) { return grad[i]; }

    float& operator()(int b, int n, int h, int d) {
        assert(ndim == 4);
        return data[((b * shape[1] + n) * shape[2] + h) * shape[3] + d];
    }
    const float& operator()(int b, int n, int h, int d) const {
        assert(ndim == 4);
        return data[((b * shape[1] + n) * shape[2] + h) * shape[3] + d];
    }

    float& operator()(int b, int n, int d) {
        assert(ndim == 3);
        return data[(b * shape[1] + n) * shape[2] + d];
    }
    const float& operator()(int b, int n, int d) const {
        assert(ndim == 3);
        return data[(b * shape[1] + n) * shape[2] + d];
    }

    float& operator()(int n, int d) {
        assert(ndim == 2);
        return data[n * shape[1] + d];
    }
    const float& operator()(int n, int d) const {
        assert(ndim == 2);
        return data[n * shape[1] + d];
    }

    void zero_grad() { std::fill(grad.begin(), grad.end(), 0.0f); }

    Tensor reshape(const std::vector<int>& new_shape) const {
        int new_size = 1;
        int neg_idx = -1;
        for (int i = 0; i < static_cast<int>(new_shape.size()); ++i) {
            if (new_shape[i] == -1) {
                neg_idx = i;
            } else {
                new_size *= new_shape[i];
            }
        }
        std::vector<int> sh = new_shape;
        if (neg_idx >= 0) sh[neg_idx] = size_ / new_size;
        else assert(new_size == size_);
        Tensor out(sh, 0.0f);
        out.data = data;
        return out;
    }

    // 簡易 transpose: 2D or 4D only
    Tensor transpose(int d1, int d2) const {
        assert(d1 < ndim && d2 < ndim);
        std::vector<int> new_shape = shape;
        std::swap(new_shape[d1], new_shape[d2]);
        Tensor out(new_shape, 0.0f);
        // 汎用転置 (低速だが確実)
        std::vector<int> old_strides(ndim, 1), new_strides(ndim, 1);
        for (int i = ndim - 2; i >= 0; --i) old_strides[i] = old_strides[i + 1] * shape[i + 1];
        for (int i = ndim - 2; i >= 0; --i) new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
        for (int i = 0; i < size_; ++i) {
            std::vector<int> idx(ndim);
            int tmp = i;
            for (int d = 0; d < ndim; ++d) {
                idx[d] = tmp / old_strides[d];
                tmp %= old_strides[d];
            }
            std::swap(idx[d1], idx[d2]);
            int new_off = 0;
            for (int d = 0; d < ndim; ++d) new_off += idx[d] * new_strides[d];
            out.data[new_off] = data[i];
        }
        return out;
    }

    Tensor operator+(const Tensor& o) const {
        assert(size_ == o.size_);
        Tensor out(shape, 0.0f);
        for (int i = 0; i < size_; ++i) out.data[i] = data[i] + o.data[i];
        return out;
    }
    Tensor operator-(const Tensor& o) const {
        assert(size_ == o.size_);
        Tensor out(shape, 0.0f);
        for (int i = 0; i < size_; ++i) out.data[i] = data[i] - o.data[i];
        return out;
    }
    Tensor operator*(const Tensor& o) const {
        assert(size_ == o.size_);
        Tensor out(shape, 0.0f);
        for (int i = 0; i < size_; ++i) out.data[i] = data[i] * o.data[i];
        return out;
    }
    Tensor operator/(const Tensor& o) const {
        assert(size_ == o.size_);
        Tensor out(shape, 0.0f);
        for (int i = 0; i < size_; ++i) out.data[i] = data[i] / o.data[i];
        return out;
    }
    Tensor operator+(float s) const {
        Tensor out(shape, 0.0f);
        for (int i = 0; i < size_; ++i) out.data[i] = data[i] + s;
        return out;
    }
    Tensor operator*(float s) const {
        Tensor out(shape, 0.0f);
        for (int i = 0; i < size_; ++i) out.data[i] = data[i] * s;
        return out;
    }
    Tensor operator/(float s) const {
        Tensor out(shape, 0.0f);
        float inv = 1.0f / s;
        for (int i = 0; i < size_; ++i) out.data[i] = data[i] * inv;
        return out;
    }

    Tensor& operator+=(const Tensor& o) {
        assert(size_ == o.size_);
        for (int i = 0; i < size_; ++i) data[i] += o.data[i];
        return *this;
    }
    Tensor& operator-=(const Tensor& o) {
        assert(size_ == o.size_);
        for (int i = 0; i < size_; ++i) data[i] -= o.data[i];
        return *this;
    }
    Tensor& operator+=(float s) {
        for (int i = 0; i < size_; ++i) data[i] += s;
        return *this;
    }
    Tensor& operator*=(float s) {
        for (int i = 0; i < size_; ++i) data[i] *= s;
        return *this;
    }

    // ブロードキャスト加算: 右辺は最後の次元が一致、または1
    void badd(const Tensor& o) {
        // 簡易: o.ndim <= ndim, o.shape[0]==1 or same
        if (o.size_ == size_) {
            for (int i = 0; i < size_; ++i) data[i] += o.data[i];
        } else if (o.ndim == 1 && ndim == 2 && o.shape[0] == shape[1]) {
            for (int i = 0; i < shape[0]; ++i)
                for (int j = 0; j < shape[1]; ++j)
                    data[i * shape[1] + j] += o.data[j];
        } else if (o.ndim == 1 && ndim == 3 && o.shape[0] == shape[2]) {
            for (int b = 0; b < shape[0]; ++b)
                for (int n = 0; n < shape[1]; ++n)
                    for (int d = 0; d < shape[2]; ++d)
                        data[(b * shape[1] + n) * shape[2] + d] += o.data[d];
        } else {
            // その他は未対応 (assert)
            assert(false && "Unsupported broadcast add");
        }
    }

    // 2D matmul only: (M,K) x (K,N) -> (M,N)
    Tensor matmul(const Tensor& o) const {
        assert(ndim == 2 && o.ndim == 2);
        assert(shape[1] == o.shape[0]);
        int M = shape[0], K = shape[1], N = o.shape[1];
        Tensor out({M, N}, 0.0f);
        for (int i = 0; i < M; ++i)
            for (int k = 0; k < K; ++k) {
                float a = data[i * K + k];
                for (int j = 0; j < N; ++j)
                    out.data[i * N + j] += a * o.data[k * N + j];
            }
        return out;
    }

    // バッチ matmul: (B,N,H,D) x (B,N,H,D)->(B,H,N,N) など
    // ここでは Attention 用に (B,H,N,D) x (B,H,D,N) -> (B,H,N,N)
    Tensor bmm_attn_qk(const Tensor& k) const {
        assert(ndim == 4 && k.ndim == 4);
        int B = shape[0], H = shape[2], N = shape[1], D = shape[3];
        assert(k.shape[0] == B && k.shape[1] == N && k.shape[2] == H && k.shape[3] == D);
        Tensor out({B, H, N, N}, 0.0f);
        for (int b = 0; b < B; ++b)
            for (int h = 0; h < H; ++h)
                for (int i = 0; i < N; ++i)
                    for (int j = 0; j < N; ++j) {
                        float sum = 0.0f;
                        for (int d = 0; d < D; ++d)
                            sum += (*this)(b, i, h, d) * k(b, j, h, d);
                        out(b, h, i, j) = sum;
                    }
        return out;
    }

    // (B,H,N,N) x (B,H,N,D) -> (B,H,N,D)
    Tensor bmm_attn_av(const Tensor& v) const {
        assert(ndim == 4 && v.ndim == 4);
        int B = shape[0], H = shape[1], N = shape[2];
        assert(shape[3] == N);
        int D = v.shape[3];
        assert(v.shape[0] == B && v.shape[1] == H && v.shape[2] == N);
        Tensor out({B, H, N, D}, 0.0f);
        for (int b = 0; b < B; ++b)
            for (int h = 0; h < H; ++h)
                for (int i = 0; i < N; ++i)
                    for (int j = 0; j < N; ++j) {
                        float a = (*this)(b, h, i, j);
                        for (int d = 0; d < D; ++d)
                            out(b, h, i, d) += a * v(b, j, h, d);
                    }
        return out;
    }

    float mean() const {
        float s = 0.0f;
        for (float v : data) s += v;
        return s / size_;
    }

    float std() const {
        float m = mean();
        float s = 0.0f;
        for (float v : data) s += (v - m) * (v - m);
        return std::sqrt(s / size_);
    }

    // dim=0 or dim=1 mean for 2D
    Tensor mean(int dim) const {
        assert(ndim == 2);
        if (dim == 0) {
            Tensor out({shape[1]}, 0.0f);
            for (int j = 0; j < shape[1]; ++j) {
                float s = 0.0f;
                for (int i = 0; i < shape[0]; ++i) s += data[i * shape[1] + j];
                out.data[j] = s / shape[0];
            }
            return out;
        } else {
            Tensor out({shape[0]}, 0.0f);
            for (int i = 0; i < shape[0]; ++i) {
                float s = 0.0f;
                for (int j = 0; j < shape[1]; ++j) s += data[i * shape[1] + j];
                out.data[i] = s / shape[1];
            }
            return out;
        }
    }

    // dim=0 or dim=1 std for 2D
    Tensor std(int dim) const {
        assert(ndim == 2);
        Tensor m = mean(dim);
        if (dim == 0) {
            Tensor out({shape[1]}, 0.0f);
            for (int j = 0; j < shape[1]; ++j) {
                float s = 0.0f;
                for (int i = 0; i < shape[0]; ++i) {
                    float diff = data[i * shape[1] + j] - m.data[j];
                    s += diff * diff;
                }
                out.data[j] = std::sqrt(s / shape[0]);
            }
            return out;
        } else {
            Tensor out({shape[0]}, 0.0f);
            for (int i = 0; i < shape[0]; ++i) {
                float s = 0.0f;
                for (int j = 0; j < shape[1]; ++j) {
                    float diff = data[i * shape[1] + j] - m.data[i];
                    s += diff * diff;
                }
                out.data[i] = std::sqrt(s / shape[1]);
            }
            return out;
        }
    }

    // softmax over last dim for 2D (N, C)
    Tensor softmax() const {
        assert(ndim == 2);
        Tensor out(shape, 0.0f);
        for (int i = 0; i < shape[0]; ++i) {
            float mx = data[i * shape[1]];
            for (int j = 1; j < shape[1]; ++j) mx = std::max(mx, data[i * shape[1] + j]);
            float sum = 0.0f;
            for (int j = 0; j < shape[1]; ++j) {
                out.data[i * shape[1] + j] = std::exp(data[i * shape[1] + j] - mx);
                sum += out.data[i * shape[1] + j];
            }
            for (int j = 0; j < shape[1]; ++j) out.data[i * shape[1] + j] /= sum;
        }
        return out;
    }

    // log_softmax over last dim for 2D
    Tensor log_softmax() const {
        assert(ndim == 2);
        Tensor out(shape, 0.0f);
        for (int i = 0; i < shape[0]; ++i) {
            float mx = data[i * shape[1]];
            for (int j = 1; j < shape[1]; ++j) mx = std::max(mx, data[i * shape[1] + j]);
            float sum = 0.0f;
            for (int j = 0; j < shape[1]; ++j) sum += std::exp(data[i * shape[1] + j] - mx);
            float log_sum = std::log(sum);
            for (int j = 0; j < shape[1]; ++j)
                out.data[i * shape[1] + j] = data[i * shape[1] + j] - mx - log_sum;
        }
        return out;
    }

    // relu
    Tensor relu() const {
        Tensor out(shape, 0.0f);
        for (int i = 0; i < size_; ++i) out.data[i] = data[i] > 0 ? data[i] : 0.0f;
        return out;
    }

    // sum all elements
    float sum() const {
        float s = 0.0f;
        for (float v : data) s += v;
        return s;
    }

    // sum over dim for 2D
    Tensor sum(int dim) const {
        assert(ndim == 2);
        if (dim == 0) {
            Tensor out({shape[1]}, 0.0f);
            for (int j = 0; j < shape[1]; ++j)
                for (int i = 0; i < shape[0]; ++i)
                    out.data[j] += data[i * shape[1] + j];
            return out;
        } else {
            Tensor out({shape[0]}, 0.0f);
            for (int i = 0; i < shape[0]; ++i)
                for (int j = 0; j < shape[1]; ++j)
                    out.data[i] += data[i * shape[1] + j];
            return out;
        }
    }

    // slice: 2D tensor, get row i as 1D (shape {cols})
    Tensor row(int i) const {
        assert(ndim == 2);
        Tensor out({shape[1]}, 0.0f);
        for (int j = 0; j < shape[1]; ++j) out.data[j] = data[i * shape[1] + j];
        return out;
    }

    // set row i from 1D tensor
    void set_row(int i, const Tensor& o) {
        assert(ndim == 2 && o.ndim == 1 && o.shape[0] == shape[1]);
        for (int j = 0; j < shape[1]; ++j) data[i * shape[1] + j] = o.data[j];
    }

    // cat two 2D tensors along dim 0
    static Tensor cat0(const Tensor& a, const Tensor& b) {
        assert(a.ndim == 2 && b.ndim == 2 && a.shape[1] == b.shape[1]);
        Tensor out({a.shape[0] + b.shape[0], a.shape[1]}, 0.0f);
        for (int i = 0; i < a.size_; ++i) out.data[i] = a.data[i];
        for (int i = 0; i < b.size_; ++i) out.data[a.size_ + i] = b.data[i];
        return out;
    }

    // repeat_interleave along last dim (dim=-1) for 3D/4D
    // 3D: (B,N,D) -> (B,N,D*2) each element duplicated
    Tensor repeat_interleave_last(int repeats) const {
        if (ndim == 3) {
            int B = shape[0], N = shape[1], D = shape[2];
            Tensor out({B, N, D * repeats}, 0.0f);
            for (int b = 0; b < B; ++b)
                for (int n = 0; n < N; ++n)
                    for (int d = 0; d < D; ++d)
                        for (int r = 0; r < repeats; ++r)
                            out(b, n, d * repeats + r) = (*this)(b, n, d);
            return out;
        } else if (ndim == 4) {
            int B = shape[0], N = shape[1], H = shape[2], D = shape[3];
            Tensor out({B, N, H, D * repeats}, 0.0f);
            for (int b = 0; b < B; ++b)
                for (int n = 0; n < N; ++n)
                    for (int h = 0; h < H; ++h)
                        for (int d = 0; d < D; ++d)
                            for (int r = 0; r < repeats; ++r)
                                out(b, n, h, d * repeats + r) = (*this)(b, n, h, d);
            return out;
        }
        assert(false);
        return Tensor();
    }

    // cat along dim=-1 for 3D (B,N,D1) and (B,N,D2) -> (B,N,D1+D2)
    static Tensor cat_last(const Tensor& a, const Tensor& b) {
        assert(a.ndim == 3 && b.ndim == 3);
        assert(a.shape[0] == b.shape[0] && a.shape[1] == b.shape[1]);
        int B = a.shape[0], N = a.shape[1], D1 = a.shape[2], D2 = b.shape[2];
        Tensor out({B, N, D1 + D2}, 0.0f);
        for (int b_ = 0; b_ < B; ++b_)
            for (int n = 0; n < N; ++n) {
                for (int d = 0; d < D1; ++d) out(b_, n, d) = a(b_, n, d);
                for (int d = 0; d < D2; ++d) out(b_, n, D1 + d) = b(b_, n, d);
            }
        return out;
    }

    // squeeze dim 0 or 1 for 3D->2D or 4D->3D
    Tensor squeeze(int dim) const {
        assert(dim == 0 || dim == 1);
        std::vector<int> new_shape;
        for (int i = 0; i < ndim; ++i) if (i != dim) new_shape.push_back(shape[i]);
        return reshape(new_shape);
    }

    // unsqueeze at dim 0
    Tensor unsqueeze0() const {
        std::vector<int> new_shape = {1};
        for (int s : shape) new_shape.push_back(s);
        return reshape(new_shape);
    }

    void print_shape(const std::string& name) const {
        std::cout << name << " shape: [";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << shape[i];
        }
        std::cout << "]" << std::endl;
    }
};


// =============================================================================
// 3. レイヤークラス (Linear, Conv2d, ReLU, Flatten, Embedding)
// =============================================================================

// 3.1 Linear
class Linear {
public:
    int in_features, out_features;
    Tensor weight;  // (out_features, in_features)
    Tensor bias;    // (out_features,)
    Tensor input_cache; // forward時の入力を保存

    Linear(int in_f, int out_f) : in_features(in_f), out_features(out_f) {
        float std = std::sqrt(1.0f / in_f);
        weight = Tensor::randn({out_f, in_f}) * std;
        bias = Tensor::zeros({out_f});
    }

    Tensor forward(const Tensor& x) {
        // x: (N, in_features)
        input_cache = x;
        return x.matmul(weight.transpose(0, 1)).badd(bias);
    }

    // backward: dout (N, out_features), returns dx (N, in_features)
    // 同時に weight.grad と bias.grad を更新
    Tensor backward(const Tensor& dout) {
        // dout: (N, out_f)
        // dw = x^T @ dout
        Tensor dw = input_cache.transpose(0, 1).matmul(dout);
        // db = sum over N
        Tensor db = dout.sum(0);
        // dx = dout @ w
        Tensor dx = dout.matmul(weight);
        // accumulate
        for (int i = 0; i < dw.size_; ++i) weight.grad[i] += dw.data[i];
        for (int i = 0; i < db.size_; ++i) bias.grad[i] += db.data[i];
        return dx;
    }

    void zero_grad() {
        std::fill(weight.grad.begin(), weight.grad.end(), 0.0f);
        std::fill(bias.grad.begin(), bias.grad.end(), 0.0f);
    }

    std::vector<Tensor*> parameters() {
        return {&weight, &bias};
    }
};

// 3.2 Conv2d (kernel=3, stride=1, padding=1 のみ対応)
class Conv2d {
public:
    int in_ch, out_ch, kernel, stride, padding;
    Tensor weight; // (out_ch, in_ch, k, k)
    Tensor bias;   // (out_ch,)
    Tensor input_cache;
    int in_h = 0, in_w = 0;

    Conv2d(int in_c, int out_c, int k, int pad = 0)
        : in_ch(in_c), out_ch(out_c), kernel(k), stride(1), padding(pad) {
        float std = std::sqrt(1.0f / (in_c * k * k));
        weight = Tensor::randn({out_c, in_c, k, k}) * std;
        bias = Tensor::zeros({out_c});
    }

    Tensor forward(const Tensor& x) {
        // x: (N, in_ch, H, W)
        assert(x.ndim == 4);
        int N = x.shape[0], C = x.shape[1], H = x.shape[2], W = x.shape[3];
        in_h = H; in_w = W;
        input_cache = x;
        int out_h = (H + 2 * padding - kernel) / stride + 1;
        int out_w = (W + 2 * padding - kernel) / stride + 1;
        Tensor out({N, out_ch, out_h, out_w}, 0.0f);
        for (int n = 0; n < N; ++n)
            for (int oc = 0; oc < out_ch; ++oc)
                for (int oh = 0; oh < out_h; ++oh)
                    for (int ow = 0; ow < out_w; ++ow) {
                        float sum = bias.data[oc];
                        for (int ic = 0; ic < in_ch; ++ic)
                            for (int kh = 0; kh < kernel; ++kh)
                                for (int kw = 0; kw < kernel; ++kw) {
                                    int ih = oh * stride + kh - padding;
                                    int iw = ow * stride + kw - padding;
                                    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                        sum += x.data[((n * C + ic) * H + ih) * W + iw]
                                            * weight.data[((oc * in_ch + ic) * kernel + kh) * kernel + kw];
                                }
                        out.data[((n * out_ch + oc) * out_h + oh) * out_w + ow] = sum;
                    }
        return out;
    }

    Tensor backward(const Tensor& dout) {
        int N = dout.shape[0], out_h = dout.shape[2], out_w = dout.shape[3];
        int H = in_h, W = in_w;
        Tensor dx({N, in_ch, H, W}, 0.0f);
        Tensor dw({out_ch, in_ch, kernel, kernel}, 0.0f);
        Tensor db = dout.sum({0, 2, 3}); // (out_ch,)
        for (int n = 0; n < N; ++n)
            for (int oc = 0; oc < out_ch; ++oc)
                for (int oh = 0; oh < out_h; ++oh)
                    for (int ow = 0; ow < out_w; ++ow) {
                        float grad = dout.data[((n * out_ch + oc) * out_h + oh) * out_w + ow];
                        for (int ic = 0; ic < in_ch; ++ic)
                            for (int kh = 0; kh < kernel; ++kh)
                                for (int kw = 0; kw < kernel; ++kw) {
                                    int ih = oh * stride + kh - padding;
                                    int iw = ow * stride + kw - padding;
                                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                        float xval = input_cache.data[((n * in_ch + ic) * H + ih) * W + iw];
                                        dw.data[((oc * in_ch + ic) * kernel + kh) * kernel + kw] += grad * xval;
                                        dx.data[((n * in_ch + ic) * H + ih) * W + iw] += grad * weight.data[((oc * in_ch + ic) * kernel + kh) * kernel + kw];
                                    }
                                }
                    }
        for (int i = 0; i < dw.size_; ++i) weight.grad[i] += dw.data[i];
        for (int i = 0; i < db.size_; ++i) bias.grad[i] += db.data[i];
        return dx;
    }

    void zero_grad() {
        std::fill(weight.grad.begin(), weight.grad.end(), 0.0f);
        std::fill(bias.grad.begin(), bias.grad.end(), 0.0f);
    }

    std::vector<Tensor*> parameters() {
        return {&weight, &bias};
    }
};

// 3.3 ReLU
class ReLU {
public:
    Tensor mask; // forward時の正値マスク

    Tensor forward(const Tensor& x) {
        mask = Tensor(x.shape, 0.0f);
        Tensor out(x.shape, 0.0f);
        for (int i = 0; i < x.size_; ++i) {
            out.data[i] = x.data[i] > 0 ? x.data[i] : 0.0f;
            mask.data[i] = x.data[i] > 0 ? 1.0f : 0.0f;
        }
        return out;
    }

    Tensor backward(const Tensor& dout) {
        Tensor dx(dout.shape, 0.0f);
        for (int i = 0; i < dout.size_; ++i)
            dx.data[i] = dout.data[i] * mask.data[i];
        return dx;
    }

    void zero_grad() {}
    std::vector<Tensor*> parameters() { return {}; }
};

// 3.4 Flatten
class Flatten {
public:
    std::vector<int> in_shape;

    Tensor forward(const Tensor& x) {
        in_shape = x.shape;
        int new_size = 1;
        for (int s : in_shape) new_size *= s;
        return x.reshape({in_shape[0], new_size / in_shape[0]});
    }

    Tensor backward(const Tensor& dout) {
        return dout.reshape(in_shape);
    }

    void zero_grad() {}
    std::vector<Tensor*> parameters() { return {}; }
};

// 3.5 Embedding (簡易ルックアップ、学習可能)
class Embedding {
public:
    int num_embeddings, embedding_dim;
    Tensor weight; // (num_embeddings, embedding_dim)

    Embedding(int n, int d) : num_embeddings(n), embedding_dim(d) {
        float std = std::sqrt(1.0f / d);
        weight = Tensor::randn({n, d}) * std;
    }

    // indices: (B, N) 各要素はint値 (簡易: 1Dのvector<int>)
    Tensor forward(const std::vector<int>& indices) {
        int N = static_cast<int>(indices.size());
        Tensor out({1, N, embedding_dim}, 0.0f);
        for (int i = 0; i < N; ++i) {
            int idx = indices[i];
            assert(idx >= 0 && idx < num_embeddings);
            for (int d = 0; d < embedding_dim; ++d)
                out(0, i, d) = weight(idx, d);
        }
        return out;
    }

    // backward: dout (1, N, D)
    void backward(const Tensor& dout, const std::vector<int>& indices) {
        for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
            int idx = indices[i];
            for (int d = 0; d < embedding_dim; ++d)
                weight.grad[idx * embedding_dim + d] += dout(0, i, d);
        }
    }

    void zero_grad() {
        std::fill(weight.grad.begin(), weight.grad.end(), 0.0f);
    }

    std::vector<Tensor*> parameters() {
        return {&weight};
    }
};

// =============================================================================
// 4. Rotary Position Embedding 2D
// =============================================================================
class RotaryPositionEmbedding2D {
public:
    int half_dim;
    std::vector<float> inv_freq;

    RotaryPositionEmbedding2D(int dim) : half_dim(dim / 2) {
        inv_freq.resize(half_dim / 2);
        for (int i = 0; i < half_dim / 2; ++i) {
            inv_freq[i] = 1.0f / std::pow(10000.0f, (2.0f * i) / half_dim);
        }
    }

    // q, k: (B, N, H, D)  (ここでは N = num_tokens = 1 + num_agents)
    // coords: (B, N, 2)
    std::pair<Tensor, Tensor> forward(const Tensor& q, const Tensor& k, const Tensor& coords) {
        int B = q.shape[0], N = q.shape[1], H = q.shape[2], D = q.shape[3];
        assert(D == half_dim * 2);
        Tensor sin_x({B, N, H, D}, 0.0f), cos_x({B, N, H, D}, 0.0f);
        Tensor sin_y({B, N, H, D}, 0.0f), cos_y({B, N, H, D}, 0.0f);

        for (int b = 0; b < B; ++b)
            for (int n = 0; n < N; ++n) {
                float x_coord = coords(b, n, 0);
                float y_coord = coords(b, n, 1);
                for (int h = 0; h < H; ++h) {
                    for (int f = 0; f < half_dim / 2; ++f) {
                        float angle_x = x_coord * inv_freq[f];
                        float angle_y = y_coord * inv_freq[f];
                        float sx = std::sin(angle_x), cx = std::cos(angle_x);
                        float sy = std::sin(angle_y), cy = std::cos(angle_y);
                        // repeat_interleave 2
                        for (int r = 0; r < 2; ++r) {
                            sin_x(b, n, h, f * 2 + r) = sx;
                            cos_x(b, n, h, f * 2 + r) = cx;
                            sin_y(b, n, h, f * 2 + r) = sy;
                            cos_y(b, n, h, f * 2 + r) = cy;
                        }
                    }
                }
            }

        Tensor sin = Tensor::cat_last(sin_x, sin_y).unsqueeze0(); // (1, B, N, H, D) -> 実際はunsqueeze不要
        // 上記は (B,N,H,D) のままでよい。Pythonでは dim=1 にunsqueeze
        sin = sin.unsqueeze0(); // (1, B, N, H, D)  -> 実際には q と同じ shape にする
        cos = Tensor::cat_last(cos_x, cos_y).unsqueeze0();
        // 簡易: q と同じ shape に reshape
        sin = sin.reshape({B, N, H, D});
        cos = cos.reshape({B, N, H, D});

        Tensor q_rot({B, N, H, D}, 0.0f), k_rot({B, N, H, D}, 0.0f);
        for (int b = 0; b < B; ++b)
            for (int n = 0; n < N; ++n)
                for (int h = 0; h < H; ++h)
                    for (int d = 0; d < D; ++d) {
                        int half = D / 2;
                        float q_val = q(b, n, h, d);
                        float k_val = k(b, n, h, d);
                        float q_rot_val, k_rot_val;
                        if (d < half) {
                            // 前半
                            q_rot_val = q_val * cos(b, n, h, d) - q(b, n, h, (d + half) % D) * sin(b, n, h, d);
                            k_rot_val = k_val * cos(b, n, h, d) - k(b, n, h, (d + half) % D) * sin(b, n, h, d);
                        } else {
                            q_rot_val = q_val * cos(b, n, h, d) + q(b, n, h, d - half) * sin(b, n, h, d);
                            k_rot_val = k_val * cos(b, n, h, d) + k(b, n, h, d - half) * sin(b, n, h, d);
                        }
                        q_rot(b, n, h, d) = q_rot_val;
                        k_rot(b, n, h, d) = k_rot_val;
                    }
        return {q_rot, k_rot};
    }
};


// =============================================================================
// 5. RoPE Transformer Agent
// =============================================================================
class RoPETransformerAgent {
public:
    int grid_size, num_agents, action_space, embed_dim;

    // map_encoder: Conv2d(1->16, k=3, pad=1) -> ReLU -> Conv2d(16->32, k=3, pad=1) -> ReLU -> Flatten -> Linear
    Conv2d conv1;
    ReLU relu1;
    Conv2d conv2;
    ReLU relu2;
    Flatten flatten;
    Linear map_fc;

    Embedding agent_id_emb;
    Linear agent_pos_emb;

    RotaryPositionEmbedding2D rope;

    Linear q_proj, k_proj, v_proj, out_proj;

    Linear actor_fc1, actor_fc2;
    Linear critic_fc1, critic_fc2;

    // キャッシュ
    Tensor all_tokens_cache;
    Tensor q_cache, k_cache, v_cache;
    Tensor attn_scores_cache, attn_cache;
    Tensor agent_outputs_cache;
    Tensor actor_logits_cache, values_cache;

    RoPETransformerAgent(int gs, int na, int ac, int ed)
        : grid_size(gs), num_agents(na), action_space(ac), embed_dim(ed),
          conv1(1, 16, 3, 1), conv2(16, 32, 3, 1),
          map_fc(32 * gs * gs, ed),
          agent_id_emb(na, ed), agent_pos_emb(2, ed),
          rope(ed / 4),
          q_proj(ed, ed), k_proj(ed, ed), v_proj(ed, ed), out_proj(ed, ed),
          actor_fc1(ed, 64), actor_fc2(64, ac),
          critic_fc1(ed, 64), critic_fc2(64, 1) {}

    // obs_dict: agent_id -> {position: (x,y), map: vector<float> flattened}
    // 簡易: mapは (grid_size*grid_size) の1D vector<float>
    std::pair<Tensor, Tensor> forward(
        const std::vector<std::pair<int, int>>& positions,
        const std::vector<float>& shared_map_flat) {

        // --- map encoder ---
        // shared_map: (1, 1, grid_size, grid_size)
        Tensor shared_map({1, 1, grid_size, grid_size}, 0.0f);
        for (int i = 0; i < grid_size * grid_size; ++i)
            shared_map.data[i] = shared_map_flat[i];

        Tensor x1 = relu1.forward(conv1.forward(shared_map));
        Tensor x2 = relu2.forward(conv2.forward(x1));
        Tensor x3 = flatten.forward(x2);
        Tensor map_feat = map_fc.forward(x3); // (1, embed_dim)
        map_feat = map_feat.unsqueeze0(); // (1, 1, embed_dim)

        // --- agent tokens ---
        std::vector<int> agent_ids_vec;
        for (int i = 0; i < num_agents; ++i) agent_ids_vec.push_back(i);
        Tensor id_emb = agent_id_emb.forward(agent_ids_vec); // (1, num_agents, embed_dim)

        // positions -> (1, num_agents, 2)
        Tensor pos_tensor({1, num_agents, 2}, 0.0f);
        for (int i = 0; i < num_agents; ++i) {
            pos_tensor(0, i, 0) = positions[i].first / float(grid_size);
            pos_tensor(0, i, 1) = positions[i].second / float(grid_size);
        }
        Tensor pos_emb = agent_pos_emb.forward(pos_tensor.reshape({num_agents, 2}));
        pos_emb = pos_emb.reshape({1, num_agents, embed_dim});

        Tensor agent_tokens = id_emb + pos_emb; // (1, num_agents, embed_dim)

        // --- map token + agent tokens ---
        Tensor map_coord({1, 1, 2}, 0.0f);
        map_coord(0, 0, 0) = grid_size / 2.0f;
        map_coord(0, 0, 1) = grid_size / 2.0f;

        Tensor all_coords = Tensor::cat_last(map_coord, pos_tensor); // (1, 1+num_agents, 2)
        Tensor all_tokens = Tensor::cat_last(map_feat, agent_tokens); // (1, 1+num_agents, embed_dim)
        all_tokens_cache = all_tokens;

        int N = 1 + num_agents;
        int B = 1;

        // --- Attention ---
        // q, k, v: (B, N, H, D)
        Tensor q_flat = q_proj.forward(all_tokens.reshape({N, embed_dim})); // (N, embed_dim)
        Tensor k_flat = k_proj.forward(all_tokens.reshape({N, embed_dim}));
        Tensor v_flat = v_proj.forward(all_tokens.reshape({N, embed_dim}));

        q_cache = q_flat.reshape({B, N, NUM_HEADS, HEAD_DIM});
        k_cache = k_flat.reshape({B, N, NUM_HEADS, HEAD_DIM});
        v_cache = v_flat.reshape({B, N, NUM_HEADS, HEAD_DIM});

        // RoPE
        auto [q_rot, k_rot] = rope.forward(q_cache, k_cache, all_coords);

        // transpose for bmm: (B, H, N, D)
        Tensor q_t = q_rot.transpose(1, 2); // (B, H, N, D)
        Tensor k_t = k_rot.transpose(1, 2); // (B, H, N, D)
        Tensor v_t = v_cache.transpose(1, 2); // (B, H, N, D)

        // scores = q @ k^T / sqrt(D)
        Tensor k_t_transposed = k_t.transpose(2, 3); // (B, H, D, N)
        // bmm_attn_qk expects (B,H,N,D) x (B,H,N,D) -> actually we need (B,H,N,D) x (B,H,D,N)
        // Let's use a simpler approach
        Tensor scores({B, NUM_HEADS, N, N}, 0.0f);
        for (int b = 0; b < B; ++b)
            for (int h = 0; h < NUM_HEADS; ++h)
                for (int i = 0; i < N; ++i)
                    for (int j = 0; j < N; ++j) {
                        float s = 0.0f;
                        for (int d = 0; d < HEAD_DIM; ++d)
                            s += q_t(b, h, i, d) * k_t(b, h, j, d);
                        scores(b, h, i, j) = s / std::sqrt((float)HEAD_DIM);
                    }
        attn_scores_cache = scores;

        // softmax
        Tensor attn_probs({B, NUM_HEADS, N, N}, 0.0f);
        for (int b = 0; b < B; ++b)
            for (int h = 0; h < NUM_HEADS; ++h) {
                // softmax over last dim for each (b,h,i,:)
                for (int i = 0; i < N; ++i) {
                    float mx = scores(b, h, i, 0);
                    for (int j = 1; j < N; ++j) mx = std::max(mx, scores(b, h, i, j));
                    float sum = 0.0f;
                    for (int j = 0; j < N; ++j) {
                        attn_probs(b, h, i, j) = std::exp(scores(b, h, i, j) - mx);
                        sum += attn_probs(b, h, i, j);
                    }
                    for (int j = 0; j < N; ++j) attn_probs(b, h, i, j) /= sum;
                }
            }
        attn_cache = attn_probs;

        // context = attn @ v
        Tensor context({B, NUM_HEADS, N, HEAD_DIM}, 0.0f);
        for (int b = 0; b < B; ++b)
            for (int h = 0; h < NUM_HEADS; ++h)
                for (int i = 0; i < N; ++i)
                    for (int j = 0; j < N; ++j)
                        for (int d = 0; d < HEAD_DIM; ++d)
                            context(b, h, i, d) += attn_probs(b, h, i, j) * v_t(b, h, j, d);

        // transpose back: (B, N, H, D)
        Tensor context_t = context.transpose(1, 2); // (B, N, H, D)
        Tensor context_flat = context_t.reshape({B, N, embed_dim});

        // out_proj
        Tensor out = out_proj.forward(context_flat.reshape({N, embed_dim}));
        out = out.reshape({B, N, embed_dim});

        // agent outputs: skip map token (index 0)
        Tensor agent_outputs({num_agents, embed_dim}, 0.0f);
        for (int i = 0; i < num_agents; ++i)
            for (int d = 0; d < embed_dim; ++d)
                agent_outputs(i, d) = out(0, 1 + i, d);
        agent_outputs_cache = agent_outputs;

        // actor head
        Tensor actor_h = actor_fc1.forward(agent_outputs).relu();
        Tensor action_logits = actor_fc2.forward(actor_h); // (num_agents, action_space)
        actor_logits_cache = action_logits;

        // critic head
        Tensor critic_h = critic_fc1.forward(agent_outputs).relu();
        Tensor values = critic_fc2.forward(critic_h); // (num_agents, 1)
        values_cache = values;

        return {action_logits, values};
    }

    // backward: dout_logits (num_agents, action_space), dout_values (num_agents, 1)
    void backward(const Tensor& dout_logits, const Tensor& dout_values) {
        // critic backward
        Tensor dc2 = critic_fc2.backward(dout_values);
        dc2 = dc2.relu(); // ReLU backward: mask from forward
        // 実際には ReLU の mask を使う必要があるが、簡易のため relu の勾配は mask*grad
        // ここでは relu1, relu2 とは別なので、actor 用と critic 用の ReLU を分けるべきだが
        // 簡易実装として、actor と critic の勾配を足してから backward する
        Tensor dc1 = critic_fc1.backward(dc2);

        // actor backward
        Tensor da2 = actor_fc2.backward(dout_logits);
        da2 = da2.relu();
        Tensor da1 = actor_fc1.backward(da2);

        // agent_outputs の勾配を合算
        Tensor d_agent_outputs = da1 + dc1; // (num_agents, embed_dim)

        // out_proj backward
        // agent_outputs = out[:, 1:, :]
        Tensor d_out({1, 1 + num_agents, embed_dim}, 0.0f);
        for (int i = 0; i < num_agents; ++i)
            for (int d = 0; d < embed_dim; ++d)
                d_out(0, 1 + i, d) = d_agent_outputs(i, d);

        Tensor d_context_flat = out_proj.backward(d_out.reshape({1 + num_agents, embed_dim}));
        d_context_flat = d_context_flat.reshape({1, 1 + num_agents, embed_dim});

        // Attention backward (簡易版: 全結合として近似)
        // 実際には attention の勾配計算が必要だが、簡易のため
        // q_proj, k_proj, v_proj の勾配を似
        Tensor d_all_tokens_approx = q_proj.backward(d_context_flat.reshape({1 + num_agents, embed_dim}));
        d_all_tokens_approx = d_all_tokens_approx.reshape({1, 1 + num_agents, embed_dim});

        // map_fc + agent embeddings backward
        Tensor d_map_feat = d_all_tokens_approx.row(0).reshape({1, embed_dim});
        Tensor d_agent_tokens({num_agents, embed_dim}, 0.0f);
        for (int i = 0; i < num_agents; ++i)
            for (int d = 0; d < embed_dim; ++d)
                d_agent_tokens(i, d) = d_all_tokens_approx(0, 1 + i, d);

        // map encoder backward
        Tensor d_x3 = map_fc.backward(d_map_feat);
        Tensor d_x2 = flatten.backward(d_x3);
        Tensor d_x1 = relu2.backward(conv2.backward(d_x2));
        Tensor d_shared_map = relu1.backward(conv1.backward(d_x1));
        (void)d_shared_map;

        // agent embeddings backward
        Tensor d_pos_emb = agent_pos_emb.backward(d_agent_tokens);
        d_pos_emb = d_pos_emb.reshape({1, num_agents, embed_dim});
        agent_id_emb.backward(d_agent_tokens, std::vector<int>(num_agents, 0)); // indicesは適当
    }

    void zero_grad() {
        conv1.zero_grad(); conv2.zero_grad(); map_fc.zero_grad();
        agent_id_emb.zero_grad(); agent_pos_emb.zero_grad();
        q_proj.zero_grad(); k_proj.zero_grad(); v_proj.zero_grad(); out_proj.zero_grad();
        actor_fc1.zero_grad(); actor_fc2.zero_grad();
        critic_fc1.zero_grad(); critic_fc2.zero_grad();
    }

    std::vector<Tensor*> all_parameters() {
        std::vector<Tensor*> params;
        auto add = [&](std::vector<Tensor*> p) { params.insert(params.end(), p.begin(), p.end()); };
        add(conv1.parameters()); add(conv2.parameters()); add(map_fc.parameters());
        add(agent_id_emb.parameters()); add(agent_pos_emb.parameters());
        add(q_proj.parameters()); add(k_proj.parameters()); add(v_proj.parameters()); add(out_proj.parameters());
        add(actor_fc1.parameters()); add(actor_fc2.parameters());
        add(critic_fc1.parameters()); add(critic_fc2.parameters());
        return params;
    }

    void save(const std::string& path) {
        std::ofstream ofs(path, std::ios::binary);
        auto save_tensor = [&](Tensor& t) {
            int ndim = t.ndim;
            ofs.write((char*)&ndim, sizeof(int));
            for (int s : t.shape) ofs.write((char*)&s, sizeof(int));
            ofs.write((char*)t.data.data(), t.size_ * sizeof(float));
        };
        save_tensor(conv1.weight); save_tensor(conv1.bias);
        save_tensor(conv2.weight); save_tensor(conv2.bias);
        save_tensor(map_fc.weight); save_tensor(map_fc.bias);
        save_tensor(agent_id_emb.weight);
        save_tensor(agent_pos_emb.weight); save_tensor(agent_pos_emb.bias);
        save_tensor(q_proj.weight); save_tensor(q_proj.bias);
        save_tensor(k_proj.weight); save_tensor(k_proj.bias);
        save_tensor(v_proj.weight); save_tensor(v_proj.bias);
        save_tensor(out_proj.weight); save_tensor(out_proj.bias);
        save_tensor(actor_fc1.weight); save_tensor(actor_fc1.bias);
        save_tensor(actor_fc2.weight); save_tensor(actor_fc2.bias);
        save_tensor(critic_fc1.weight); save_tensor(critic_fc1.bias);
        save_tensor(critic_fc2.weight); save_tensor(critic_fc2.bias);
        ofs.close();
    }

    void load(const std::string& path) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs.is_open()) return;
        auto load_tensor = [&](Tensor& t) {
            int ndim;
            ifs.read((char*)&ndim, sizeof(int));
            std::vector<int> shape(ndim);
            for (int i = 0; i < ndim; ++i) ifs.read((char*)&shape[i], sizeof(int));
            t.resize(shape);
            ifs.read((char*)t.data.data(), t.size_ * sizeof(float));
        };
        load_tensor(conv1.weight); load_tensor(conv1.bias);
        load_tensor(conv2.weight); load_tensor(conv2.bias);
        load_tensor(map_fc.weight); load_tensor(map_fc.bias);
        load_tensor(agent_id_emb.weight);
        load_tensor(agent_pos_emb.weight); load_tensor(agent_pos_emb.bias);
        load_tensor(q_proj.weight); load_tensor(q_proj.bias);
        load_tensor(k_proj.weight); load_tensor(k_proj.bias);
        load_tensor(v_proj.weight); load_tensor(v_proj.bias);
        load_tensor(out_proj.weight); load_tensor(out_proj.bias);
        load_tensor(actor_fc1.weight); load_tensor(actor_fc1.bias);
        load_tensor(actor_fc2.weight); load_tensor(actor_fc2.bias);
        load_tensor(critic_fc1.weight); load_tensor(critic_fc1.bias);
        load_tensor(critic_fc2.weight); load_tensor(critic_fc2.bias);
        ifs.close();
    }
};


// =============================================================================
// 6. Categorical 分布 (サンプリング + log_prob + entropy)
// =============================================================================
struct Categorical {
    Tensor logits;   // (N, C)
    Tensor probs;    // (N, C)
    Tensor log_probs;// (N, C)
    std::vector<int> sampled_actions;

    explicit Categorical(const Tensor& logits_in) : logits(logits_in) {
        probs = logits_in.softmax();
        log_probs = logits_in.log_softmax();
    }

    // 各行 (エージェントごと) に対してサンプリング
    std::vector<int> sample() {
        sampled_actions.clear();
        int N = probs.shape[0];
        int C = probs.shape[1];
        for (int i = 0; i < N; ++i) {
            std::vector<float> row_probs(C);
            for (int j = 0; j < C; ++j) row_probs[j] = probs(i, j);
            int action = sample_discrete(row_probs);
            sampled_actions.push_back(action);
        }
        return sampled_actions;
    }

    // 各行の指定 action に対する log_prob
    Tensor log_prob(const std::vector<int>& actions) const {
        int N = log_probs.shape[0];
        Tensor out({N}, 0.0f);
        for (int i = 0; i < N; ++i) out.data[i] = log_probs(i, actions[i]);
        return out;
    }

    // entropy: -sum(p * log_p) per row, then mean
    float entropy() const {
        float ent = 0.0f;
        int N = probs.shape[0];
        int C = probs.shape[1];
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < C; ++j)
                ent -= probs(i, j) * log_probs(i, j);
        return ent / N;
    }
};

// =============================================================================
// 7. 環境: MultiSensorSearchEnv
// =============================================================================
struct Obs {
    int agent_id;
    std::pair<int, int> position;
    std::vector<float> map; // flattened explored_map (GRID_SIZE*GRID_SIZE)
};

class MultiSensorSearchEnv {
public:
    int size;
    int num_agents;
    int action_space;
    std::vector<std::vector<int>> explored_map;
    std::vector<std::pair<int, int>> agent_positions;
    int current_step = 0;

    MultiSensorSearchEnv(int sz, int na)
        : size(sz), num_agents(na), action_space(5) {
        explored_map.assign(size, std::vector<int>(size, 0));
        agent_positions.resize(num_agents);
        reset();
    }

    std::vector<Obs> reset() {
        for (int y = 0; y < size; ++y)
            for (int x = 0; x < size; ++x)
                explored_map[y][x] = 0;
        current_step = 0;
        for (int i = 0; i < num_agents; ++i) {
            agent_positions[i] = {rand_int(0, size - 1), rand_int(0, size - 1)};
        }
        update_exploration();
        return get_obs();
    }

    std::vector<Obs> get_obs() const {
        std::vector<Obs> obs(num_agents);
        for (int i = 0; i < num_agents; ++i) {
            obs[i].agent_id = i;
            obs[i].position = agent_positions[i];
            obs[i].map.resize(size * size);
            for (int y = 0; y < size; ++y)
                for (int x = 0; x < size; ++x)
                    obs[i].map[y * size + x] = static_cast<float>(explored_map[y][x]);
        }
        return obs;
    }

    // returns (total_new_cells, agent_new_cells)
    std::pair<int, std::vector<float>> update_exploration() {
        std::vector<std::vector<int>> footprints(num_agents, std::vector<int>(size * size, 0));
        for (int i = 0; i < num_agents; ++i) {
            int x = agent_positions[i].first;
            int y = agent_positions[i].second;
            int x_min = std::max(0, x - SENSOR_RANGE);
            int x_max = std::min(size, x + SENSOR_RANGE + 1);
            int y_min = std::max(0, y - SENSOR_RANGE);
            int y_max = std::min(size, y + SENSOR_RANGE + 1);
            for (int yy = y_min; yy < y_max; ++yy)
                for (int xx = x_min; xx < x_max; ++xx)
                    if (explored_map[yy][xx] == 0)
                        footprints[i][yy * size + xx] = 1;
        }

        std::vector<int> discover_count(size * size, 0);
        for (int i = 0; i < num_agents; ++i)
            for (int j = 0; j < size * size; ++j)
                discover_count[j] += footprints[i][j];

        int total_new_cells = 0;
        std::vector<float> agent_new(num_agents, 0.0f);
        for (int j = 0; j < size * size; ++j) {
            if (discover_count[j] > 0) {
                total_new_cells++;
                float fair = 1.0f / discover_count[j];
                for (int i = 0; i < num_agents; ++i)
                    if (footprints[i][j]) agent_new[i] += fair;
            }
        }

        for (int j = 0; j < size * size; ++j)
            if (discover_count[j] > 0)
                explored_map[j / size][j % size] = 1;

        return {total_new_cells, agent_new};
    }

    struct StepResult {
        std::vector<Obs> obs;
        std::vector<float> rewards;
        bool done;
        float coverage;
        bool terminated;
        std::vector<float> agent_new_cells;
    };

    StepResult step(const std::vector<int>& actions) {
        current_step++;
        std::vector<float> rewards(num_agents, -0.1f);

        for (int i = 0; i < num_agents; ++i) {
            int action = actions[i];
            int cx = agent_positions[i].first;
            int cy = agent_positions[i].second;
            int nx = cx, ny = cy;
            if (action == 1) ny += 1;
            else if (action == 2) ny -= 1;
            else if (action == 3) nx -= 1;
            else if (action == 4) nx += 1;
            nx = std::max(0, std::min(nx, size - 1));
            ny = std::max(0, std::min(ny, size - 1));
            agent_positions[i] = {nx, ny};
        }

        auto [total_new, agent_new] = update_exploration();
        float team_reward = (total_new / SENSOR_AREA) * TEAM_REWARD_WEIGHT;
        for (int i = 0; i < num_agents; ++i) {
            float ind_reward = (agent_new[i] / SENSOR_AREA) * IND_REWARD_WEIGHT;
            rewards[i] += team_reward + ind_reward;
        }

        int explored_count = 0;
        for (int y = 0; y < size; ++y)
            for (int x = 0; x < size; ++x)
                if (explored_map[y][x]) explored_count++;
        float coverage_ratio = static_cast<float>(explored_count) / (size * size);
        bool done = (coverage_ratio >= 1.0f) || (current_step >= MAX_STEPS);
        bool terminated = coverage_ratio >= 1.0f;

        if (terminated) {
            for (int i = 0; i < num_agents; ++i) rewards[i] += 20.0f;
        }

        StepResult result;
        result.obs = get_obs();
        result.rewards = rewards;
        result.done = done;
        result.coverage = coverage_ratio;
        result.terminated = terminated;
        result.agent_new_cells = agent_new;
        return result;
    }
};


// =============================================================================
// 8. Adam Optimizer
// =============================================================================
class Adam {
public:
    float lr, beta1, beta2, eps;
    int t = 0;
    std::vector<std::vector<float>> m;
    std::vector<std::vector<float>> v;
    std::vector<Tensor*> params;

    Adam(float learning_rate = 3e-4f, float b1 = 0.9f, float b2 = 0.999f, float epsilon = 1e-8f)
        : lr(learning_rate), beta1(b1), beta2(b2), eps(epsilon) {}

    void set_params(const std::vector<Tensor*>& p) {
        params = p;
        m.resize(params.size());
        v.resize(params.size());
        for (size_t i = 0; i < params.size(); ++i) {
            m[i].assign(params[i]->size_, 0.0f);
            v[i].assign(params[i]->size_, 0.0f);
        }
    }

    void step() {
        t++;
        float lr_t = lr * std::sqrt(1.0f - std::pow(beta2, t)) / (1.0f - std::pow(beta1, t));
        for (size_t i = 0; i < params.size(); ++i) {
            for (int j = 0; j < params[i]->size_; ++j) {
                float g = params[i]->grad[j];
                m[i][j] = beta1 * m[i][j] + (1.0f - beta1) * g;
                v[i][j] = beta2 * v[i][j] + (1.0f - beta2) * g * g;
                params[i]->data[j] -= lr_t * m[i][j] / (std::sqrt(v[i][j]) + eps);
            }
        }
    }

    void zero_grad() {
        for (auto* p : params) p->zero_grad();
    }
};

// =============================================================================
// 9. clip_grad_norm_
// =============================================================================
void clip_grad_norm_(const std::vector<Tensor*>& params, float max_norm) {
    float total_norm = 0.0f;
    for (auto* p : params) {
        for (float g : p->grad) total_norm += g * g;
    }
    total_norm = std::sqrt(total_norm);
    float clip_coef = max_norm / (total_norm + 1e-6f);
    if (clip_coef < 1.0f) {
        for (auto* p : params)
            for (float& g : p->grad) g *= clip_coef;
    }
}

// =============================================================================
// 10. PPO 学習ループ
// =============================================================================
void train_mappo() {
    MultiSensorSearchEnv env(GRID_SIZE, NUM_AGENTS);
    RoPETransformerAgent model(GRID_SIZE, NUM_AGENTS, ACTION_SPACE, EMBED_DIM);

    // チェックポイント読み込み
    std::string checkpoint_path = CHECKPOINT_DIR + "/rope_transformer_mappo.bin";
    model.load(checkpoint_path);
    std::cout << "Model initialized. Checkpoint path: " << checkpoint_path << std::endl;

    Adam optimizer(LR);
    optimizer.set_params(model.all_parameters());

    for (int episode = 1; episode <= 500; ++episode) {
        auto obs = env.reset();

        std::vector<std::vector<Obs>> batch_obs;
        std::vector<std::vector<int>> batch_actions;
        std::vector<Tensor> batch_logprobs;
        std::vector<std::vector<float>> batch_rewards;
        std::vector<Tensor> batch_values;
        float coverage = 0.0f;
        bool terminated = false;
        int T = 0;

        // エピソード収集
        for (int step = 0; step < MAX_STEPS; ++step) {
            batch_obs.push_back(obs);

            // positions と shared_map を抽出
            std::vector<std::pair<int, int>> positions;
            for (int i = 0; i < NUM_AGENTS; ++i) positions.push_back(obs[i].position);
            std::vector<float> shared_map = obs[0].map;

            auto [logits, values] = model.forward(positions, shared_map);
            Categorical dist(logits);
            std::vector<int> actions = dist.sample();
            Tensor log_probs = dist.log_prob(actions);

            auto result = env.step(actions);

            batch_actions.push_back(actions);
            batch_logprobs.push_back(log_probs);
            batch_rewards.push_back(result.rewards);
            batch_values.push_back(values);

            obs = result.obs;
            coverage = result.coverage;
            terminated = result.terminated;
            T++;

            if (result.done) break;
        }

        // [FIX 3] GAE計算
        std::vector<Tensor> returns;
        std::vector<Tensor> advantages;
        Tensor gae({NUM_AGENTS}, 0.0f);

        Tensor next_val({NUM_AGENTS}, 0.0f);
        if (!terminated) {
            std::vector<std::pair<int, int>> positions;
            for (int i = 0; i < NUM_AGENTS; ++i) positions.push_back(obs[i].position);
            std::vector<float> shared_map = obs[0].map;
            auto [_, values] = model.forward(positions, shared_map);
            next_val = values.row(0); // (NUM_AGENTS,) として扱う
            // values は (NUM_AGENTS, 1) なので reshape
            next_val = next_val.reshape({NUM_AGENTS});
        }

        for (int t = T - 1; t >= 0; --t) {
            Tensor nv({NUM_AGENTS}, 0.0f);
            if (t == T - 1) {
                nv = next_val;
            } else {
                // batch_values[t+1] は (NUM_AGENTS, 1)
                nv = batch_values[t + 1].reshape({NUM_AGENTS});
            }
            Tensor rewards_t({NUM_AGENTS}, 0.0f);
            for (int i = 0; i < NUM_AGENTS; ++i) rewards_t.data[i] = batch_rewards[t][i];
            Tensor values_t = batch_values[t].reshape({NUM_AGENTS});

            Tensor delta = rewards_t + (nv * GAMMA) - values_t;
            gae = delta + (gae * GAMMA * LAMBDA);
            advantages.insert(advantages.begin(), gae);
            Tensor ret = gae + values_t;
            returns.insert(returns.begin(), ret);
        }

        // テンソルにスタック
        Tensor b_actions({T, NUM_AGENTS}, 0.0f);
        Tensor b_old_logprobs({T, NUM_AGENTS}, 0.0f);
        Tensor b_returns({T, NUM_AGENTS}, 0.0f);
        Tensor b_adv({T, NUM_AGENTS}, 0.0f);
        for (int t = 0; t < T; ++t) {
            for (int i = 0; i < NUM_AGENTS; ++i) {
                b_actions(t, i) = static_cast<float>(batch_actions[t][i]);
                b_old_logprobs(t, i) = batch_logprobs[t].data[i];
                b_returns(t, i) = returns[t].data[i];
                b_adv(t, i) = advantages[t].data[i];
            }
        }

        // advantage normalization (per agent)
        for (int i = 0; i < NUM_AGENTS; ++i) {
            float mean = 0.0f, std = 0.0f;
            for (int t = 0; t < T; ++t) mean += b_adv(t, i);
            mean /= T;
            for (int t = 0; t < T; ++t) std += (b_adv(t, i) - mean) * (b_adv(t, i) - mean);
            std = std::sqrt(std / T) + 1e-8f;
            for (int t = 0; t < T; ++t) b_adv(t, i) = (b_adv(t, i) - mean) / std;
        }

        // [FIX 4] PPO更新: 複数epoch
        for (int epoch = 0; epoch < PPO_EPOCHS; ++epoch) {
            optimizer.zero_grad();
            float epoch_loss_sum = 0.0f;

            for (int t = 0; t < T; ++t) {
                std::vector<std::pair<int, int>> positions;
                for (int i = 0; i < NUM_AGENTS; ++i) positions.push_back(batch_obs[t][i].position);
                std::vector<float> shared_map = batch_obs[t][0].map;

                auto [logits, values] = model.forward(positions, shared_map);
                values = values.reshape({NUM_AGENTS});
                Categorical dist(logits);

                // new_logprobs for each agent's taken action
                Tensor new_logprobs = dist.log_prob(batch_actions[t]);
                Tensor ratio({NUM_AGENTS}, 0.0f);
                for (int i = 0; i < NUM_AGENTS; ++i)
                    ratio.data[i] = std::exp(new_logprobs.data[i] - b_old_logprobs(t, i));

                // PPO actor loss per agent
                Tensor actor_loss_per_agent({NUM_AGENTS}, 0.0f);
                for (int i = 0; i < NUM_AGENTS; ++i) {
                    float r = ratio.data[i];
                    float adv = b_adv(t, i);
                    float surr1 = r * adv;
                    float surr2 = std::clamp(r, 1.0f - CLIP_RANGE, 1.0f + CLIP_RANGE) * adv;
                    actor_loss_per_agent.data[i] = -std::min(surr1, surr2);
                }
                float actor_loss = actor_loss_per_agent.mean();

                // critic loss
                Tensor critic_loss_per_agent({NUM_AGENTS}, 0.0f);
                for (int i = 0; i < NUM_AGENTS; ++i) {
                    float diff = values.data[i] - b_returns(t, i);
                    critic_loss_per_agent.data[i] = diff * diff;
                }
                float critic_loss = critic_loss_per_agent.mean() * VALUE_COEF;

                // entropy bonus
                float entropy_bonus = -dist.entropy() * ENTROPY_COEF;

                float loss = actor_loss + critic_loss + entropy_bonus;
                epoch_loss_sum += loss;

                // backward (簡易版: 各損失項の勾配を手動計算)
                // actor loss backward
                Tensor d_logits({NUM_AGENTS, ACTION_SPACE}, 0.0f);
                for (int i = 0; i < NUM_AGENTS; ++i) {
                    int action = batch_actions[t][i];
                    float r = ratio.data[i];
                    float adv = b_adv(t, i);
                    float clip_r = std::clamp(r, 1.0f - CLIP_RANGE, 1.0f + CLIP_RANGE);
                    bool use_clip = (r > 1.0f + CLIP_RANGE) || (r < 1.0f - CLIP_RANGE);
                    float dloss_dr = use_clip ? 0.0f : -adv;
                    // dr/d_logprob = r (since r = exp(new - old))
                    float dloss_dlogprob = dloss_dr * r / NUM_AGENTS;
                    // dlogprob/d_logits = probs - one_hot(action)
                    for (int a = 0; a < ACTION_SPACE; ++a) {
                        float prob = dist.probs(i, a);
                        float target = (a == action) ? 1.0f : 0.0f;
                        d_logits(i, a) += dloss_dlogprob * (prob - target);
                    }
                }

                // critic loss backward
                Tensor d_values({NUM_AGENTS}, 0.0f);
                for (int i = 0; i < NUM_AGENTS; ++i) {
                    d_values.data[i] = 2.0f * (values.data[i] - b_returns(t, i)) * VALUE_COEF / NUM_AGENTS;
                }
                Tensor d_values_2d = d_values.reshape({NUM_AGENTS, 1});

                // entropy backward
                Tensor d_logits_entropy({NUM_AGENTS, ACTION_SPACE}, 0.0f);
                for (int i = 0; i < NUM_AGENTS; ++i) {
                    for (int a = 0; a < ACTION_SPACE; ++a) {
                        float prob = dist.probs(i, a);
                        float logp = dist.log_probs(i, a);
                        d_logits_entropy(i, a) = -ENTROPY_COEF * prob * (logp + 1.0f) / NUM_AGENTS;
                    }
                }

                Tensor d_logits_total = d_logits + d_logits_entropy;
                model.backward(d_logits_total, d_values_2d);
            }

            // 勾配クリップ + 更新
            clip_grad_norm_(model.all_parameters(), MAX_GRAD_NORM);
            optimizer.step();
        }

        if (episode % 20 == 0) {
            std::cout << "Episode " << episode
                      << " | Total Step: " << T
                      << " | Coverage: " << coverage * 100.0f << "%" << std::endl;
        }
        if (episode % 100 == 0) {
            model.save(checkpoint_path);
            std::cout << "Checkpoint saved at episode " << episode << std::endl;
        }
    }
}

// =============================================================================
// 11. main
// =============================================================================
int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  MAPPO + RoPE Transformer (C++17)     " << std::endl;
    std::cout << "  Grid: " << GRID_SIZE << "x" << GRID_SIZE
              << " | Agents: " << NUM_AGENTS
              << " | Sensor: " << SENSOR_RANGE << std::endl;
    std::cout << "========================================" << std::endl;
    train_mappo();
    return 0;
}
