// GoGUI_v3.cpp
// Visual Studio 2022 / Win32 API / Unicode / Windows Subsystem
//
// v3 の変更点
//   1. Python (PPO) で学習した GoNet を ONNX Runtime で読み込み、AI として使用
//      (モデルが無い/読めない場合は従来の RandomAgent にフォールバック)
//   2. 終局判定を追加し、勝敗が決まった時点でゲームを終了して結果パネルを表示
//   3. 石やコーナー点が正しく描画されない問題の修正 (ブラシが選択されていなかった)
//
// ビルド手順
//   - NuGet で "Microsoft.ML.OnnxRuntime" をプロジェクトに追加 (x64 / C++17 推奨)
//     ヘッダ・lib・onnxruntime.dll は NuGet が自動で設定/コピーします
//   - Python 側の学習で出力された go_agent_trained.onnx を exe と同じフォルダに置く
//     (VS からデバッグ実行する場合は x64\Debug など exe の出力先)

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// ===================== Go Engine =====================
enum class Stone { EMPTY = 0, BLACK = 1, WHITE = 2 };

inline Stone opponent_of(Stone c) {
    return (c == Stone::BLACK) ? Stone::WHITE : Stone::BLACK;
}

class GoBoard {
public:
    static constexpr int SIZE = 9;
    static constexpr float KOMI = 6.5f;
    static constexpr int PASS_MOVE = -1;

private:
    std::vector<Stone> board;
    int ko_pos;
    int last_captured_count;

    mutable std::vector<int> parent;
    mutable std::vector<int> liberties;
    mutable std::vector<int> group_size;

    int dx[4] = { 1, -1, 0, 0 };
    int dy[4] = { 0, 0, 1, -1 };

public:
    GoBoard() : board(SIZE* SIZE, Stone::EMPTY), ko_pos(-1), last_captured_count(0) {
        reset_union_find();
    }

    Stone get(int x, int y) const {
        if (x < 0 || x >= SIZE || y < 0 || y >= SIZE) return Stone::EMPTY;
        return board[index(x, y)];
    }

    int get_ko_pos() const { return ko_pos; }
    int get_last_captured_count() const { return last_captured_count; }   // v3: 捕獲数の集計用

    bool is_legal(int x, int y, Stone color) const {
        if (x == PASS_MOVE) return true;
        if (x < 0 || x >= SIZE || y < 0 || y >= SIZE) return false;
        int idx = index(x, y);
        if (board[idx] != Stone::EMPTY) return false;
        if (idx == ko_pos) return false;

        GoBoard temp = *this;
        int captured = 0, dummy_ko = -1;
        bool ok = temp.place_stone_raw(x, y, color, captured, dummy_ko);
        return ok;
    }

    std::vector<std::pair<int, int>> get_legal_moves(Stone color) const {
        std::vector<std::pair<int, int>> moves;
        moves.reserve(SIZE * SIZE);
        for (int y = 0; y < SIZE; ++y)
            for (int x = 0; x < SIZE; ++x)
                if (is_legal(x, y, color))
                    moves.emplace_back(x, y);
        return moves;
    }

    bool play_move(int x, int y, Stone color) {
        if (x == PASS_MOVE) {
            ko_pos = -1;
            last_captured_count = 0;
            return true;
        }
        int captured = 0, new_ko = -1;
        bool ok = place_stone_raw(x, y, color, captured, new_ko);
        if (!ok) return false;
        ko_pos = new_ko;
        last_captured_count = captured;
        return true;
    }

    struct Score {
        float black_score;
        float white_score;
        int black_stones;
        int white_stones;
        float black_territory;
        float white_territory;
    };

    Score calculate_score() const {
        Score s = {};
        std::vector<bool> visited(SIZE * SIZE, false);

        for (int i = 0; i < SIZE * SIZE; ++i) {
            if (board[i] == Stone::BLACK) s.black_stones++;
            else if (board[i] == Stone::WHITE) s.white_stones++;
        }

        for (int y = 0; y < SIZE; ++y) {
            for (int x = 0; x < SIZE; ++x) {
                int idx = index(x, y);
                if (board[idx] != Stone::EMPTY || visited[idx]) continue;

                std::vector<int> region;
                std::queue<int> q;
                q.push(idx);
                visited[idx] = true;
                bool adj_black = false, adj_white = false;

                while (!q.empty()) {
                    int cur = q.front(); q.pop();
                    region.push_back(cur);
                    int cx = cur % SIZE, cy = cur / SIZE;
                    for (int d = 0; d < 4; ++d) {
                        int nx = cx + dx[d], ny = cy + dy[d];
                        if (!in_bounds(nx, ny)) continue;
                        int nidx = index(nx, ny);
                        if (board[nidx] == Stone::EMPTY && !visited[nidx]) {
                            visited[nidx] = true;
                            q.push(nidx);
                        }
                        else if (board[nidx] == Stone::BLACK) adj_black = true;
                        else if (board[nidx] == Stone::WHITE) adj_white = true;
                    }
                }

                if (adj_black && !adj_white) s.black_territory += static_cast<float>(region.size());
                else if (adj_white && !adj_black) s.white_territory += static_cast<float>(region.size());
            }
        }

        s.black_score = s.black_territory + static_cast<float>(s.black_stones);
        s.white_score = s.white_territory + static_cast<float>(s.white_stones) + KOMI;
        return s;
    }

private:
    inline int index(int x, int y) const { return y * SIZE + x; }
    inline bool in_bounds(int x, int y) const { return x >= 0 && x < SIZE && y >= 0 && y < SIZE; }

    void reset_union_find() const {
        int n = SIZE * SIZE;
        parent.resize(n);
        liberties.resize(n);
        group_size.resize(n);
        for (int i = 0; i < n; ++i) {
            parent[i] = i;
            liberties[i] = 0;
            group_size[i] = (board[i] == Stone::EMPTY) ? 0 : 1;
        }

        for (int y = 0; y < SIZE; ++y)
            for (int x = 0; x < SIZE; ++x) {
                int idx = index(x, y);
                if (board[idx] == Stone::EMPTY) continue;
                for (int d = 0; d < 2; ++d) {
                    int nx = x + dx[d], ny = y + dy[d];
                    if (!in_bounds(nx, ny)) continue;
                    int nidx = index(nx, ny);
                    if (board[nidx] == board[idx]) unite(idx, nidx);
                }
            }

        for (int y = 0; y < SIZE; ++y)
            for (int x = 0; x < SIZE; ++x) {
                int idx = index(x, y);
                if (board[idx] != Stone::EMPTY) continue;
                std::set<int> roots;
                for (int d = 0; d < 4; ++d) {
                    int nx = x + dx[d], ny = y + dy[d];
                    if (!in_bounds(nx, ny)) continue;
                    int nidx = index(nx, ny);
                    if (board[nidx] != Stone::EMPTY) roots.insert(find_root(nidx));
                }
                for (int r : roots) liberties[r]++;
            }
    }

    int find_root(int x) const {
        if (parent[x] == x) return x;
        parent[x] = find_root(parent[x]);
        return parent[x];
    }

    void unite(int a, int b) const {
        a = find_root(a); b = find_root(b);
        if (a == b) return;
        if (a > b) std::swap(a, b);
        parent[b] = a;
        liberties[a] += liberties[b];
        group_size[a] += group_size[b];
    }

    bool place_stone_raw(int x, int y, Stone color, int& out_captured, int& out_new_ko) {
        out_captured = 0;
        out_new_ko = -1;

        int idx = index(x, y);
        board[idx] = color;
        Stone opp = opponent_of(color);

        std::vector<Stone> backup_board = board;

        reset_union_find();
        std::vector<int> captured_indices;

        for (int d = 0; d < 4; ++d) {
            int nx = x + dx[d], ny = y + dy[d];
            if (!in_bounds(nx, ny)) continue;
            int nidx = index(nx, ny);
            if (board[nidx] != opp) continue;

            int root = find_root(nidx);
            if (liberties[root] == 0) {
                for (int i = 0; i < SIZE * SIZE; ++i)
                    if (find_root(i) == root) {
                        captured_indices.push_back(i);
                        board[i] = Stone::EMPTY;
                    }
            }
        }

        out_captured = static_cast<int>(captured_indices.size());

        reset_union_find();
        int my_root = find_root(idx);
        if (liberties[my_root] == 0) {
            board = backup_board;
            return false;
        }

        if (out_captured == 1) {
            out_new_ko = captured_indices[0];
        }

        return true;
    }
};

class RandomAgent {
    std::mt19937 rng;
public:
    explicit RandomAgent(unsigned seed = 42) : rng(seed) {}

    std::pair<int, int> select_move(const GoBoard& board, Stone color) {
        auto legals = board.get_legal_moves(color);
        if (legals.empty()) return { GoBoard::PASS_MOVE, GoBoard::PASS_MOVE };
        std::uniform_int_distribution<size_t> dist(0, legals.size() - 1);
        return legals[dist(rng)];
    }
};

// ===================== NN Helpers (Pythonの学習環境と同一仕様) =====================
// Python: GoEnv._get_obs() と同じ 3ch の手番視点観測
//   ch0: 手番側の石 / ch1: 相手の石 / ch2: 手番が黒なら全面 1.0、白なら 0.0
// 盤面インデックスは Python の action = r * 9 + c に対応 (r = y, c = x)
constexpr int NN_CHANNELS = 3;
constexpr int NN_CELLS = GoBoard::SIZE * GoBoard::SIZE;

inline void BuildObservation(const GoBoard& board, Stone me,
    std::array<float, NN_CHANNELS* NN_CELLS>& obs) {
    obs.fill(0.0f);
    Stone opp = opponent_of(me);
    for (int y = 0; y < GoBoard::SIZE; ++y) {
        for (int x = 0; x < GoBoard::SIZE; ++x) {
            int idx = y * GoBoard::SIZE + x;
            Stone s = board.get(x, y);
            if (s == me) obs[0 * NN_CELLS + idx] = 1.0f;
            else if (s == opp) obs[1 * NN_CELLS + idx] = 1.0f;
        }
    }
    float turn_plane = (me == Stone::BLACK) ? 1.0f : 0.0f;
    for (int i = 0; i < NN_CELLS; ++i) obs[2 * NN_CELLS + i] = turn_plane;
}

// 合法手のみを対象に、温度付きソフトマックスでサンプリング (temperature <= 0 で argmax)
// 戻り値: 選んだ action (0..80)。合法手が無ければ -1
inline int SelectAction(const float* logits, const std::vector<char>& legal,
    float temperature, std::mt19937& rng) {
    int best = -1;
    float best_v = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < NN_CELLS; ++i) {
        if (legal[i] && logits[i] > best_v) { best_v = logits[i]; best = i; }
    }
    if (best < 0) return -1;
    if (temperature <= 0.0f) return best;

    std::vector<double> w(NN_CELLS, 0.0);
    for (int i = 0; i < NN_CELLS; ++i) {
        if (legal[i]) w[i] = std::exp(static_cast<double>(logits[i] - best_v) / temperature);
    }
    std::discrete_distribution<int> dist(w.begin(), w.end());
    return dist(rng);
}

// ===================== ONNX Agent =====================
#include <onnxruntime_cxx_api.h>

static std::wstring Widen(const char* s) {
    int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0);
    if (n <= 1) return L"";
    std::wstring w(static_cast<size_t>(n - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s, -1, &w[0], n);
    return w;
}

class NNAgent {
    Ort::Env env_{ ORT_LOGGING_LEVEL_WARNING, "GoAgent" };
    Ort::Session session_{ nullptr };
    std::mt19937 rng_{ std::random_device{}() };
    bool loaded_ = false;

public:
    bool loaded() const { return loaded_; }

    bool load(const std::wstring& path, std::wstring& error) {
        try {
            Ort::SessionOptions opts;
            opts.SetIntraOpNumThreads(1);
            opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            session_ = Ort::Session(env_, path.c_str(), opts);   // Windows では wchar_t パス
            loaded_ = true;
            return true;
        }
        catch (const std::exception& e) {
            error = Widen(e.what());
            loaded_ = false;
            return false;
        }
    }

    // legals は board.get_legal_moves(color) の結果 (ko / 自殺手を除外済み)
    std::pair<int, int> select_move(const GoBoard& board, Stone color,
        const std::vector<std::pair<int, int>>& legals,
        float temperature) {
        const std::pair<int, int> pass_move(GoBoard::PASS_MOVE, GoBoard::PASS_MOVE);
        if (!loaded_ || legals.empty()) return pass_move;

        std::vector<char> legal_mask(NN_CELLS, 0);
        for (const auto& m : legals) legal_mask[m.second * GoBoard::SIZE + m.first] = 1;

        std::array<float, NN_CHANNELS* NN_CELLS> obs;
        BuildObservation(board, color, obs);

        try {
            Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            std::array<int64_t, 4> shape = { 1, NN_CHANNELS, GoBoard::SIZE, GoBoard::SIZE };
            Ort::Value input = Ort::Value::CreateTensor<float>(
                mem, obs.data(), obs.size(), shape.data(), shape.size());

            const char* in_names[] = { "state" };
            const char* out_names[] = { "policy_logits", "value" };
            auto outputs = session_.Run(Ort::RunOptions{ nullptr }, in_names, &input, 1, out_names, 2);

            const float* logits = outputs[0].GetTensorData<float>();      // (1, 81)
            int a = SelectAction(logits, legal_mask, temperature, rng_);
            if (a < 0) return pass_move;
            return std::make_pair(a % GoBoard::SIZE, a / GoBoard::SIZE);
        }
        catch (const std::exception&) {
            return pass_move;      // 推論に失敗した場合はパス (呼び出し側でランダムにフォールバックしても良い)
        }
    }
};

// ===================== 設定 =====================
const wchar_t* const MODEL_FILE = L"go_agent_trained.onnx";
const float AI_TEMPERATURE = 0.5f;       // 0 で貪欲 (最大確率の手)。学習時のサンプリングは 1.0
const int   MAX_TOTAL_MOVES = 300;       // 学習環境の打ち切り (300手) に合わせる (パスも1手と数える)
const bool  WIN_BY_CAPTURES = false;     // false: 面積計算+コミ / true: 捕獲数で勝敗 (学習環境の報酬と同じ基準)

// ===================== GUI Constants =====================
const int CELL_SIZE = 50;
const int BOARD_OFFSET_X = 50;
const int BOARD_OFFSET_Y = 50;
const int BOARD_PX_SIZE = CELL_SIZE * 9;

const unsigned long C_BOARD2 = 0x00A58E5C;
const unsigned long C_LINE = 0x00000000;
const unsigned long C_BLACK = 0x00000000;
const unsigned long C_WHITE = 0x00FFFFFF;
const unsigned long C_RED = 0x000000FF;
const unsigned long C_GRAY = 0x00808080;
const unsigned long C_HOSHI = 0x00000000;
const unsigned long C_TEXT = 0x00000000;

// ===================== Game State =====================
enum class GameState { MENU, PLAYING, GAME_OVER };

struct GameResult {
    Stone winner = Stone::EMPTY;         // EMPTY = 引き分け
    float margin = 0.0f;                 // 勝ち幅 (目 or 捕獲数)
    bool by_captures = false;
    GoBoard::Score score = {};
    int black_caps = 0;
    int white_caps = 0;
    std::wstring reason;
};

// ===================== Global State =====================
GoBoard g_board;
RandomAgent g_random_ai(123);
std::unique_ptr<NNAgent> g_nn;
std::wstring g_ai_info = L"AI: Random";

GameState g_gameState = GameState::MENU;
Stone g_human_color = Stone::WHITE;
Stone g_current_turn = Stone::BLACK;
bool g_ai_thinking = false;
bool g_game_over = false;
int g_pass_count = 0;
int g_move_count = 0;
int g_black_caps = 0;
int g_white_caps = 0;
int g_last_x = -1;
int g_last_y = -1;
int g_hoverX = -1;
int g_hoverY = -1;
bool g_trackingMouse = false;
std::wstring g_status_note;
GameResult g_result;
HWND g_hWnd = NULL;

// ===================== Helpers =====================
int ToPixelX(int bx) { return BOARD_OFFSET_X + bx * CELL_SIZE + CELL_SIZE / 2; }
int ToPixelY(int by) { return BOARD_OFFSET_Y + by * CELL_SIZE + CELL_SIZE / 2; }
bool ToBoardCoord(int px, int py, int& bx, int& by) {
    if (px < BOARD_OFFSET_X || py < BOARD_OFFSET_Y) return false;   // 負数の除算で 0 に丸められるのを防ぐ
    bx = (px - BOARD_OFFSET_X) / CELL_SIZE;
    by = (py - BOARD_OFFSET_Y) / CELL_SIZE;
    return (bx >= 0 && bx < 9 && by >= 0 && by < 9);
}

std::wstring GetExeDir() {
    wchar_t buf[MAX_PATH];
    DWORD n = GetModuleFileNameW(NULL, buf, MAX_PATH);
    std::wstring p(buf, n);
    size_t pos = p.find_last_of(L"\\/");
    return (pos == std::wstring::npos) ? L"" : p.substr(0, pos + 1);
}

bool FileExists(const std::wstring& path) {
    DWORD a = GetFileAttributesW(path.c_str());
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}

const wchar_t* ColorName(Stone c) { return (c == Stone::BLACK) ? L"Black" : L"White"; }

std::wstring Fmt1(float v) {
    std::wstringstream ss;
    ss << std::fixed << std::setprecision(1) << v;
    return ss.str();
}

void LoadAgent() {
    std::wstring path = GetExeDir() + MODEL_FILE;
    if (!FileExists(path)) path = MODEL_FILE;      // カレントディレクトリも試す
    std::wstring err;
    try {
        g_nn = std::make_unique<NNAgent>();
        if (g_nn->load(path, err)) {
            g_ai_info = std::wstring(L"AI: ") + MODEL_FILE;
            return;
        }
    }
    catch (const std::exception& e) {
        err = Widen(e.what());
    }
    g_nn.reset();
    g_ai_info = L"AI: Random (model not loaded)";
    MessageBoxW(NULL, (std::wstring(L"Could not load ") + MODEL_FILE +
        L"\nFalling back to the random AI.\n\n" + err).c_str(), L"Go 9x9", MB_OK | MB_ICONWARNING);
}

// ===================== Drawing =====================
void FillCircle(HDC hdc, int cx, int cy, int r, COLORREF fill, COLORREF outline, int pen_width = 1) {
    HBRUSH br = CreateSolidBrush(fill);
    HPEN pen = CreatePen(PS_SOLID, pen_width, outline);
    HGDIOBJ old_br = SelectObject(hdc, br);
    HGDIOBJ old_pen = SelectObject(hdc, pen);
    Ellipse(hdc, cx - r, cy - r, cx + r, cy + r);
    SelectObject(hdc, old_br);
    SelectObject(hdc, old_pen);
    DeleteObject(br);
    DeleteObject(pen);
}

HFONT MakeFont(int height, int weight) {
    return CreateFont(height, 0, 0, 0, weight, FALSE, FALSE, FALSE,
        DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
        DEFAULT_QUALITY, DEFAULT_PITCH | FF_SWISS, L"Microsoft Sans Serif");
}

void DrawStone(HDC hdc, int bx, int by, Stone s) {
    if (s == Stone::EMPTY) return;
    int r = CELL_SIZE / 2 - 2;
    if (s == Stone::BLACK) FillCircle(hdc, ToPixelX(bx), ToPixelY(by), r, C_BLACK, C_BLACK);
    else                   FillCircle(hdc, ToPixelX(bx), ToPixelY(by), r, C_WHITE, C_BLACK);
}

// Hover preview (pseudo-transparent)
void DrawPreviewStone(HDC hdc, int bx, int by, Stone s) {
    int r = CELL_SIZE / 2 - 6;
    if (s == Stone::BLACK) FillCircle(hdc, ToPixelX(bx), ToPixelY(by), r, RGB(100, 100, 100), RGB(100, 100, 100));
    else                   FillCircle(hdc, ToPixelX(bx), ToPixelY(by), r, RGB(220, 220, 220), RGB(120, 120, 120));
}

// Invalid move mark (red X)
void DrawInvalidMark(HDC hdc, int bx, int by) {
    int px = ToPixelX(bx);
    int py = ToPixelY(by);
    int r = CELL_SIZE / 2 - 10;
    HPEN pen = CreatePen(PS_SOLID, 3, RGB(255, 0, 0));
    HGDIOBJ old = SelectObject(hdc, pen);
    MoveToEx(hdc, px - r, py - r, NULL);
    LineTo(hdc, px + r, py + r);
    MoveToEx(hdc, px + r, py - r, NULL);
    LineTo(hdc, px - r, py + r);
    SelectObject(hdc, old);
    DeleteObject(pen);
}

// 勝敗結果パネル (盤面の上に重ねて表示)
void DrawResultOverlay(HDC hdc) {
    const GameResult& r = g_result;

    RECT panel = { BOARD_OFFSET_X + 25, BOARD_OFFSET_Y + 90,
                   BOARD_OFFSET_X + BOARD_PX_SIZE - 25, BOARD_OFFSET_Y + 360 };
    HBRUSH br = CreateSolidBrush(RGB(250, 250, 250));
    HPEN pen = CreatePen(PS_SOLID, 3, RGB(30, 30, 30));
    HGDIOBJ ob = SelectObject(hdc, br);
    HGDIOBJ op = SelectObject(hdc, pen);
    Rectangle(hdc, panel.left, panel.top, panel.right, panel.bottom);
    SelectObject(hdc, ob);
    SelectObject(hdc, op);
    DeleteObject(br);
    DeleteObject(pen);

    std::wstring title, headline;
    COLORREF title_color;
    if (r.winner == Stone::EMPTY) {
        title = L"Draw";
        title_color = RGB(80, 80, 80);
        headline = L"Both players are even";
    }
    else {
        bool human_won = (r.winner == g_human_color);
        title = human_won ? L"You Win!" : L"You Lose...";
        title_color = human_won ? RGB(0, 130, 0) : RGB(190, 0, 0);
        headline = std::wstring(ColorName(r.winner)) + L" wins by " + Fmt1(r.margin) +
            (r.by_captures ? L" captures" : L" points");
    }

    std::vector<std::wstring> lines;
    lines.push_back(headline);
    lines.push_back(L"Black: " + Fmt1(r.score.black_score) + L"  (stones " + std::to_wstring(r.score.black_stones) +
        L" + territory " + Fmt1(r.score.black_territory) + L")");
    lines.push_back(L"White: " + Fmt1(r.score.white_score) + L"  (stones " + std::to_wstring(r.score.white_stones) +
        L" + territory " + Fmt1(r.score.white_territory) + L" + komi " + Fmt1(GoBoard::KOMI) + L")");
    lines.push_back(L"Captures  Black: " + std::to_wstring(r.black_caps) + L"   White: " + std::to_wstring(r.white_caps));
    lines.push_back(r.reason);

    SetBkMode(hdc, TRANSPARENT);

    HFONT ft = MakeFont(40, FW_BOLD);
    HGDIOBJ of = SelectObject(hdc, ft);
    SetTextColor(hdc, title_color);
    RECT rt = { panel.left, panel.top + 12, panel.right, panel.top + 62 };
    DrawText(hdc, title.c_str(), -1, &rt, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
    SelectObject(hdc, of);
    DeleteObject(ft);

    HFONT fb = MakeFont(17, FW_NORMAL);
    of = SelectObject(hdc, fb);
    SetTextColor(hdc, C_TEXT);
    int y = panel.top + 75;
    for (size_t i = 0; i < lines.size(); ++i) {
        RECT rl = { panel.left + 15, y, panel.right - 15, y + 24 };
        DrawText(hdc, lines[i].c_str(), -1, &rl,
            (i == 0 ? DT_CENTER : DT_LEFT) | DT_VCENTER | DT_SINGLELINE);
        y += (i == 0) ? 34 : 26;
    }
    SetTextColor(hdc, RGB(90, 90, 90));
    RECT rh = { panel.left, panel.bottom - 34, panel.right, panel.bottom - 8 };
    DrawText(hdc, L"Click to return to the menu", -1, &rh, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
    SelectObject(hdc, of);
    DeleteObject(fb);
}

void DrawBoard(HDC hdc) {
    RECT brc = { BOARD_OFFSET_X, BOARD_OFFSET_Y,
                 BOARD_OFFSET_X + BOARD_PX_SIZE, BOARD_OFFSET_Y + BOARD_PX_SIZE };
    HBRUSH board_br = CreateSolidBrush(C_BOARD2);
    FillRect(hdc, &brc, board_br);
    DeleteObject(board_br);

    HPEN pen = CreatePen(PS_SOLID, 1, C_LINE);
    HGDIOBJ old = SelectObject(hdc, pen);
    for (int i = 0; i < 9; ++i) {
        int x = ToPixelX(i);
        MoveToEx(hdc, x, ToPixelY(0), NULL);
        LineTo(hdc, x, ToPixelY(8));

        int y = ToPixelY(i);
        MoveToEx(hdc, ToPixelX(0), y, NULL);
        LineTo(hdc, ToPixelX(8), y);
    }
    SelectObject(hdc, old);
    DeleteObject(pen);

    const int hoshi[5][2] = { {2,2}, {2,6}, {4,4}, {6,2}, {6,6} };
    for (int i = 0; i < 5; ++i)
        FillCircle(hdc, ToPixelX(hoshi[i][0]), ToPixelY(hoshi[i][1]), 4, C_HOSHI, C_HOSHI);

    for (int y = 0; y < 9; ++y)
        for (int x = 0; x < 9; ++x)
            DrawStone(hdc, x, y, g_board.get(x, y));

    bool human_turn = (g_gameState == GameState::PLAYING && !g_game_over && !g_ai_thinking &&
        g_current_turn == g_human_color);

    // Hover preview
    if (human_turn && g_hoverX != -1) {
        if (g_board.get(g_hoverX, g_hoverY) == Stone::EMPTY) {
            if (g_board.is_legal(g_hoverX, g_hoverY, g_human_color))
                DrawPreviewStone(hdc, g_hoverX, g_hoverY, g_human_color);
            else
                DrawInvalidMark(hdc, g_hoverX, g_hoverY);
        }
    }

    // 直前の手
    if (g_last_x != -1) {
        int px = ToPixelX(g_last_x);
        int py = ToPixelY(g_last_y);
        int r = 6;
        HPEN redpen = CreatePen(PS_SOLID, 2, C_RED);
        HGDIOBJ old2 = SelectObject(hdc, redpen);
        HGDIOBJ old_br = SelectObject(hdc, GetStockObject(NULL_BRUSH));
        Rectangle(hdc, px - r, py - r, px + r, py + r);
        SelectObject(hdc, old_br);
        SelectObject(hdc, old2);
        DeleteObject(redpen);
    }

    // 合法手の表示
    if (human_turn) {
        auto legals = g_board.get_legal_moves(g_human_color);
        for (size_t i = 0; i < legals.size(); ++i)
            FillCircle(hdc, ToPixelX(legals[i].first), ToPixelY(legals[i].second), 4, C_GRAY, C_GRAY);
    }

    // 座標ラベル
    SetTextColor(hdc, C_TEXT);
    SetBkMode(hdc, TRANSPARENT);
    HFONT hf = MakeFont(16, FW_NORMAL);
    HGDIOBJ oldf = SelectObject(hdc, hf);

    for (int i = 0; i < 9; ++i) {
        wchar_t buf[2];
        buf[0] = L'A' + i; buf[1] = 0;
        if (buf[0] >= L'I') buf[0]++;
        int px = ToPixelX(i);
        RECT r1 = { px - 10, BOARD_OFFSET_Y - 20, px + 10, BOARD_OFFSET_Y };
        DrawText(hdc, buf, -1, &r1, DT_CENTER | DT_BOTTOM);
        RECT r2 = { px - 10, BOARD_OFFSET_Y + BOARD_PX_SIZE, px + 10, BOARD_OFFSET_Y + BOARD_PX_SIZE + 20 };
        DrawText(hdc, buf, -1, &r2, DT_CENTER | DT_TOP);

        int py = ToPixelY(i);
        std::wstring num = std::to_wstring(9 - i);
        RECT r3 = { BOARD_OFFSET_X - 20, py - 10, BOARD_OFFSET_X, py + 10 };
        DrawText(hdc, num.c_str(), -1, &r3, DT_RIGHT | DT_VCENTER | DT_SINGLELINE);
        RECT r4 = { BOARD_OFFSET_X + BOARD_PX_SIZE, py - 10, BOARD_OFFSET_X + BOARD_PX_SIZE + 20, py + 10 };
        DrawText(hdc, num.c_str(), -1, &r4, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
    }

    // ステータス (2行)
    std::wstring line1, line2;
    if (g_gameState == GameState::GAME_OVER) {
        line1 = L"Game Over";
    }
    else if (g_ai_thinking) {
        line1 = L"AI thinking...";
    }
    else {
        line1 = std::wstring(L"Turn: ") + ColorName(g_current_turn) +
            (g_current_turn == g_human_color ? L" (You)" : L" (AI)");
        if (g_current_turn == g_human_color) line1 += L"  |  Left click: Move / Right click: Pass";
    }
    line2 = L"Move " + std::to_wstring(g_move_count) + L"  |  Captures  B: " + std::to_wstring(g_black_caps) +
        L"  W: " + std::to_wstring(g_white_caps);
    if (!g_status_note.empty()) line2 += L"  |  " + g_status_note;

    int sy = BOARD_OFFSET_Y + BOARD_PX_SIZE + 30;
    RECT rs1 = { BOARD_OFFSET_X, sy, BOARD_OFFSET_X + BOARD_PX_SIZE + 40, sy + 22 };
    RECT rs2 = { BOARD_OFFSET_X, sy + 24, BOARD_OFFSET_X + BOARD_PX_SIZE + 40, sy + 46 };
    DrawText(hdc, line1.c_str(), -1, &rs1, DT_LEFT | DT_TOP | DT_SINGLELINE);
    DrawText(hdc, line2.c_str(), -1, &rs2, DT_LEFT | DT_TOP | DT_SINGLELINE);

    SelectObject(hdc, oldf);
    DeleteObject(hf);

    if (g_gameState == GameState::GAME_OVER) DrawResultOverlay(hdc);
}

// Menu screen
void DrawMenuScreen(HDC hdc) {
    RECT rc;
    GetClientRect(g_hWnd, &rc);
    int cx = (rc.left + rc.right) / 2;
    int cy = (rc.top + rc.bottom) / 2;

    FillRect(hdc, &rc, (HBRUSH)GetStockObject(WHITE_BRUSH));

    SetTextColor(hdc, C_TEXT);
    SetBkMode(hdc, TRANSPARENT);

    HFONT hTitle = MakeFont(36, FW_BOLD);
    HFONT oldf = (HFONT)SelectObject(hdc, hTitle);
    RECT rTitle = { rc.left, cy - 100, rc.right, cy - 60 };
    DrawText(hdc, L"Go 9x9", -1, &rTitle, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
    SelectObject(hdc, oldf);
    DeleteObject(hTitle);

    HFONT hf = MakeFont(18, FW_NORMAL);
    oldf = (HFONT)SelectObject(hdc, hf);

    RECT rBlack = { cx - 130, cy - 20, cx + 130, cy + 20 };
    RECT rWhite = { cx - 130, cy + 35, cx + 130, cy + 75 };
    const RECT* btns[2] = { &rBlack, &rWhite };
    const wchar_t* labels[2] = { L"Play as Black (First move)", L"Play as White (Second move)" };
    for (int i = 0; i < 2; ++i) {
        HBRUSH br = CreateSolidBrush(RGB(240, 240, 240));
        HPEN pen = CreatePen(PS_SOLID, 2, RGB(0, 0, 0));
        HGDIOBJ ob = SelectObject(hdc, br);
        HGDIOBJ op = SelectObject(hdc, pen);
        Rectangle(hdc, btns[i]->left, btns[i]->top, btns[i]->right, btns[i]->bottom);
        SelectObject(hdc, ob);
        SelectObject(hdc, op);
        DeleteObject(br);
        DeleteObject(pen);
        RECT t = *btns[i];
        DrawText(hdc, labels[i], -1, &t, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
    }

    SetTextColor(hdc, RGB(90, 90, 90));
    RECT rInfo = { rc.left, cy + 95, rc.right, cy + 125 };
    DrawText(hdc, g_ai_info.c_str(), -1, &rInfo, DT_CENTER | DT_VCENTER | DT_SINGLELINE);

    SelectObject(hdc, oldf);
    DeleteObject(hf);
}

// ===================== Game Logic =====================
void FinishGame(const std::wstring& reason) {
    if (g_game_over) return;
    g_game_over = true;
    g_ai_thinking = false;
    g_gameState = GameState::GAME_OVER;
    g_hoverX = g_hoverY = -1;
    if (g_hWnd) KillTimer(g_hWnd, 1);

    GameResult r;
    r.score = g_board.calculate_score();
    r.black_caps = g_black_caps;
    r.white_caps = g_white_caps;
    r.reason = reason;
    r.by_captures = WIN_BY_CAPTURES;

    float diff;
    if (WIN_BY_CAPTURES) diff = static_cast<float>(g_black_caps - g_white_caps);
    else                 diff = r.score.black_score - r.score.white_score;

    r.winner = (diff > 0) ? Stone::BLACK : (diff < 0) ? Stone::WHITE : Stone::EMPTY;
    r.margin = std::fabs(diff);
    g_result = r;
}

void CheckGameOver() {
    if (g_game_over) return;
    if (g_pass_count >= 2) {
        FinishGame(L"Both players passed");
    }
    else if (g_move_count >= MAX_TOTAL_MOVES) {
        FinishGame(L"Move limit (" + std::to_wstring(MAX_TOTAL_MOVES) + L") reached");
    }
}

// 着手 (x == PASS_MOVE でパス) を盤面とゲーム状態に反映し、手番を交代する
void ApplyMove(int x, int y, Stone color) {
    bool is_pass = (x == GoBoard::PASS_MOVE);
    if (!is_pass && !g_board.play_move(x, y, color)) is_pass = true;   // 不正手は(起きない想定だが)パス扱い

    if (is_pass) {
        g_board.play_move(GoBoard::PASS_MOVE, GoBoard::PASS_MOVE, color);
        g_pass_count++;
        g_last_x = g_last_y = -1;
        g_status_note = std::wstring(ColorName(color)) + L" passed";
    }
    else {
        int cap = g_board.get_last_captured_count();
        if (color == Stone::BLACK) g_black_caps += cap; else g_white_caps += cap;
        g_pass_count = 0;
        g_last_x = x;
        g_last_y = y;
        g_status_note.clear();
    }
    g_move_count++;
    g_current_turn = opponent_of(color);
    CheckGameOver();
}

// 現在の勝敗指標で color が相手を上回っているか (AI のパス判断用)
bool IsWinning(Stone color) {
    if (WIN_BY_CAPTURES) {
        int mine = (color == Stone::BLACK) ? g_black_caps : g_white_caps;
        int theirs = (color == Stone::BLACK) ? g_white_caps : g_black_caps;
        return mine > theirs;
    }
    GoBoard::Score s = g_board.calculate_score();
    float mine = (color == Stone::BLACK) ? s.black_score : s.white_score;
    float theirs = (color == Stone::BLACK) ? s.white_score : s.black_score;
    return mine > theirs;
}

// 手番側に合法手が無ければ自動でパスし、AI の手番ならタイマーで思考を起動する
void ProcessNextTurn() {
    while (!g_game_over && g_board.get_legal_moves(g_current_turn).empty()) {
        std::wstring who = ColorName(g_current_turn);
        ApplyMove(GoBoard::PASS_MOVE, GoBoard::PASS_MOVE, g_current_turn);
        g_status_note = who + L" has no legal move (pass)";
    }
    if (!g_game_over && g_current_turn != g_human_color && g_hWnd) {
        SetTimer(g_hWnd, 1, 10, NULL);
    }
    if (g_hWnd) InvalidateRect(g_hWnd, NULL, FALSE);
}

void DoAIMove() {
    if (g_game_over || g_gameState != GameState::PLAYING) return;
    if (g_current_turn == g_human_color) return;

    g_ai_thinking = true;
    InvalidateRect(g_hWnd, NULL, FALSE);
    UpdateWindow(g_hWnd);
    Sleep(300);

    Stone me = g_current_turn;
    auto legals = g_board.get_legal_moves(me);

    // 学習済みエージェントには「パス」が無いため、終局はここで判断する:
    //   人間が直前にパスし (g_pass_count == 1)、AI が優勢なら AI もパスして終局する
    bool do_pass = legals.empty() || (g_pass_count == 1 && IsWinning(me));

    std::pair<int, int> mv(GoBoard::PASS_MOVE, GoBoard::PASS_MOVE);
    if (!do_pass) {
        if (g_nn && g_nn->loaded()) {
            mv = g_nn->select_move(g_board, me, legals, AI_TEMPERATURE);
            if (mv.first == GoBoard::PASS_MOVE) mv = g_random_ai.select_move(g_board, me);   // 推論失敗時の保険
        }
        else {
            mv = g_random_ai.select_move(g_board, me);
        }
    }

    g_ai_thinking = false;
    ApplyMove(mv.first, mv.second, me);
    ProcessNextTurn();
}

void HandleLeftClick(int px, int py) {
    if (g_gameState != GameState::PLAYING || g_game_over || g_ai_thinking) return;
    if (g_current_turn != g_human_color) return;

    int bx = 0, by = 0;
    if (!ToBoardCoord(px, py, bx, by)) return;
    if (!g_board.is_legal(bx, by, g_human_color)) return;

    ApplyMove(bx, by, g_human_color);
    ProcessNextTurn();
}

void HandleRightClick() {
    if (g_gameState != GameState::PLAYING || g_game_over || g_ai_thinking) return;
    if (g_current_turn != g_human_color) return;

    ApplyMove(GoBoard::PASS_MOVE, GoBoard::PASS_MOVE, g_human_color);
    ProcessNextTurn();
}

void StartGame(Stone human) {
    g_human_color = human;
    g_board = GoBoard();
    g_current_turn = Stone::BLACK;
    g_game_over = false;
    g_ai_thinking = false;
    g_pass_count = 0;
    g_move_count = 0;
    g_black_caps = g_white_caps = 0;
    g_last_x = g_last_y = -1;
    g_hoverX = g_hoverY = -1;
    g_status_note.clear();
    g_result = GameResult();
    g_gameState = GameState::PLAYING;
    SetWindowText(g_hWnd, human == Stone::BLACK ? L"Go 9x9 (You: Black, AI: White)"
        : L"Go 9x9 (You: White, AI: Black)");
    InvalidateRect(g_hWnd, NULL, TRUE);
    ProcessNextTurn();       // 人間が白番なら AI (黒) の初手をここで起動
}

void HandleMenuClick(int px, int py) {
    RECT rc;
    GetClientRect(g_hWnd, &rc);
    int cx = (rc.left + rc.right) / 2;
    int cy = (rc.top + rc.bottom) / 2;

    RECT rBlack = { cx - 130, cy - 20, cx + 130, cy + 20 };
    RECT rWhite = { cx - 130, cy + 35, cx + 130, cy + 75 };
    POINT pt = { px, py };

    if (PtInRect(&rBlack, pt))      StartGame(Stone::BLACK);
    else if (PtInRect(&rWhite, pt)) StartGame(Stone::WHITE);
}

// ===================== Window Procedure =====================
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_CREATE:
        g_hWnd = hWnd;
        LoadAgent();
        return 0;

    case WM_ERASEBKGND:
        return 1;     // ダブルバッファで全面を描くので背景消去は不要 (ちらつき防止)

    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hWnd, &ps);
        RECT rc;
        GetClientRect(hWnd, &rc);

        HDC mem = CreateCompatibleDC(hdc);
        HBITMAP bmp = CreateCompatibleBitmap(hdc, rc.right, rc.bottom);
        HGDIOBJ old_bmp = SelectObject(mem, bmp);
        FillRect(mem, &rc, (HBRUSH)GetStockObject(WHITE_BRUSH));

        if (g_gameState == GameState::MENU) DrawMenuScreen(mem);
        else                                DrawBoard(mem);

        BitBlt(hdc, 0, 0, rc.right, rc.bottom, mem, 0, 0, SRCCOPY);
        SelectObject(mem, old_bmp);
        DeleteObject(bmp);
        DeleteDC(mem);
        EndPaint(hWnd, &ps);
        return 0;
    }

    case WM_MOUSEMOVE: {
        if (g_gameState == GameState::PLAYING && !g_game_over && !g_ai_thinking &&
            g_current_turn == g_human_color) {
            int bx = 0, by = 0;
            if (ToBoardCoord(LOWORD(lParam), HIWORD(lParam), bx, by)) {
                if (bx != g_hoverX || by != g_hoverY) {
                    g_hoverX = bx;
                    g_hoverY = by;
                    InvalidateRect(hWnd, NULL, FALSE);
                }
            }
            else if (g_hoverX != -1) {
                g_hoverX = -1;
                g_hoverY = -1;
                InvalidateRect(hWnd, NULL, FALSE);
            }
        }
        if (!g_trackingMouse) {
            TRACKMOUSEEVENT tme = { sizeof(TRACKMOUSEEVENT), TME_LEAVE, hWnd, 0 };
            TrackMouseEvent(&tme);
            g_trackingMouse = true;
        }
        return 0;
    }

    case WM_MOUSELEAVE:
        g_trackingMouse = false;
        if (g_hoverX != -1) {
            g_hoverX = -1;
            g_hoverY = -1;
            InvalidateRect(hWnd, NULL, FALSE);
        }
        return 0;

    case WM_LBUTTONUP: {
        int x = LOWORD(lParam);
        int y = HIWORD(lParam);
        if (g_gameState == GameState::MENU) {
            HandleMenuClick(x, y);
        }
        else if (g_gameState == GameState::PLAYING) {
            HandleLeftClick(x, y);
        }
        else if (g_gameState == GameState::GAME_OVER) {
            g_gameState = GameState::MENU;
            SetWindowText(g_hWnd, L"Go 9x9 - Select Color");
            InvalidateRect(hWnd, NULL, FALSE);
        }
        return 0;
    }

    case WM_RBUTTONUP: {
        if (g_gameState == GameState::PLAYING) {
            HandleRightClick();
        }
        return 0;
    }

    case WM_TIMER:
        if (wParam == 1) {
            KillTimer(hWnd, 1);
            DoAIMove();
        }
        return 0;

    case WM_DESTROY:
        g_nn.reset();
        PostQuitMessage(0);
        return 0;

    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
}

// ===================== Entry Point =====================
int APIENTRY wWinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPWSTR lpCmdLine, _In_ int nCmdShow) {
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);

    WNDCLASSEXW wcex = {};
    wcex.cbSize = sizeof(WNDCLASSEX);
    wcex.style = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc = WndProc;
    wcex.hInstance = hInstance;
    wcex.hIcon = LoadIcon(nullptr, IDI_APPLICATION);
    wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wcex.lpszClassName = L"GoGUI";
    wcex.hIconSm = LoadIcon(nullptr, IDI_APPLICATION);

    if (!RegisterClassExW(&wcex)) return 1;

    int winW = BOARD_OFFSET_X * 2 + BOARD_PX_SIZE + 40;
    int winH = BOARD_OFFSET_Y * 2 + BOARD_PX_SIZE + 120;

    HWND hWnd = CreateWindowExW(0, L"GoGUI", L"Go 9x9 - Select Color",
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
        CW_USEDEFAULT, CW_USEDEFAULT, winW, winH,
        nullptr, nullptr, hInstance, nullptr);

    if (!hWnd) return 1;

    ShowWindow(hWnd, nCmdShow);
    UpdateWindow(hWnd);

    MSG msg;
    while (GetMessage(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return (int)msg.wParam;
}
