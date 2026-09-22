#include <iostream>
#include <vector>
#include <queue>
#include <set>
#include <algorithm>
#include <random>
#include <cmath>
#include <cstring>
#include <string>

// ===================== 基本定義 =====================
enum class Stone : int { EMPTY = 0, BLACK = 1, WHITE = 2 };

inline Stone opponent_of(Stone c) {
    return (c == Stone::BLACK) ? Stone::WHITE : Stone::BLACK;
}

inline int stone_to_int(Stone c) {
    return static_cast<int>(c);
}

// 表示用文字（コンソールで文字化けしにくいASCII版とUnicode版）
constexpr bool USE_UNICODE_STONES = true;  // false にすると + - . に切り替わります

char stone_char(Stone s) {
    if (USE_UNICODE_STONES) {
        if (s == Stone::BLACK) return '●';
        if (s == Stone::WHITE) return '○';
    }
    else {
        if (s == Stone::BLACK) return '@';
        if (s == Stone::WHITE) return 'O';
    }
    return '.';
}

// ===================== 盤面クラス =====================
class GoBoard {
public:
    static constexpr int SIZE = 9;
    static constexpr float KOMI = 6.5f;
    static constexpr int PASS_MOVE = -1;

private:
    // 盤面状態
    std::vector<Stone> board;
    int ko_pos;  // コウの位置（-1でなし）
    int last_captured_count; // 直前の手で取った石の数

    // Union-Find用
    mutable std::vector<int> parent;
    mutable std::vector<int> liberties;
    mutable std::vector<int> group_size;

    // 方向ベクトル
    const int dx[4] = { 1, -1, 0, 0 };
    const int dy[4] = { 0, 0, 1, -1 };

public:
    GoBoard() : board(SIZE* SIZE, Stone::EMPTY), ko_pos(-1), last_captured_count(0) {
        reset_union_find();
    }

    // --- アクセサ ---
    Stone get(int x, int y) const {
        if (x < 0 || x >= SIZE || y < 0 || y >= SIZE) return Stone::EMPTY;
        return board[index(x, y)];
    }

    int get_ko_pos() const { return ko_pos; }

    // --- 合法手判定 ---
    bool is_legal(int x, int y, Stone color) const {
        if (x == PASS_MOVE) return true;
        if (x < 0 || x >= SIZE || y < 0 || y >= SIZE) return false;
        int idx = index(x, y);
        if (board[idx] != Stone::EMPTY) return false;
        if (idx == ko_pos) return false;

        // 仮に置いてみて自殺手かチェック
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

    // --- 着手実行（外部から呼ぶ用） ---
    bool play_move(int x, int y, Stone color) {
        if (x == PASS_MOVE) {
            ko_pos = -1;  // パスでコウ解除
            return true;
        }
        int captured = 0, new_ko = -1;
        bool ok = place_stone_raw(x, y, color, captured, new_ko);
        if (!ok) return false;
        ko_pos = new_ko;
        last_captured_count = captured;
        return true;
    }

    // --- 地の計算（領域＋石数、簡易トロール） ---
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

        // 石数カウント
        for (int i = 0; i < SIZE * SIZE; ++i) {
            if (board[i] == Stone::BLACK) s.black_stones++;
            else if (board[i] == Stone::WHITE) s.white_stones++;
        }

        // 領域探索（BFS）
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

    // --- 盤面表示 ---
    void print() const {
        std::cout << "\n    ";
        for (int x = 0; x < SIZE; ++x) {
            char c = 'A' + x;
            // I列をスキップする伝統的な表記
            if (c >= 'I') c++;
            std::cout << c << " ";
        }
        std::cout << "\n   +";
        for (int x = 0; x < SIZE; ++x) std::cout << "--";
        std::cout << "-+\n";

        for (int y = 0; y < SIZE; ++y) {
            std::cout << (SIZE - y) << "  |";
            for (int x = 0; x < SIZE; ++x) {
                std::cout << " " << stone_char(board[index(x, y)]);
            }
            std::cout << " | " << (SIZE - y) << "\n";
        }

        std::cout << "   +";
        for (int x = 0; x < SIZE; ++x) std::cout << "--";
        std::cout << "-+\n    ";
        for (int x = 0; x < SIZE; ++x) {
            char c = 'A' + x;
            if (c >= 'I') c++;
            std::cout << c << " ";
        }
        std::cout << "\n" << std::endl;
    }

    // --- デバッグ用: 気の表示 ---
    void print_liberties() const {
        reset_union_find();
        std::cout << "\nLiberties map:\n";
        for (int y = 0; y < SIZE; ++y) {
            for (int x = 0; x < SIZE; ++x) {
                int idx = index(x, y);
                if (board[idx] == Stone::EMPTY) {
                    std::cout << " .";
                }
                else {
                    int root = find_root(idx);
                    std::cout << " " << liberties[root];
                }
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

private:
    inline int index(int x, int y) const { return y * SIZE + x; }
    inline bool in_bounds(int x, int y) const { return x >= 0 && x < SIZE && y >= 0 && y < SIZE; }

    // --- Union-Find ---
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

        // 同じ色の隣接を統合
        for (int y = 0; y < SIZE; ++y)
            for (int x = 0; x < SIZE; ++x) {
                int idx = index(x, y);
                if (board[idx] == Stone::EMPTY) continue;
                for (int d = 0; d < 2; ++d) { // 右と下だけ見れば十分
                    int nx = x + dx[d], ny = y + dy[d];
                    if (!in_bounds(nx, ny)) continue;
                    int nidx = index(nx, ny);
                    if (board[nidx] == board[idx]) unite(idx, nidx);
                }
            }

        // 気をカウント
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

    // --- 内部: 着手の実処理 ---
    bool place_stone_raw(int x, int y, Stone color, int& out_captured, int& out_new_ko) {
        out_captured = 0;
        out_new_ko = -1;

        int idx = index(x, y);
        board[idx] = color;
        Stone opp = opponent_of(color);

        // 盤面のバックアップ（自殺手の復元用）
        std::vector<Stone> backup_board = board;

        // 相手の連をチェックして捕獲
        reset_union_find();
        std::vector<int> captured_indices;

        for (int d = 0; d < 4; ++d) {
            int nx = x + dx[d], ny = y + dy[d];
            if (!in_bounds(nx, ny)) continue;
            int nidx = index(nx, ny);
            if (board[nidx] != opp) continue;

            int root = find_root(nidx);
            if (liberties[root] == 0) {
                // この連を除去
                for (int i = 0; i < SIZE * SIZE; ++i)
                    if (find_root(i) == root) {
                        captured_indices.push_back(i);
                        board[i] = Stone::EMPTY;
                    }
            }
        }

        out_captured = static_cast<int>(captured_indices.size());

        // 自殺手チェック
        reset_union_find();
        int my_root = find_root(idx);
        if (liberties[my_root] == 0) {
            // 元に戻す
            board = backup_board;
            return false;
        }

        // コウ判定: 1石だけ取り、かつ取った位置に打つと1気の連になる場合
        if (out_captured == 1) {
            out_new_ko = captured_indices[0];
        }

        return true;
    }
};

// ===================== ランダムAI =====================
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

// ===================== メイン =====================
int main() {
    GoBoard board;
    RandomAgent agent_black(123);
    RandomAgent agent_white(456);

    Stone current = Stone::BLACK;
    int pass_count = 0;
    int move_num = 1;

    std::cout << "========================================\n";
    std::cout << "  Visual Studio C++ Go Engine (9x9)\n";
    std::cout << "========================================\n";
    std::cout << "Black: " << (USE_UNICODE_STONES ? "●" : "@") << "\n";
    std::cout << "White: " << (USE_UNICODE_STONES ? "○" : "O") << "\n";
    std::cout << "Komi: " << GoBoard::KOMI << "\n";
    board.print();

    while (pass_count < 2 && move_num <= 400) {
        auto [x, y] = (current == Stone::BLACK)
            ? agent_black.select_move(board, current)
            : agent_white.select_move(board, current);

        if (x == GoBoard::PASS_MOVE) {
            std::cout << move_num << ". " << (current == Stone::BLACK ? "Black" : "White") << " passes.\n";
            board.play_move(x, y, current);  // パス処理
            pass_count++;
        }
        else {
            bool ok = board.play_move(x, y, current);
            if (!ok) {
                std::cout << "Illegal move attempted at (" << x << "," << y << "). Skipping.\n";
                current = opponent_of(current);
                continue;
            }

            pass_count = 0;
            char col = 'A' + x;
            if (col >= 'I') col++;  // 伝統的な飛び地
            int row = GoBoard::SIZE - y;

            std::cout << move_num << ". "
                << (current == Stone::BLACK ? "Black" : "White")
                << " " << col << row;

            // 直前の着手で取った石数は board 内部で管理しているが、
            // 簡易のためここでは表示しない（必要ならGoBoardにゲッターを追加可能）
            std::cout << "\n";
        }

        current = opponent_of(current);
        move_num++;

        // 50手ごとに盤面表示
        if (move_num % 50 == 0) {
            board.print();
        }
    }

    std::cout << "\n========== Game Over ==========\n";
    board.print();

    auto score = board.calculate_score();
    std::cout << "Black stones: " << score.black_stones << " | Territory: " << score.black_territory << "\n";
    std::cout << "White stones: " << score.white_stones << " | Territory: " << score.white_territory << " | Komi: " << GoBoard::KOMI << "\n";
    std::cout << "---------------------------------\n";
    std::cout << "Black total: " << score.black_score << "\n";
    std::cout << "White total: " << score.white_score << "\n";

    if (score.black_score > score.white_score)
        std::cout << "Winner: Black by " << (score.black_score - score.white_score) << " points.\n";
    else
        std::cout << "Winner: White by " << (score.white_score - score.black_score) << " points.\n";

    return 0;
}