// GoGUI_v2.cpp
// Visual Studio 2022 / Win32 API / Unicode / Windows Subsystem

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <string>
#include <vector>
#include <queue>
#include <set>
#include <algorithm>
#include <random>
#include <sstream>

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

// ===================== Global State =====================
GoBoard g_board;
RandomAgent g_ai(123);
GameState g_gameState = GameState::MENU;
Stone g_human_color = Stone::WHITE;
Stone g_current_turn = Stone::BLACK;
bool g_ai_thinking = false;
bool g_game_over = false;
int g_pass_count = 0;
int g_last_x = -1;
int g_last_y = -1;
int g_hoverX = -1;
int g_hoverY = -1;
bool g_trackingMouse = false;
HWND g_hWnd = NULL;

// ===================== Helpers =====================
int ToPixelX(int bx) { return BOARD_OFFSET_X + bx * CELL_SIZE + CELL_SIZE / 2; }
int ToPixelY(int by) { return BOARD_OFFSET_Y + by * CELL_SIZE + CELL_SIZE / 2; }
bool ToBoardCoord(int px, int py, int& bx, int& by) {
    bx = (px - BOARD_OFFSET_X) / CELL_SIZE;
    by = (py - BOARD_OFFSET_Y) / CELL_SIZE;
    return (bx >= 0 && bx < 9 && by >= 0 && by < 9);
}

// ===================== Drawing =====================
void DrawStone(HDC hdc, int bx, int by, Stone s) {
    if (s == Stone::EMPTY) return;
    int px = ToPixelX(bx);
    int py = ToPixelY(by);
    int r = CELL_SIZE / 2 - 2;
    RECT rc = { px - r, py - r, px + r, py + r };
    if (s == Stone::BLACK) {
        HBRUSH br = CreateSolidBrush(C_BLACK);
        Ellipse(hdc, rc.left, rc.top, rc.right, rc.bottom);
        DeleteObject(br);
    }
    else {
        HBRUSH br = CreateSolidBrush(C_WHITE);
        HPEN pen = CreatePen(PS_SOLID, 1, C_BLACK);
        HGDIOBJ old_pen = SelectObject(hdc, pen);
        Ellipse(hdc, rc.left, rc.top, rc.right, rc.bottom);
        FillRect(hdc, &rc, br);
        Ellipse(hdc, rc.left, rc.top, rc.right, rc.bottom);
        SelectObject(hdc, old_pen);
        DeleteObject(pen);
        DeleteObject(br);
    }
}

// NEW: Hover preview (pseudo-transparent)
void DrawPreviewStone(HDC hdc, int bx, int by, Stone s) {
    int px = ToPixelX(bx);
    int py = ToPixelY(by);
    int r = CELL_SIZE / 2 - 6;
    RECT rc = { px - r, py - r, px + r, py + r };
    if (s == Stone::BLACK) {
        HBRUSH br = CreateSolidBrush(RGB(100, 100, 100)); // dim black
        HGDIOBJ old = SelectObject(hdc, br);
        Ellipse(hdc, rc.left, rc.top, rc.right, rc.bottom);
        FillRect(hdc, &rc, br);
        Ellipse(hdc, rc.left, rc.top, rc.right, rc.bottom);
        SelectObject(hdc, old);
        DeleteObject(br);
    }
    else {
        HBRUSH br = CreateSolidBrush(RGB(220, 220, 220)); // dim white
        HPEN pen = CreatePen(PS_SOLID, 1, RGB(120, 120, 120));
        HGDIOBJ old_pen = SelectObject(hdc, pen);
        HGDIOBJ old_br = SelectObject(hdc, br);
        Ellipse(hdc, rc.left, rc.top, rc.right, rc.bottom);
        FillRect(hdc, &rc, br);
        Ellipse(hdc, rc.left, rc.top, rc.right, rc.bottom);
        SelectObject(hdc, old_br);
        SelectObject(hdc, old_pen);
        DeleteObject(pen);
        DeleteObject(br);
    }
}

// NEW: Invalid move mark (red X)
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
        int y0 = ToPixelY(0);
        int y1 = ToPixelY(8);
        MoveToEx(hdc, x, y0, NULL);
        LineTo(hdc, x, y1);

        int y = ToPixelY(i);
        int x0 = ToPixelX(0);
        int x1 = ToPixelX(8);
        MoveToEx(hdc, x0, y, NULL);
        LineTo(hdc, x1, y);
    }
    SelectObject(hdc, old);
    DeleteObject(pen);

    const int hoshi[5][2] = { {2,2}, {2,6}, {4,4}, {6,2}, {6,6} };
    HBRUSH hbr = CreateSolidBrush(C_HOSHI);
    for (int i = 0; i < 5; ++i) {
        int px = ToPixelX(hoshi[i][0]);
        int py = ToPixelY(hoshi[i][1]);
        RECT r = { px - 3, py - 3, px + 3, py + 3 };
        Ellipse(hdc, r.left, r.top, r.right, r.bottom);
        FillRect(hdc, &r, hbr);
    }
    DeleteObject(hbr);

    for (int y = 0; y < 9; ++y)
        for (int x = 0; x < 9; ++x)
            DrawStone(hdc, x, y, g_board.get(x, y));

    // NEW: Hover preview
    if (g_gameState == GameState::PLAYING && !g_game_over && !g_ai_thinking &&
        g_current_turn == g_human_color && g_hoverX != -1) {
        if (g_board.get(g_hoverX, g_hoverY) == Stone::EMPTY) {
            if (g_board.is_legal(g_hoverX, g_hoverY, g_human_color)) {
                DrawPreviewStone(hdc, g_hoverX, g_hoverY, g_human_color);
            }
            else {
                DrawInvalidMark(hdc, g_hoverX, g_hoverY);
            }
        }
    }

    if (g_last_x != -1) {
        int px = ToPixelX(g_last_x);
        int py = ToPixelY(g_last_y);
        int r = 6;
        HPEN redpen = CreatePen(PS_SOLID, 2, C_RED);
        HGDIOBJ old2 = SelectObject(hdc, redpen);
        HBRUSH nullbr = (HBRUSH)GetStockObject(NULL_BRUSH);
        HGDIOBJ old_br = SelectObject(hdc, nullbr);
        Rectangle(hdc, px - r, py - r, px + r, py + r);
        SelectObject(hdc, old_br);
        SelectObject(hdc, old2);
        DeleteObject(redpen);
    }

    if (g_gameState == GameState::PLAYING && !g_game_over && !g_ai_thinking && g_current_turn == g_human_color) {
        auto legals = g_board.get_legal_moves(g_human_color);
        HBRUSH gbr = CreateSolidBrush(C_GRAY);
        for (size_t i = 0; i < legals.size(); ++i) {
            int px = ToPixelX(legals[i].first);
            int py = ToPixelY(legals[i].second);
            RECT r = { px - 4, py - 4, px + 4, py + 4 };
            Ellipse(hdc, r.left, r.top, r.right, r.bottom);
            FillRect(hdc, &r, gbr);
        }
        DeleteObject(gbr);
    }

    SetTextColor(hdc, C_TEXT);
    SetBkMode(hdc, TRANSPARENT);
    HFONT hf = CreateFont(16, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
        DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
        DEFAULT_QUALITY, DEFAULT_PITCH | FF_SWISS, L"Microsoft Sans Serif");
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
        RECT r3 = { BOARD_OFFSET_Y - 20, py - 10, BOARD_OFFSET_Y, py + 10 };
        DrawText(hdc, num.c_str(), -1, &r3, DT_RIGHT | DT_VCENTER | DT_SINGLELINE);
        RECT r4 = { BOARD_OFFSET_Y + BOARD_PX_SIZE, py - 10, BOARD_OFFSET_Y + BOARD_PX_SIZE + 20, py + 10 };
        DrawText(hdc, num.c_str(), -1, &r4, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
    }

    RECT rc = { BOARD_OFFSET_X, BOARD_OFFSET_Y + BOARD_PX_SIZE + 30,
                BOARD_OFFSET_X + BOARD_PX_SIZE, BOARD_OFFSET_Y + BOARD_PX_SIZE + 80 };
    std::wstring status;
    if (g_gameState == GameState::GAME_OVER) {
        status = L"Game Over. ";
        auto sc = g_board.calculate_score();
        std::wstringstream wss;
        wss << L"B:" << sc.black_score << L" W:" << sc.white_score << L" ";
        if (sc.black_score > sc.white_score) wss << L"Black wins";
        else wss << L"White wins";
        status += wss.str();
        status += L" | Click to return to menu";
    }
    else if (g_ai_thinking) {
        status = L"AI thinking...";
    }
    else {
        status = (g_current_turn == Stone::BLACK) ? L"Turn: Black" : L"Turn: White";
        status += L" | LeftClick: Move | RightClick: Pass";
    }
    DrawText(hdc, status.c_str(), -1, &rc, DT_LEFT | DT_TOP);

    SelectObject(hdc, oldf);
    DeleteObject(hf);
}

// NEW: Menu screen
void DrawMenuScreen(HDC hdc) {
    RECT rc;
    GetClientRect(g_hWnd, &rc);
    int cx = (rc.left + rc.right) / 2;
    int cy = (rc.top + rc.bottom) / 2;

    FillRect(hdc, &rc, (HBRUSH)GetStockObject(WHITE_BRUSH));

    SetTextColor(hdc, C_TEXT);
    SetBkMode(hdc, TRANSPARENT);

    HFONT hTitle = CreateFont(36, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE,
        DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
        DEFAULT_QUALITY, DEFAULT_PITCH | FF_SWISS, L"Microsoft Sans Serif");
    HFONT oldf = (HFONT)SelectObject(hdc, hTitle);
    RECT rTitle = { rc.left, cy - 100, rc.right, cy - 60 };
    DrawText(hdc, L"Go 9x9", -1, &rTitle, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
    SelectObject(hdc, oldf);
    DeleteObject(hTitle);

    HFONT hf = CreateFont(18, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
        DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
        DEFAULT_QUALITY, DEFAULT_PITCH | FF_SWISS, L"Microsoft Sans Serif");
    oldf = (HFONT)SelectObject(hdc, hf);

    RECT rBlack = { cx - 130, cy - 20, cx + 130, cy + 20 };
    HBRUSH br = CreateSolidBrush(RGB(240, 240, 240));
    FillRect(hdc, &rBlack, br);
    DeleteObject(br);
    HPEN pen = CreatePen(PS_SOLID, 2, RGB(0, 0, 0));
    HGDIOBJ oldp = SelectObject(hdc, pen);
    Rectangle(hdc, rBlack.left, rBlack.top, rBlack.right, rBlack.bottom);
    SelectObject(hdc, oldp);
    DeleteObject(pen);
    DrawText(hdc, L"Play as Black (First move)", -1, &rBlack, DT_CENTER | DT_VCENTER | DT_SINGLELINE);

    RECT rWhite = { cx - 130, cy + 35, cx + 130, cy + 75 };
    br = CreateSolidBrush(RGB(240, 240, 240));
    FillRect(hdc, &rWhite, br);
    DeleteObject(br);
    pen = CreatePen(PS_SOLID, 2, RGB(0, 0, 0));
    oldp = SelectObject(hdc, pen);
    Rectangle(hdc, rWhite.left, rWhite.top, rWhite.right, rWhite.bottom);
    SelectObject(hdc, oldp);
    DeleteObject(pen);
    DrawText(hdc, L"Play as White (Second move)", -1, &rWhite, DT_CENTER | DT_VCENTER | DT_SINGLELINE);

    SelectObject(hdc, oldf);
    DeleteObject(hf);
}

// ===================== Game Logic =====================
void CheckGameOver() {
    if (g_pass_count >= 2) {
        g_game_over = true;
        g_gameState = GameState::GAME_OVER;
    }
}

void DoAIMove() {
    if (g_game_over || g_gameState != GameState::PLAYING) return;
    if (g_current_turn == g_human_color) return;

    g_ai_thinking = true;
    InvalidateRect(g_hWnd, NULL, FALSE);
    UpdateWindow(g_hWnd);

    Sleep(400);

    std::pair<int, int> mv = g_ai.select_move(g_board, g_current_turn);
    int x = mv.first;
    int y = mv.second;

    if (x == GoBoard::PASS_MOVE) {
        g_pass_count++;
        CheckGameOver();
    }
    else {
        bool ok = g_board.play_move(x, y, g_current_turn);
        if (ok) {
            g_last_x = x;
            g_last_y = y;
            g_pass_count = 0;
        }
    }

    g_current_turn = opponent_of(g_current_turn);
    g_ai_thinking = false;
    InvalidateRect(g_hWnd, NULL, FALSE);
}

void HandleLeftClick(int px, int py) {
    if (g_gameState != GameState::PLAYING || g_game_over || g_ai_thinking) return;
    if (g_current_turn != g_human_color) return;

    int bx = 0, by = 0;
    if (!ToBoardCoord(px, py, bx, by)) return;

    if (!g_board.is_legal(bx, by, g_human_color)) return;

    bool ok = g_board.play_move(bx, by, g_human_color);
    if (!ok) return;

    g_last_x = bx;
    g_last_y = by;
    g_pass_count = 0;
    g_current_turn = opponent_of(g_current_turn);
    InvalidateRect(g_hWnd, NULL, FALSE);

    if (!g_game_over) {
        SetTimer(g_hWnd, 1, 10, NULL);
    }
}

void HandleRightClick() {
    if (g_gameState != GameState::PLAYING || g_game_over || g_ai_thinking) return;
    if (g_current_turn != g_human_color) return;

    g_board.play_move(GoBoard::PASS_MOVE, GoBoard::PASS_MOVE, g_human_color);
    g_pass_count++;
    CheckGameOver();
    g_current_turn = opponent_of(g_current_turn);
    InvalidateRect(g_hWnd, NULL, FALSE);

    if (!g_game_over) {
        SetTimer(g_hWnd, 1, 10, NULL);
    }
}

// NEW: Menu click
void HandleMenuClick(int px, int py) {
    RECT rc;
    GetClientRect(g_hWnd, &rc);
    int cx = (rc.left + rc.right) / 2;
    int cy = (rc.top + rc.bottom) / 2;

    RECT rBlack = { cx - 130, cy - 20, cx + 130, cy + 20 };
    RECT rWhite = { cx - 130, cy + 35, cx + 130, cy + 75 };
    POINT pt = { px, py };

    if (PtInRect(&rBlack, pt)) {
        g_human_color = Stone::BLACK;
        g_board = GoBoard();
        g_current_turn = Stone::BLACK;
        g_game_over = false;
        g_pass_count = 0;
        g_last_x = -1; g_last_y = -1;
        g_hoverX = -1; g_hoverY = -1;
        g_gameState = GameState::PLAYING;
        SetWindowText(g_hWnd, L"Go 9x9 (You: Black, AI: White)");
        InvalidateRect(g_hWnd, NULL, TRUE);
    }
    else if (PtInRect(&rWhite, pt)) {
        g_human_color = Stone::WHITE;
        g_board = GoBoard();
        g_current_turn = Stone::BLACK;
        g_game_over = false;
        g_pass_count = 0;
        g_last_x = -1; g_last_y = -1;
        g_hoverX = -1; g_hoverY = -1;
        g_gameState = GameState::PLAYING;
        SetWindowText(g_hWnd, L"Go 9x9 (You: White, AI: Black)");
        SetTimer(g_hWnd, 1, 500, NULL); // AI moves first
        InvalidateRect(g_hWnd, NULL, TRUE);
    }
}

// ===================== Window Procedure =====================
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_CREATE:
        g_hWnd = hWnd;
        return 0;

    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hWnd, &ps);
        if (g_gameState == GameState::MENU) {
            DrawMenuScreen(hdc);
        }
        else {
            DrawBoard(hdc);
        }
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
            else {
                if (g_hoverX != -1) {
                    g_hoverX = -1;
                    g_hoverY = -1;
                    InvalidateRect(hWnd, NULL, FALSE);
                }
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
            InvalidateRect(hWnd, NULL, TRUE);
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