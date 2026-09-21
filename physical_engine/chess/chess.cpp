enum Stone { EMPTY = 0, BLACK = 1, WHITE = 2 };
const int BOARD_SIZE = 9;
const float KOMI = 6.5f;

// 盤は1次元配列で管理: idx = y * BOARD_SIZE + x
int board[BOARD_SIZE * BOARD_SIZE];


int parent[BOARD_SIZE * BOARD_SIZE];
int liberties[BOARD_SIZE * BOARD_SIZE];  // 連の気の数（ルートのみ有効）
int stone_count[BOARD_SIZE * BOARD_SIZE]; // 連の石数（ルートのみ有効）

int find(int x) {
    if (parent[x] == x) return x;
    parent[x] = find(parent[x]);
    return parent[x];
}

void unite(int a, int b) {
    a = find(a), b = find(b);
    if (a == b) return;
    if (a > b) std::swap(a, b);
    parent[b] = a;
    liberties[a] += liberties[b];
    stone_count[a] += stone_count[b];
}

void reset_groups() {
    for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
        parent[i] = i;
        liberties[i] = 0;
        stone_count[i] = (board[i] == EMPTY) ? 0 : 1;
    }
    // 同じ色の隣接石を統合
    for (int y = 0; y < BOARD_SIZE; y++)
        for (int x = 0; x < BOARD_SIZE; x++) {
            int idx = y * BOARD_SIZE + x;
            if (board[idx] == EMPTY) continue;
            const int dx[4] = {1,-1,0,0};
            const int dy[4] = {0,0,1,-1};
            for (int d = 0; d < 4; d++) {
                int nx = x + dx[d], ny = y + dy[d];
                if (nx < 0 || nx >= BOARD_SIZE || ny < 0 || ny >= BOARD_SIZE) continue;
                int nidx = ny * BOARD_SIZE + nx;
                if (board[nidx] == board[idx]) unite(idx, nidx);
            }
        }
    // 各空点が隣接する連の気を+1
    for (int y = 0; y < BOARD_SIZE; y++)
        for (int x = 0; x < BOARD_SIZE; x++) {
            int idx = y * BOARD_SIZE + x;
            if (board[idx] != EMPTY) continue;
            std::set<int> roots;
            for (int d = 0; d < 4; d++) {
                int nx = x + dx[d], ny = y + dy[d];
                if (nx < 0 || nx >= BOARD_SIZE || ny < 0 || ny >= BOARD_SIZE) continue;
                int nidx = ny * BOARD_SIZE + nx;
                if (board[nidx] != EMPTY) roots.insert(find(nidx));
            }
            for (int r : roots) liberties[r]++;
        }
}

int ko_point = -1;  // グローバル

bool is_legal(int x, int y, Stone color) {
    int idx = y * BOARD_SIZE + x;
    if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) return false;
    if (board[idx] != EMPTY) return false;
    if (idx == ko_point) return false;  // コウ
    
    // 仮に置いてみて自殺手かチェック
    int captured, ko;
    Stone opponent = (color == BLACK) ? WHITE : BLACK;
    int old_board[BOARD_SIZE * BOARD_SIZE];
    std::copy(board, board + BOARD_SIZE * BOARD_SIZE, old_board);
    
    bool ok = place_stone(x, y, color, captured, ko);
    if (!ok) {
        std::copy(old_board, old_board + BOARD_SIZE * BOARD_SIZE, board);
        return false;
    }
    std::copy(old_board, old_board + BOARD_SIZE * BOARD_SIZE, board);
    return true;
}

struct Score {
    float black;
    float white;
};

Score calculate_score() {
    std::vector<bool> visited(BOARD_SIZE * BOARD_SIZE, false);
    float black_territory = 0, white_territory = 0;
    int black_stones = 0, white_stones = 0;
    
    for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
        if (board[i] == BLACK) black_stones++;
        if (board[i] == WHITE) white_stones++;
    }
    
    for (int y = 0; y < BOARD_SIZE; y++)
        for (int x = 0; x < BOARD_SIZE; x++) {
            int idx = y * BOARD_SIZE + x;
            if (board[idx] != EMPTY || visited[idx]) continue;
            
            // BFSで空領域を探索
            std::vector<int> region;
            std::queue<int> q;
            q.push(idx); visited[idx] = true;
            bool black_adj = false, white_adj = false;
            
            while (!q.empty()) {
                int cur = q.front(); q.pop();
                region.push_back(cur);
                int cx = cur % BOARD_SIZE, cy = cur / BOARD_SIZE;
                for (int d = 0; d < 4; d++) {
                    int nx = cx + dx[d], ny = cy + dy[d];
                    if (nx < 0 || nx >= BOARD_SIZE || ny < 0 || ny >= BOARD_SIZE) continue;
                    int nidx = ny * BOARD_SIZE + nx;
                    if (board[nidx] == EMPTY && !visited[nidx]) {
                        visited[nidx] = true;
                        q.push(nidx);
                    } else if (board[nidx] == BLACK) black_adj = true;
                    else if (board[nidx] == WHITE) white_adj = true;
                }
            }
            
            if (black_adj && !white_adj) black_territory += region.size();
            else if (white_adj && !black_adj) white_territory += region.size();
        }
    
    Score s;
    s.black = black_territory + black_stones;
    s.white = white_territory + white_stones + KOMI;
    return s;
}

std::pair<int,int> get_random_move(Stone color, StdRng& rng) {
    std::vector<std::pair<int,int>> legals;
    for (int y = 0; y < BOARD_SIZE; y++)
        for (int x = 0; x < BOARD_SIZE; x++)
            if (is_legal(x, y, color))
                legals.push_back({x, y});
    
    if (legals.empty()) return {-1, -1}; // パス
    std::uniform_int_distribution<int> dist(0, legals.size() - 1);
    return legals[dist(rng)];
}

