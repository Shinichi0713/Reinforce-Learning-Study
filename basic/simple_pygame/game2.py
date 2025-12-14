
import sys, time, os, random
import pygame
from pygame.locals import *

""" グローバル変数 """
gravity = 1
is_collide =False

def make_screen():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    return screen

def update_screen(screen):
    screen.fill((120, 255, 255))

# 味方のクラス
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.image.load(f'{os.path.dirname(__file__)}/goomba.jpg')
        self.image = pygame.transform.scale(self.image, (80, 80))
        self.rect = self.image.get_rect()
        self.rect.topleft = (0, 300)

        # 速度の初期化
        self.vx = 0
        self.vy = 0
        self.velocity_x = 0

        # 速度のリミット
        self.vx_limit = 5
        self.vy_limit = 5

        # 重力更新のインターバル
        self.inverval_gravity = 30
        self.count_interval_gravity = 0

        # 摩擦係数
        self.coefficient_friction = 0.1
        # スコア
        self.score = 0

    def update(self, screen):
        width = screen.get_width()
        height = screen.get_height()
        keys = pygame.key.get_pressed()

        if keys[K_LEFT] and abs(self.vx) < self.vx_limit:
            self.vx = -1
        if keys[K_RIGHT] and abs(self.vx) < self.vx_limit:
            self.vx = 1
        
        # 摩擦を加える
        if self.rect.bottom >= height:
            self.velocity_x = abs(self.vx) - self.coefficient_friction * abs(self.vx)
        if self.vx > 0:
            self.vx = self.velocity_x
        elif self.vx < 0:
            self.vx = -self.velocity_x
        if keys[K_UP] and self.rect.bottom >= height:
            self.vy = -5
            self.count_interval_gravity = 0
        
        # 重力を加える
        if self.rect.bottom < height and self.count_interval_gravity>=self.inverval_gravity:
            self.vy += gravity
            self.count_interval_gravity = 0
        self.count_interval_gravity += 1

        # 画面からはみ出さないようにする
        if self.rect.left < 0:
            self.vx = 0
            self.rect.left = 0
        if self.rect.right > width:
            self.vx = 0
            self.rect.right = width
        if self.rect.bottom >= height and self.vy >= 0:
            self.vy = 0
            self.rect.bottom = height
            
        # 座標の更新
        self.rect.move_ip(int(round(self.vx, 0)), int(round(self.vy, 0)))
        screen.blit(self.image, self.rect)

class Enemy(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.image.load(f'{os.path.dirname(__file__)}/mario.jpg')
        self.image = pygame.transform.scale(self.image, (80, 80))
        self.rect = self.image.get_rect()
        self.rect.topleft = (300, 300)

        # 速度の初期化
        self.vx = 0
        self.vy = 0
        self.velocity_x = 0

        # 速度のリミット
        self.vx_limit = 5
        self.vy_limit = 5

        # 重力更新のインターバル
        self.inverval_gravity = 30
        self.count_interval_gravity = 0

        # 摩擦係数
        self.coefficient_friction = 0.1

        self.action_table = {
            0: "move_left",
            1: "move_right",
            2: "jump",
            3: "stop",
            4: "stop",
            5: "stop",
            6: "stop",
        }
        self.score = 0

    def update(self, screen):
        width = screen.get_width()
        height = screen.get_height()
        self.action = random.choice([0, 1, 2, 3, 4, 5, 6])

        if self.action_table[self.action] == "move_left" and abs(self.vx) < self.vx_limit:
            self.vx = -1
        if self.action_table[self.action] == "move_right" and abs(self.vx) < self.vx_limit:
            self.vx = 1
        # 摩擦を加える
        if self.rect.bottom >= height:
            self.velocity_x = abs(self.vx) - self.coefficient_friction * abs(self.vx)
        else:
            self.velocity_x = abs(self.vx)
        if self.vx > 0:
            self.vx = self.velocity_x
        elif self.vx < 0:
            self.vx = -self.velocity_x
        if self.action_table[self.action] == "jump" and self.rect.bottom >= height:
            self.vy = -5
            self.count_interval_gravity = 0
        
        # 重力を加える
        if self.rect.bottom < height and self.count_interval_gravity>=self.inverval_gravity:
            self.vy += gravity
            self.count_interval_gravity = 0
        self.count_interval_gravity += 1

        # 画面からはみ出さないようにする
        if self.rect.left < 0:
            self.vx = 0
            self.rect.left = 0
        if self.rect.right > width:
            self.vx = 0
            self.rect.right = width
        if self.rect.bottom >= height and self.vy >= 0:
            self.vy = 0
            self.rect.bottom = height
            
        # 座標の更新
        self.rect.move_ip(int(round(self.vx, 0)), int(round(self.vy, 0)))


        # 画面更新
        screen.blit(self.image, self.rect)

# 味方と敵の接触判定→動作
def check_collision(player, enemy):
    global is_collide
    collide = pygame.sprite.collide_rect(player, enemy)
    if collide:
        # 衝突した座標を取得
        collision_rect = player.rect.clip(enemy.rect)
        if collision_rect.width > collision_rect.height:
            if player.rect.bottom == collision_rect.bottom:
                position_vertical = (player.rect.bottom + enemy.rect.top) // 2
                player.rect.bottom = position_vertical
                enemy.rect.top = position_vertical
                # 敵を踏みつける
                if player.vy - enemy.vy >= 2:
                    player.score += 1
                velocity_vertival = (player.vy + enemy.vy) // 2
                player.vy = velocity_vertival
                enemy.vy = velocity_vertival
                return
            elif player.rect.top == collision_rect.top:
                position_vertical = (player.rect.top + enemy.rect.bottom) // 2
                player.rect.top = position_vertical
                enemy.rect.bottom = position_vertical
                # 見方が踏みつける
                if player.vy - enemy.vy <= -2:
                    enemy.score += 1
                velocity_vertival = (player.vy + enemy.vy) // 2
                player.vy = velocity_vertival
                enemy.vy = velocity_vertival
                return
        else:
            if player.rect.right == collision_rect.right:
                position_horizontal = (player.rect.right + enemy.rect.left) // 2
                player.rect.right = position_horizontal
                enemy.rect.left = position_horizontal
                player.vx = -player.velocity_x
                enemy.vx = enemy.velocity_x
                return
            elif player.rect.left == collision_rect.left:
                position_horizontal = (player.rect.left + enemy.rect.right) // 2
                player.rect.left = position_horizontal
                enemy.rect.right = position_horizontal
                player.vx = player.velocity_x
                enemy.vx = -enemy.velocity_x
                return
        is_collide = True
    elif not collide:
        is_collide = False

def display_score(screen, player, enemy):
    font = pygame.font.Font(None, 36)
    text = font.render(f"Player: {player.score}, Enemy: {enemy.score}", True, (0, 0, 0))
    screen.blit(text, (0, 0))


if __name__ == "__main__":
    screen = make_screen()
    player = Player()
    enemy = Enemy()
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
        update_screen(screen)
        
        # キャラクターの更新
        player.update(screen)
        enemy.update(screen)
        check_collision(player, enemy)
        screen.blit(player.image, player.rect)
        screen.blit(enemy.image, enemy.rect)
        
        display_score(screen, player, enemy)

        pygame.display.update()
        time.sleep(0.001)
        
