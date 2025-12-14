
import sys, time, os
import pygame
from pygame.locals import *

os.chdir(os.path.dirname(__file__))
px = 0
py = 300
vy = 0      # 速度
pygame.init()
sc = pygame.display.set_mode((800, 600))
width_sc, height_sc = sc.get_size() # 画面サイズ

# 重力加速度
gravity = -0.0001
offset = 0

# クリボーの画像をロード
goomba_image = pygame.image.load('goomba.jpg')
goomba_image = pygame.transform.scale(goomba_image, (80, 80))
goomba_rect = goomba_image.get_rect()
print(goomba_rect)

def event():
    global px, py, sc, vy

    x_size, y_size = sc.get_size()
    # イベント処理
    for event in pygame.event.get():  # イベントを取得
        if event.type == pygame.QUIT:        # 閉じるボタンが押されたら
            pygame.quit()             
            sys.exit()                # 終了
    keys = pygame.key.get_pressed()
    if keys[K_LEFT] and px > -x_size // 2:
        px -= 1
    if keys[K_RIGHT] and px < x_size // 2:
        px += 1
    if keys[K_SPACE] and py == 0:
        vy += 0.2
    # オブジェクトy方向の座標
    py, vy = calculate_height(py, vy)


def calculate_height(py, vy):
    # 速度を計算
    vy = vy + gravity
    # 座標を計算
    py = py + vy
    if py < 0:
        py = 0
        vy = 0
    return py, vy


def make_object(sc, px, py):
    width = 50
    height = 100

    # 頭を描く（円）
    pygame.draw.circle(sc, (0, 255, 0), (px, py-120), int(width/3))

    # 体を描く（長方形）
    pygame.draw.rect(sc, (0, 255, 0), (px-int(width/3), py-100, int(width/1.5), 50))

    # 腕を描く（長方形）
    pygame.draw.rect(sc, (0, 255, 0), (px-width/2-width, py-100, width, 10))
    pygame.draw.rect(sc, (0, 255, 0), (px+width/2, py-100, width, 10))

    # 脚を描く（長方形）
    pygame.draw.rect(sc, (0, 255, 0), (px-20, py-50, 10, 50))
    pygame.draw.rect(sc, (0, 255, 0), (px+10, py-50, 10, 50))
    
    # あたり判定
    return pygame.Rect(px-int(width/3), py-120, int(width/1.5), height)


def make_goomba(sc, px, py, count):
    if count % 20 == 0:
        if px + 350 > goomba_rect.x + 10:
            goomba_rect.x += 1
        elif px + 350 < goomba_rect.x - 10:
            goomba_rect.x -= 1
    goomba_rect.y = height_sc - goomba_rect.height
    sc.blit(goomba_image, goomba_rect)

count = 0
while True:
    sc.fill((255, 255, 255))
    # pygame.draw.circle(sc, (0, 255, 0), (400 + px,  height_sc-py), 20)
    make_goomba(sc, px, py, count)
    player_rects = make_object(sc, 400 + px, height_sc-py)
    
    if player_rects.colliderect(goomba_rect):
        print('hit')
        font = pygame.font.SysFont(None, 75)
        game_over_text = font.render("Game Over", True, (255, 0, 0))
        sc.blit(game_over_text, (250, 250))

    pygame.display.update()
    event()
    count += 1
    # print(px, py)