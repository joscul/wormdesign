#!/usr/bin/env python3
"""
caterpillar.py — Pixel-art style animated caterpillar, 9 round segments.
Requirements:  pip install pygame
Press  Q / Esc  to quit.
"""

import math, sys
import pygame

WIN_W, WIN_H = 900, 400
FPS          = 60
BG_COLOR     = (255, 255, 255)

N_SECTIONS  = 9
SPACING     = 52

WAVE_AMP    = 0.30
WAVE_FREQ   = 0.20
PHASE_STEP  = 0.70

# Radii — head slightly smaller, body balls nice and round, tail tapers a tiny bit
SEG_R = [26, 28, 28, 28, 27, 27, 26, 25, 22]

# Pixel-art greens: dark outline, mid fill, light square highlight
COLOR_DARK  = ( 30, 110,  30)   # outline / shadow
COLOR_MID   = ( 80, 185,  60)   # main fill
COLOR_LIGHT = (120, 220,  90)   # lighter band
COLOR_HI    = (180, 240, 130)   # highlight square

# ════════════════════════════════════════════════════════════════════════════
def make_sections(t):
    x, y, ang = 0.0, 0.0, 0.0
    pts = [(x, y, ang)]
    for i in range(N_SECTIONS - 1):
        wave = WAVE_AMP * math.sin(2 * math.pi * WAVE_FREQ * t - i * PHASE_STEP)
        ang += wave
        x += math.cos(ang) * SPACING
        y += math.sin(ang) * SPACING
        pts.append((x, y, ang))
    mx = sum(p[0] for p in pts) / N_SECTIONS
    my = sum(p[1] for p in pts) / N_SECTIONS
    ox, oy = WIN_W/2 - mx, WIN_H/2 - my
    return [(p[0]+ox, p[1]+oy, p[2]) for p in pts]

# ════════════════════════════════════════════════════════════════════════════
def draw_pixel_circle(surf, cx, cy, r, col_dark, col_mid, col_light, col_hi):
    """Draw a chunky pixel-art style ball."""
    icx, icy = int(cx), int(cy)

    # Dark outline ring
    pygame.draw.circle(surf, col_dark, (icx, icy), r + 2)

    # Main fill
    pygame.draw.circle(surf, col_mid, (icx, icy), r)

    # Lighter crescent on upper-left (~2/3 radius)
    lighter_surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
    pygame.draw.circle(lighter_surf, (*col_light, 180),
                       (icx - r//5, icy - r//6), int(r * 0.72))
    # Mask back to circle
    mask = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
    pygame.draw.circle(mask, (255,255,255,255), (icx, icy), r)
    lighter_surf.blit(mask, (0,0), special_flags=pygame.BLEND_RGBA_MULT)
    surf.blit(lighter_surf, (0,0))

    # Pixel highlight square — top-left quadrant, pixelated
    hi_size = max(4, r // 3)
    hx = icx - r//3
    hy = icy - r//3
    # Snap to pixel grid (gives that chunky pixel-art feel)
    hx = (hx // 2) * 2
    hy = (hy // 2) * 2
    pygame.draw.rect(surf, col_hi, (hx, hy, hi_size, hi_size))


def draw_head(surf, sec, anim_t):
    cx, cy, ang = sec
    r  = SEG_R[0]
    # Flip 180°: head faces away from the body, not into it
    tx = -math.cos(ang)
    ty = -math.sin(ang)
    nx = -ty
    ny =  tx
    icx, icy = int(cx), int(cy)

    # Head ball
    draw_pixel_circle(surf, cx, cy, r, COLOR_DARK, COLOR_MID, COLOR_LIGHT, COLOR_HI)

    # Eyes — two black dots with white glint
    for side in (-1, +1):
        ex = int(cx + tx * r * 0.35 + nx * side * r * 0.42)
        ey = int(cy + ty * r * 0.35 + ny * side * r * 0.42)
        pygame.draw.rect(surf, (20, 20, 20), (ex-4, ey-4, 8, 8))   # pixel square eye
        pygame.draw.rect(surf, (255,255,255), (ex-1, ey-3, 3, 3))   # glint

    # Mouth — small red rectangle
    mx2 = int(cx + tx * r * 0.70)
    my2 = int(cy + ty * r * 0.70)
    pygame.draw.rect(surf, (200, 50, 50), (mx2 - 4, my2 - 2, 8, 4))

    # Antennae — two pixel-art lines + dots on head (segment 0 is head)
    for side in (-1, +1):
        # Base of antenna: top of head
        bx = int(cx - tx * r * 0.15 + nx * side * r * 0.30)
        by_ = int(cy - ty * r * 0.15 + ny * side * r * 0.30)
        # Tip: slightly animated bob
        bob = math.sin(2 * math.pi * 1.2 * anim_t + side) * 3
        tip_x = bx + int(-ty * side * 10 - tx * 14 + bob)
        tip_y = by_ + int( tx * side * 10 - ty * 14 + bob)
        # Draw as staircase line (pixel art feel)
        pygame.draw.line(surf, COLOR_DARK, (bx, by_), (tip_x, tip_y), 3)
        # Antenna tip dot
        pygame.draw.rect(surf, COLOR_DARK, (tip_x - 3, tip_y - 3, 6, 6))
        pygame.draw.rect(surf, (20, 20, 20), (tip_x - 2, tip_y - 2, 4, 4))


# ════════════════════════════════════════════════════════════════════════════
def draw_snake(screen, sections, anim_t):
    # Connectors first (small dark overlap between balls)
    for i in range(N_SECTIONS - 1):
        ax, ay, _ = sections[i]
        bx, by, _ = sections[i+1]
        mx2 = (ax + bx) / 2
        my2 = (ay + by) / 2
        pygame.draw.circle(screen, COLOR_DARK, (int(mx2), int(my2)), 6)

    # Balls back-to-front
    for i in range(N_SECTIONS - 1, 0, -1):
        cx, cy, ang = sections[i]
        draw_pixel_circle(screen, cx, cy, SEG_R[i],
                          COLOR_DARK, COLOR_MID, COLOR_LIGHT, COLOR_HI)

    # Head last (on top)
    draw_head(screen, sections[0], anim_t)


# ════════════════════════════════════════════════════════════════════════════
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("caterpillar")
    clock  = pygame.time.Clock()

    t  = 0.0
    dt = 1.0 / FPS

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN and ev.key in (pygame.K_q, pygame.K_ESCAPE):
                pygame.quit(); sys.exit()

        screen.fill(BG_COLOR)
        secs = make_sections(t)
        draw_snake(screen, secs, anim_t=t)
        pygame.display.flip()
        clock.tick(FPS)
        t += dt

if __name__ == "__main__":
    main()