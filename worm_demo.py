#!/usr/bin/env python3
"""
worm_demo.py — Standalone animated worm (no CUDA required)

Matches the visual style of the physevol simulation:
  dark agar-plate background, colour-gradient segments, eyes + smile + cheeks,
  bioluminescent glow, joint connectors, undulatory sine-wave motion.

Requirements:  pip install pygame numpy
Python 3.8+, Pygame 2.0+

Press  Q / Esc  to quit.
"""

import math, random, sys
import numpy as np
import pygame

# ════════════════════════════════════════════════════════════════════════════
# DESIGN PARAMETERS  ← edit these
# ════════════════════════════════════════════════════════════════════════════

# -- Window ----------------------------------------------------------------
SCALE   = 0.5           # 1.0 = full 1080×1920;  0.5 = 540×960 (fits most screens)
FPS     = 60

# -- Worm body -------------------------------------------------------------
NUM_SEGMENTS = 9
HW           = 32       # segment half-width  (along body axis)
HH           = 30       # segment half-height (across body axis)
JOINT_AX     = HW + 9  # joint attachment distance from segment centre

# Colour gradient: segment 0 (head) → segment 8 (tail)
SEG_COLORS = [
    (0x22, 0xBB, 0x44),   # 0  head:  green
    (0xEE, 0xCC, 0x00),   # 1         yellow
    (0xFF, 0x88, 0x00),   # 2         orange
    (0xEE, 0x44, 0x00),   # 3         red-orange
    (0xCC, 0x22, 0x44),   # 4         red
    (0xAA, 0x22, 0xAA),   # 5         purple
    (0x77, 0x44, 0xCC),   # 6         violet
    (0x44, 0x88, 0xEE),   # 7         light blue
    (0x22, 0x55, 0xDD),   # 8  tail:  blue
]

# -- Glow ------------------------------------------------------------------
GLOW_EXPAND = 14        # pixels beyond segment edge (world units)
GLOW_ALPHA  = 90        # 0–255 (lower = more transparent)

# -- Joint connectors ------------------------------------------------------
JOINT_COLOR = (80, 180, 200)
JOINT_WIDTH = 5         # world pixels

# -- Head face -------------------------------------------------------------
# All fractions are relative to HW (x) or HH (y)
EYE_Y_FRAC   = -0.18   # eye centre y  (negative = toward forward tip)
EYE_X_FRAC   =  0.33   # eye centre ±x
EYE_R_FRAC   =  0.22   # eyeball radius
PUP_R_FRAC   =  0.10   # pupil radius
HI_R_FRAC    =  0.06   # highlight dot radius

SMILE_CY_FRAC = 0.05   # smile-arc centre y offset
SMILE_R_FRAC  = 0.52   # smile-arc radius
SMILE_T_FRAC  = 0.085  # smile stroke thickness

CHEEK_Y_FRAC  = 0.18
CHEEK_X_FRAC  = 0.52
CHEEK_R_FRAC  = 0.28
CHEEK_ALPHA   = 55      # 0–255

# -- Animation -------------------------------------------------------------
WAVE_AMP    = 0.45      # max joint angle (radians, ~26°)
WAVE_FREQ   = 1.2       # full cycles per second
PHASE_STEP  = 0.75      # phase lag per joint (radians)

# -- Background ------------------------------------------------------------
VIGNETTE_STR = 3.2      # higher = darker corners
N_DOTS       = 2500     # small stipple dots
N_BLOBS      = 250      # larger agar blobs

# ════════════════════════════════════════════════════════════════════════════
# DERIVED CONSTANTS  (don't edit)
# ════════════════════════════════════════════════════════════════════════════

WORLD_W = 1080
WORLD_H = 1920
WIN_W   = int(WORLD_W * SCALE)
WIN_H   = int(WORLD_H * SCALE)

def _s(v):
    """World pixel value → screen pixel (int, min 1)."""
    return max(1, int(v * SCALE + 0.5))

def _sp(wx, wy):
    """World-space point → screen pixel centre."""
    return int(WIN_W / 2 + wx * SCALE), int(WIN_H / 2 + wy * SCALE)

# ════════════════════════════════════════════════════════════════════════════
# BACKGROUND
# ════════════════════════════════════════════════════════════════════════════

def make_background():
    """Pre-render the static agar-plate background."""
    # Build per-pixel colour with numpy (grid + vignette)
    px = np.arange(WIN_W, dtype=np.float32)
    py = np.arange(WIN_H, dtype=np.float32)
    FX, FY = np.meshgrid(px / SCALE, py / SCALE)   # world coords

    R = np.full((WIN_H, WIN_W), 7.0,  dtype=np.float32)
    G = np.full((WIN_H, WIN_W), 11.0, dtype=np.float32)
    B = np.full((WIN_H, WIN_W), 22.0, dtype=np.float32)

    # Minor grid every 60 world-px
    gx = np.mod(FX + 1e6, 60.0); gy = np.mod(FY + 1e6, 60.0)
    lw = np.minimum(gx, np.minimum(60-gx, np.minimum(gy, 60-gy)))
    a  = np.clip(1.0 - lw * 0.9, 0, 1)
    R += a * 5;  G += a * 7;  B += a * 12

    # Major grid every 300 world-px
    gx = np.mod(FX + 1e6, 300.0); gy = np.mod(FY + 1e6, 300.0)
    lw = np.minimum(gx, np.minimum(300-gx, np.minimum(gy, 300-gy)))
    a  = np.clip(1.0 - lw * 0.55, 0, 1)
    R += a * 14; G += a * 18; B += a * 28

    # Vignette
    vx = px / WIN_W - 0.5; vy = py / WIN_H - 0.5
    VX, VY = np.meshgrid(vx, vy)
    v  = np.clip(1.0 - (VX*VX + VY*VY) * VIGNETTE_STR, 0, 1)
    R *= v;  G *= v;  B *= (0.65 + 0.35*v)

    rgb = np.stack([np.clip(R,0,255).astype(np.uint8),
                    np.clip(G,0,255).astype(np.uint8),
                    np.clip(B,0,255).astype(np.uint8)], axis=2)

    surf = pygame.Surface((WIN_W, WIN_H))
    pygame.surfarray.blit_array(surf, rgb.transpose(1, 0, 2))

    # Stipple dots
    rng = random.Random(42)
    for _ in range(N_DOTS):
        x = rng.random() * WIN_W
        y = rng.random() * WIN_H
        r = _s(0.9 + rng.random() * 1.4)
        pygame.draw.circle(surf, (25, 35, 60), (int(x), int(y)), r)

    # Larger agar blobs
    for _ in range(N_BLOBS):
        x  = rng.random() * WIN_W
        y  = rng.random() * WIN_H
        rx = _s(5 + rng.random() * 12)
        ry = max(1, int(rx * (0.5 + rng.random() * 1.0)))
        blob = pygame.Surface((rx*2+1, ry*2+1), pygame.SRCALPHA)
        pygame.draw.ellipse(blob, (20, 32, 57, 90), blob.get_rect())
        surf.blit(blob, (int(x-rx), int(y-ry)))

    return surf

# ════════════════════════════════════════════════════════════════════════════
# KINEMATICS
# ════════════════════════════════════════════════════════════════════════════

def worm_positions(t):
    """
    Compute (world_x, world_y, angle_rad) for each segment at time t.
    Segment 0 is the head (forward in the +x direction when angle=0).
    The chain is centred at the origin each frame.
    """
    x, y, ang = 0.0, 0.0, 0.0
    segs = [(x, y, ang)]

    for i in range(NUM_SEGMENTS - 1):
        ja  = WAVE_AMP * math.sin(2.0 * math.pi * WAVE_FREQ * t - i * PHASE_STEP)
        jx  = x + math.cos(ang) * JOINT_AX
        jy  = y + math.sin(ang) * JOINT_AX
        ang = ang + ja
        x   = jx + math.cos(ang) * JOINT_AX
        y   = jy + math.sin(ang) * JOINT_AX
        segs.append((x, y, ang))

    mx = sum(s[0] for s in segs) / NUM_SEGMENTS
    my = sum(s[1] for s in segs) / NUM_SEGMENTS
    return [(s[0]-mx, s[1]-my, s[2]) for s in segs]

# ════════════════════════════════════════════════════════════════════════════
# RENDERING
# ════════════════════════════════════════════════════════════════════════════

def _seg_surface(color, with_face=False, seg_ang=0.0):
    """
    Build a local-coords RGBA surface for one segment (64×60 world px, scaled).
    The segment long axis is horizontal (lx), short axis is vertical (ly).
    """
    w = _s(HW * 2)
    h = _s(HH * 2)
    surf = pygame.Surface((w, h), pygame.SRCALPHA)

    cr = max(2, _s(HW * 0.28))   # corner radius

    # Dark border (the SDF edge-darkening effect)
    dark = tuple(max(0, int(c * 0.45)) for c in color)
    pygame.draw.rect(surf, (*dark, 255), surf.get_rect(), border_radius=cr)

    # Main fill — slightly inset for the border effect
    pad = max(1, _s(2.5))
    inner = pygame.Rect(pad, pad, w - 2*pad, h - 2*pad)
    pygame.draw.rect(surf, (*color, 255), inner,
                     border_radius=max(1, cr - pad))

    if with_face:
        _draw_face(surf, w, h, color)

    return surf


def _draw_face(surf, w, h, body_color):
    """Draw eyes, smile, and rosy cheeks onto a segment surface (local coords)."""
    shw, shh = w / 2, h / 2   # screen half extents

    # Cheeks  (drawn first, under eyes)
    for sign in (-1, +1):
        cx = int(shw + sign * CHEEK_X_FRAC * shw)
        cy = int(shh + CHEEK_Y_FRAC * shh)
        cr = max(1, int(CHEEK_R_FRAC * shw))
        chk = pygame.Surface((cr*2, cr*2), pygame.SRCALPHA)
        pygame.draw.circle(chk, (220, 80, 100, CHEEK_ALPHA), (cr, cr), cr)
        surf.blit(chk, (cx - cr, cy - cr))

    # Eyes
    for sign in (-1, +1):
        ex = int(shw + sign * EYE_X_FRAC * shw)
        ey = int(shh + EYE_Y_FRAC * shh)
        er = max(2, int(EYE_R_FRAC * shw))
        pr = max(1, int(PUP_R_FRAC * shw))
        hr = max(1, int(HI_R_FRAC  * shw))

        pygame.draw.circle(surf, (245, 245, 250), (ex, ey), er)   # white
        pygame.draw.circle(surf, (15, 10, 20),    (ex, ey), pr)   # pupil
        # Highlight — offset up-right
        pygame.draw.circle(surf, (255, 255, 255),
                           (int(ex + er*0.35), int(ey - er*0.35)), hr)

    # Smile  (lower arc of a circle)
    sm_cy = int(shh + SMILE_CY_FRAC * shh)
    sm_r  = int(SMILE_R_FRAC * shw)
    sm_t  = max(1, int(SMILE_T_FRAC * shw))
    if sm_r > sm_t:
        rect = pygame.Rect(int(shw) - sm_r, sm_cy - sm_r, 2*sm_r, 2*sm_r)
        # In pygame (y-down), arc goes CCW in math coords.
        # Angles 0→π sweep the bottom of the circle visually.
        pygame.draw.arc(surf, (18, 12, 25), rect,
                        math.pi * 0.12, math.pi * 0.88, sm_t)


def draw_glow(screen, wx, wy, ang, color):
    """Bioluminescent glow: a soft, rotated ellipse around the segment."""
    ge = _s(GLOW_EXPAND)
    gw = _s(HW * 2) + 2 * ge
    gh = _s(HH * 2) + 2 * ge
    gsurf = pygame.Surface((gw, gh), pygame.SRCALPHA)
    gc = (*color, GLOW_ALPHA)
    # Inner bright core
    pygame.draw.ellipse(gsurf, gc, gsurf.get_rect())
    # Fade: draw a slightly smaller darker ellipse on top (makes edges softer)
    inner = pygame.Rect(ge//2, ge//2, gw - ge, gh - ge)
    pygame.draw.ellipse(gsurf, (*color, min(255, GLOW_ALPHA + 40)), inner)

    rotated = pygame.transform.rotate(gsurf, -math.degrees(ang))
    rw, rh  = rotated.get_size()
    sx, sy  = _sp(wx, wy)
    screen.blit(rotated, (sx - rw//2, sy - rh//2))


def draw_segment(screen, wx, wy, ang, color, is_head=False):
    seg = _seg_surface(color, with_face=is_head)
    rotated = pygame.transform.rotate(seg, -math.degrees(ang))
    rw, rh  = rotated.get_size()
    sx, sy  = _sp(wx, wy)
    screen.blit(rotated, (sx - rw//2, sy - rh//2))


def draw_joint(screen, seg_a, seg_b):
    """Thin cyan connector between two adjacent segments."""
    ax, ay, a_ang = seg_a
    bx, by, b_ang = seg_b

    # Attachment points: rear of A and front of B
    jax = ax + math.cos(a_ang) * JOINT_AX
    jay = ay + math.sin(a_ang) * JOINT_AX
    jbx = bx - math.cos(b_ang) * JOINT_AX
    jby = by - math.sin(b_ang) * JOINT_AX

    p0 = _sp(jax, jay)
    p1 = _sp(jbx, jby)
    jw = max(1, _s(JOINT_WIDTH))

    jsurf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
    pygame.draw.line(jsurf, (*JOINT_COLOR, int(0.55 * 255)), p0, p1, jw)
    screen.blit(jsurf, (0, 0))


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("worm_demo")
    clock  = pygame.time.Clock()

    print(f"Building background ({WIN_W}×{WIN_H})…", end=" ", flush=True)
    bg = make_background()
    print("done.")

    t = 0.0
    dt = 1.0 / FPS

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN and ev.key in (pygame.K_q, pygame.K_ESCAPE):
                pygame.quit(); sys.exit()

        screen.blit(bg, (0, 0))

        segs = worm_positions(t)

        # Glow pass (back-to-front)
        for i in range(NUM_SEGMENTS - 1, -1, -1):
            draw_glow(screen, segs[i][0], segs[i][1], segs[i][2], SEG_COLORS[i])

        # Joint connectors
        for i in range(NUM_SEGMENTS - 1):
            draw_joint(screen, segs[i], segs[i+1])

        # Segments (back-to-front)
        for i in range(NUM_SEGMENTS - 1, -1, -1):
            draw_segment(screen, segs[i][0], segs[i][1], segs[i][2],
                         SEG_COLORS[i], is_head=(i == 0))

        pygame.display.flip()
        clock.tick(FPS)
        t += dt


if __name__ == "__main__":
    main()
