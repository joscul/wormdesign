#!/usr/bin/env python3
"""realtime_brain.py — load a saved brain genome and run the directional
swimming simulation live on screen (pure Python / numpy / pygame).

This is a faithful but self-contained implementation of a worm brain simulation. 
The worm is a 9-segment chain that swims by undulation (anisotropic drag,
no gravity, no contacts). Synaptic weights start random and are shaped
online by reward-modulated STDP: the worm is rewarded for moving in the
chosen direction (default: up), so it should gradually learn to swim
that way over the first ~30-60 s.

Usage:
    python3 realtime_brain.py [worm_brain.model] [--dir up]
    python3 realtime_brain.py --dir up --seed 42 --scale 0.5

    --dir   up | down | left | right   (reward direction, default up)
    --seed  RNG seed for weight init + input spikes (default 42)
    --scale window scale vs the 1080x1920 world (default 0.5)
    SPACE   pause/resume,  R restart sim,  ESC/Q quit
"""

import sys
import math
import argparse
import numpy as np

# ── Constants
WORLD_W, WORLD_H = 1080, 1920
FPS = 60
DT = 1.0 / FPS

NUM_WORM = 9
NUM_JOINTS = NUM_WORM - 1
WORM_HW, WORM_HH = 32.0, 30.0
WORM_JOINT_AX = WORM_HW + 9.0 # 41
JOINT_BIAS = 0.25
MAX_JOINT_ANG = 1.5708 # pi/2
LIMIT_BIAS = 0.08
ITERATIONS = 4
DAMP_AXIAL = 0.997
DAMP_LATERAL = 0.970
ANG_DAMP = 0.990

N_NEURONS = 64
N_IN_JOINT = NUM_JOINTS # neurons 0..7
N_IN_DIR = 4 # neurons 8..11: right,left,down,up vel cells
N_IN_RWD = 1 # neuron 12: reward-speed cell
N_IN = N_IN_JOINT + N_IN_DIR + N_IN_RWD # 13
N_OUT = NUM_JOINTS * 2 # neurons 48..63, antagonist pairs
OUT_BASE = N_NEURONS - N_OUT # 48
LIF_REFRAC = 3
SYN_SCALE = 5.0
LP_BETA = 0.90
MOTOR_SCALE = 500.0
VX_REF = 50.0
REWARD_BASELINE = 0.1

# Genome layout: 7 per-neuron blocks + 2 scalars + 64x64 topology = 4546 floats.
OFF_TAU_M = 0
OFF_V_REST = 1 * N_NEURONS
OFF_V_THRESH = 2 * N_NEURONS
OFF_V_RESET = 3 * N_NEURONS
OFF_R_M = 4 * N_NEURONS
OFF_PT_DECAY = 5 * N_NEURONS
OFF_EXC_GENE = 6 * N_NEURONS
OFF_LOG_ETA = 7 * N_NEURONS
OFF_W_CLIP = 7 * N_NEURONS + 1
OFF_TOPO = 7 * N_NEURONS + 2
N_PARAMS = OFF_TOPO + N_NEURONS * N_NEURONS   # 4546

# Per-neuron parameter clamp ranges.
TAU_M_MIN, TAU_M_MAX = 5.0, 50.0
V_REST_MIN, V_REST_MAX = -80.0, -50.0
V_THRESH_MIN, V_THRESH_MAX = -60.0, -40.0
V_RESET_MIN, V_RESET_MAX = -90.0, -65.0
R_M_MIN, R_M_MAX = 1.0, 100.0
PT_DECAY_MIN, PT_DECAY_MAX = 0.50, 0.99
LR_LOG_MIN, LR_LOG_MAX = -4.0, -1.0
W_CLIP_MIN, W_CLIP_MAX = 0.01, 10.0

WORM_COLORS = [
    (0x22, 0xBB, 0x44), (0xEE, 0xCC, 0x00), (0xFF, 0x88, 0x00),
    (0xEE, 0x44, 0x00), (0xCC, 0x22, 0x44), (0xAA, 0x22, 0xAA),
    (0x77, 0x44, 0xCC), (0x44, 0x88, 0xEE), (0x22, 0x55, 0xDD),
]

DIRS = {
    "right": (1.0, 0.0),
    "left": (-1.0, 0.0),
    "down": (0.0, 1.0),
    "up": (0.0, -1.0),
}


def _clamp(v, lo, hi):
    return np.minimum(np.maximum(v, lo), hi)


class Brain:
    """LIF network + reward-modulated STDP, decoded from a genome file."""

    def __init__(self, genome, rng):
        g = genome
        self.inv_tau_m = 1.0 / _clamp(g[OFF_TAU_M:OFF_TAU_M + N_NEURONS], TAU_M_MIN, TAU_M_MAX)
        self.v_rest = _clamp(g[OFF_V_REST:OFF_V_REST + N_NEURONS], V_REST_MIN, V_REST_MAX)
        self.v_thresh = _clamp(g[OFF_V_THRESH:OFF_V_THRESH + N_NEURONS], V_THRESH_MIN, V_THRESH_MAX)
        self.v_reset = _clamp(g[OFF_V_RESET:OFF_V_RESET + N_NEURONS], V_RESET_MIN, V_RESET_MAX)
        self.r_m = _clamp(g[OFF_R_M:OFF_R_M + N_NEURONS], R_M_MIN, R_M_MAX)
        self.pt_decay = _clamp(g[OFF_PT_DECAY:OFF_PT_DECAY + N_NEURONS], PT_DECAY_MIN, PT_DECAY_MAX)
        self.exc = (g[OFF_EXC_GENE:OFF_EXC_GENE + N_NEURONS] > 0.0)
        self.lr = 10.0 ** float(np.clip(g[OFF_LOG_ETA], LR_LOG_MIN, LR_LOG_MAX))
        self.w_clip = float(np.clip(g[OFF_W_CLIP], W_CLIP_MIN, W_CLIP_MAX))

        # topo[pre, post]: synapse exists if gene > 0 (no autapses).
        topo = (g[OFF_TOPO:].reshape(N_NEURONS, N_NEURONS) > 0.0)
        np.fill_diagonal(topo, False)
        self.topo = topo

        # sign per presynaptic neuron (Dale's law): +1 excitatory, -1 inhibitory.
        self.sgn = np.where(self.exc, 1.0, -1.0)

        self.rng = rng
        self.reset()

    def reset(self):
        # Random initial weights in [0, w_clip], signed by exc, masked by topo.
        u = self.rng.random((N_NEURONS, N_NEURONS)).astype(np.float32) * self.w_clip
        w = np.where(self.exc[:, None], u, -u)
        self.weight = np.where(self.topo, w, 0.0).astype(np.float32)

        self.v = self.v_rest.copy()
        self.refrac = np.zeros(N_NEURONS, dtype=np.int32)
        self.pt = np.zeros(N_NEURONS, dtype=np.float32)
        self.prev_spikes = np.zeros(N_NEURONS, dtype=bool)
        self.lp = np.full(N_OUT, 0.5, dtype=np.float32)

    def step(self, jang, dir_prob, rew_prob, reward):
        """Advance one timestep. Returns the spike vector (bool[64])."""
        # Synaptic input from last frame's spikes.
        i_syn = SYN_SCALE * (self.prev_spikes @ self.weight)

        spiked = np.zeros(N_NEURONS, dtype=bool)

        # ── Input neurons 0..7: probabilistic rate coding based on joint angles ──
        prob = np.empty(N_IN, dtype=np.float32)
        prob[0:N_IN_JOINT] = (jang / MAX_JOINT_ANG + 1.0) * 0.5

        # Input neurons 8-11 encodes direction
        prob[N_IN_JOINT:N_IN_JOINT + N_IN_DIR] = dir_prob

        # Input neuron 12 encodes reward
        prob[N_IN_JOINT + N_IN_DIR] = rew_prob

        in_spk = self.rng.random(N_IN).astype(np.float32) < prob
        spiked[:N_IN] = in_spk
        self.v[:N_IN] = np.where(in_spk, self.v_reset[:N_IN], self.v[:N_IN])
        self.refrac[:N_IN] = np.where(
            in_spk, LIF_REFRAC, np.maximum(self.refrac[:N_IN] - 1, 0))

        # ── Hidden + output neurons 13..63: leaky integrate-and-fire ──
        idx = slice(N_IN, N_NEURONS)
        ref = self.refrac[idx]
        in_refrac = ref > 0
        v = self.v[idx].copy()
        # refractory neurons: held at reset, count down
        v[in_refrac] = self.v_reset[idx][in_refrac]
        # active neurons: integrate dv = (1/tau)*(-(v-v_rest) + r_m*i_syn)
        act = ~in_refrac
        v[act] += (self.inv_tau_m[idx][act]
                   * (-(v[act] - self.v_rest[idx][act]) + self.r_m[idx][act] * i_syn[idx][act]))
        fired = act & (v >= self.v_thresh[idx])
        v[fired] = self.v_reset[idx][fired]
        new_ref = np.where(fired, LIF_REFRAC, np.where(in_refrac, ref - 1, ref))
        self.v[idx] = v
        self.refrac[idx] = new_ref
        spiked[idx] = fired

        # ── Motor LP filter on output neurons (48..63) ──
        out_spk = spiked[OUT_BASE:OUT_BASE + N_OUT].astype(np.float32)
        self.lp = LP_BETA * self.lp + (1.0 - LP_BETA) * out_spk

        # ── Pre-synaptic trace ──
        self.pt = self.pt_decay * self.pt + spiked.astype(np.float32)

        # ── Reward-modulated STDP: update columns of post-neurons that spiked ──
        eta = self.lr * reward
        if eta != 0.0:
            col = spiked  # post neurons that fired this frame
            if col.any():
                dpre = (self.sgn * self.pt * eta).astype(np.float32)  # per-pre delta
                upd = self.topo & col[None, :]                        # (pre, post) mask
                self.weight = np.where(upd, self.weight + dpre[:, None], self.weight)
                np.clip(self.weight, -self.w_clip, self.w_clip, out=self.weight)

        self.prev_spikes = spiked
        return spiked

    def motor(self, joint):
        """Net antagonist drive for a joint, in [-1, 1]-ish."""
        return self.lp[2 * joint] - self.lp[2 * joint + 1]


class Worm:
    """9-segment swimmer: free-floating chain, anisotropic drag, no contacts."""

    def __init__(self):
        self.inv_m = 1.0 / (4.0 * WORM_HW * WORM_HH / 1000.0)
        inertia = (1.0 / self.inv_m) * (WORM_HW * WORM_HW + WORM_HH * WORM_HH) / 3.0
        self.inv_i = 1.0 / inertia
        self.reset()

    def reset(self):
        span = (NUM_WORM - 1) * 2.0 * WORM_JOINT_AX
        x0 = (WORLD_W - span) * 0.5
        self.px = np.array([x0 + b * 2.0 * WORM_JOINT_AX for b in range(NUM_WORM)], dtype=np.float64)
        self.py = np.full(NUM_WORM, WORLD_H * 0.5, dtype=np.float64)
        self.rot = np.zeros(NUM_WORM, dtype=np.float64)
        self.vx = np.zeros(NUM_WORM, dtype=np.float64)
        self.vy = np.zeros(NUM_WORM, dtype=np.float64)
        self.w = np.zeros(NUM_WORM, dtype=np.float64)
        self.jang = np.zeros(NUM_JOINTS, dtype=np.float64)

    def cog(self):
        return self.px.mean(), self.py.mean()

    def step(self, motor_cmd):
        """One physics frame. motor_cmd: callable joint->net drive."""
        px, py, rot = self.px, self.py, self.rot
        vx, vy, w = self.vx, self.vy, self.w
        inv_m, inv_i = self.inv_m, self.inv_i
        inv_dt = float(FPS)

        # ── Per-joint Jacobians (computed once per frame) ──
        a = np.arange(NUM_JOINTS)
        b = a + 1
        cA, sA = np.cos(rot[a]), np.sin(rot[a])
        cB, sB = np.cos(rot[b]), np.sin(rot[b])
        # anchors la1=(+AX,0) on A, la2=(-AX,0) on B, rotated to world
        r1x = cA * WORM_JOINT_AX
        r1y = sA * WORM_JOINT_AX
        r2x = -cB * WORM_JOINT_AX
        r2y = -sB * WORM_JOINT_AX
        invMs = inv_m + inv_m
        K00 = invMs + inv_i * r1y * r1y + inv_i * r2y * r2y
        K01 = -inv_i * r1x * r1y - inv_i * r2x * r2y
        K11 = invMs + inv_i * r1x * r1x + inv_i * r2x * r2x
        det = K00 * K11 - K01 * K01
        invDet = 1.0 / det
        M00 = K11 * invDet
        M01 = -K01 * invDet
        M11 = K00 * invDet
        biasx = -JOINT_BIAS * inv_dt * (px[b] + r2x - px[a] - r1x)
        biasy = -JOINT_BIAS * inv_dt * (py[b] + r2y - py[a] - r1y)
        ang = rot[b] - rot[a]
        ang -= 2.0 * math.pi * np.rint(ang / (2.0 * math.pi))
        self.jang = ang
        jlim_invm = 1.0 / (inv_i + inv_i)

        # ── Constraint solver (sequential Gauss-Seidel sweeps) ──
        for _ in range(ITERATIONS):
            for j in range(NUM_JOINTS):
                ja, jb = j, j + 1
                # joint angle limit
                dw = w[jb] - w[ja]
                if ang[j] > MAX_JOINT_ANG:
                    lam = jlim_invm * (-LIMIT_BIAS * inv_dt * (ang[j] - MAX_JOINT_ANG) - dw)
                    lam = min(lam, 0.0)
                    w[ja] -= inv_i * lam
                    w[jb] += inv_i * lam
                elif ang[j] < -MAX_JOINT_ANG:
                    lam = jlim_invm * (-LIMIT_BIAS * inv_dt * (ang[j] + MAX_JOINT_ANG) - dw)
                    lam = max(lam, 0.0)
                    w[ja] -= inv_i * lam
                    w[jb] += inv_i * lam
            for j in range(NUM_JOINTS):
                ja, jb = j, j + 1
                # point-to-point velocity constraint (keep anchors together)
                dvx = (vx[jb] - w[jb] * r2y[j]) - (vx[ja] - w[ja] * r1y[j])
                dvy = (vy[jb] + w[jb] * r2x[j]) - (vy[ja] + w[ja] * r1x[j])
                diffx = biasx[j] - dvx
                diffy = biasy[j] - dvy
                impx = M00[j] * diffx + M01[j] * diffy
                impy = M01[j] * diffx + M11[j] * diffy
                vx[ja] -= inv_m * impx
                vy[ja] -= inv_m * impy
                w[ja] -= inv_i * (r1x[j] * impy - r1y[j] * impx)
                vx[jb] += inv_m * impx
                vy[jb] += inv_m * impy
                w[jb] += inv_i * (r2x[j] * impy - r2y[j] * impx)

        # ── Motor torques ──
        for j in range(NUM_JOINTS):
            m = motor_cmd(j)
            if ang[j] >= MAX_JOINT_ANG and m < 0.0:
                m = 0.0
            if ang[j] <= -MAX_JOINT_ANG and m > 0.0:
                m = 0.0
            L = m * MOTOR_SCALE * DT * jlim_invm
            w[j] += inv_i * L
            w[j + 1] -= inv_i * L

        # ── Integration with anisotropic drag (lateral >> axial → propulsion) ──
        c, s = np.cos(rot), np.sin(rot)
        va = vx * c + vy * s
        vl = -vx * s + vy * c
        va *= DAMP_AXIAL
        vl *= DAMP_LATERAL
        self.vx = va * c - vl * s
        self.vy = va * s + vl * c
        self.w = w * ANG_DAMP
        self.px = px + self.vx * DT
        self.py = py + self.vy * DT
        self.rot = rot + self.w * DT


class Sim:
    """Couples the worm + brain + reward signal"""

    def __init__(self, brain, worm, eval_dir):
        self.brain = brain
        self.worm = worm
        self.eval_nx, self.eval_ny = eval_dir
        self.reset()

    def reset(self):
        self.worm.reset()
        self.brain.reset()
        cx, cy = self.worm.cog()
        self.start_cx, self.start_cy = cx, cy
        self.prev_cx, self.prev_cy = cx, cy
        self.ema_vx = self.ema_vy = 0.0
        self.dir_prob = np.zeros(N_IN_DIR, dtype=np.float32)
        self.rew_prob = 0.0
        self.reward = 0.0
        self.frame = 0
        self.spike_count = 0

    def step(self):
        # Physics first (uses last frame's motor output).
        self.worm.step(self.brain.motor)

        # Center-of-gravity velocity (EMA) and direction/reward cells.
        cx, cy = self.worm.cog()
        vx = (cx - self.prev_cx) * FPS
        vy = (cy - self.prev_cy) * FPS
        self.ema_vx = 0.8 * self.ema_vx + 0.2 * vx
        self.ema_vy = 0.8 * self.ema_vy + 0.2 * vy
        self.dir_prob[0] = min(max(self.ema_vx / VX_REF, 0.0), 1.0)   # right
        self.dir_prob[1] = min(max(-self.ema_vx / VX_REF, 0.0), 1.0)  # left
        self.dir_prob[2] = min(max(self.ema_vy / VX_REF, 0.0), 1.0)   # down
        self.dir_prob[3] = min(max(-self.ema_vy / VX_REF, 0.0), 1.0)  # up
        proj = self.ema_vx * self.eval_nx + self.ema_vy * self.eval_ny
        vlen = math.hypot(self.ema_vx, self.ema_vy)
        rew = proj / VX_REF if proj > 0.70710678 * vlen else 0.0   # 45° cone
        self.rew_prob = min(max(rew, 0.0), 1.0)
        self.reward = REWARD_BASELINE + (1.0 - REWARD_BASELINE) * self.rew_prob
        self.prev_cx, self.prev_cy = cx, cy

        # Neural step.
        spikes = self.brain.step(self.worm.jang, self.dir_prob, self.rew_prob, self.reward)
        self.spike_count = int(spikes.sum())
        self.frame += 1

    def progress(self):
        """Net displacement along the reward direction, in pixels."""
        cx, cy = self.worm.cog()
        return (cx - self.start_cx) * self.eval_nx + (cy - self.start_cy) * self.eval_ny


# ── Rendering ─────────────────────────────────────────────────────────────────
# pygame.font is broken in some wheels on Python 3.14, so we ship a tiny 3x5
# bitmap font and draw text as scaled rects — zero dependency on SDL_ttf.

_GLYPHS = {
    "0": ("###", "# #", "# #", "# #", "###"), "1": (" # ", "## ", " # ", " # ", "###"),
    "2": ("###", "  #", "###", "#  ", "###"), "3": ("###", "  #", "###", "  #", "###"),
    "4": ("# #", "# #", "###", "  #", "  #"), "5": ("###", "#  ", "###", "  #", "###"),
    "6": ("###", "#  ", "###", "# #", "###"), "7": ("###", "  #", "  #", "  #", "  #"),
    "8": ("###", "# #", "###", "# #", "###"), "9": ("###", "# #", "###", "  #", "###"),
    "A": ("###", "# #", "###", "# #", "# #"), "B": ("## ", "# #", "## ", "# #", "## "),
    "C": ("###", "#  ", "#  ", "#  ", "###"), "D": ("## ", "# #", "# #", "# #", "## "),
    "E": ("###", "#  ", "###", "#  ", "###"), "F": ("###", "#  ", "###", "#  ", "#  "),
    "G": ("###", "#  ", "# #", "# #", "###"), "H": ("# #", "# #", "###", "# #", "# #"),
    "I": ("###", " # ", " # ", " # ", "###"), "J": ("  #", "  #", "  #", "# #", "###"),
    "K": ("# #", "# #", "## ", "# #", "# #"), "L": ("#  ", "#  ", "#  ", "#  ", "###"),
    "M": ("# #", "###", "###", "# #", "# #"), "N": ("# #", "###", "###", "###", "# #"),
    "O": ("###", "# #", "# #", "# #", "###"), "P": ("###", "# #", "###", "#  ", "#  "),
    "Q": ("###", "# #", "# #", "###", "  #"), "R": ("###", "# #", "###", "## ", "# #"),
    "S": ("###", "#  ", "###", "  #", "###"), "T": ("###", " # ", " # ", " # ", " # "),
    "U": ("# #", "# #", "# #", "# #", "###"), "V": ("# #", "# #", "# #", "# #", " # "),
    "W": ("# #", "# #", "###", "###", "# #"), "X": ("# #", "# #", " # ", "# #", "# #"),
    "Y": ("# #", "# #", "###", " # ", " # "), "Z": ("###", "  #", " # ", "#  ", "###"),
    " ": ("   ", "   ", "   ", "   ", "   "), ".": ("   ", "   ", "   ", "   ", " # "),
    ":": ("   ", " # ", "   ", " # ", "   "), "+": ("   ", " # ", "###", " # ", "   "),
    "-": ("   ", "   ", "###", "   ", "   "), "/": ("  #", "  #", " # ", "#  ", "#  "),
    "%": ("# #", "  #", " # ", "#  ", "# #"), "(": (" # ", "#  ", "#  ", "#  ", " # "),
    ")": (" # ", "  #", "  #", "  #", " # "),
}


def draw_text(surf, x, y, text, px, color):
    import pygame
    cx = x
    for ch in text.upper():
        g = _GLYPHS.get(ch, _GLYPHS[" "])
        for ry, row in enumerate(g):
            for rx, c in enumerate(row):
                if c == "#":
                    pygame.draw.rect(surf, color, (cx + rx * px, y + ry * px, px, px))
        cx += 4 * px  # 3 wide + 1 spacing
    return cx


def run(sim, scale, dir_name):
    import pygame
    pygame.init()
    RW, RH = int(WORLD_W * scale), int(WORLD_H * scale)
    screen = pygame.display.set_mode((RW, RH))
    pygame.display.set_caption(f"physevol - directional brain ({dir_name})")
    clock = pygame.time.Clock()

    cam_x, cam_y = sim.worm.cog()
    reward_ema = 0.0
    paused = False

    def w2s(wx, wy):
        sx = (wx - cam_x + WORLD_W * 0.5) * scale
        sy = (wy - cam_y + WORLD_H * 0.5) * scale
        return sx, sy

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key == pygame.K_r:
                    sim.reset()
                    cam_x, cam_y = sim.worm.cog()
                    reward_ema = 0.0

        if not paused:
            sim.step()

        # Camera follows the center of gravity.
        cx, cy = sim.worm.cog()
        cam_x += (cx - cam_x) * 0.05
        cam_y += (cy - cam_y) * 0.05

        screen.fill((24, 26, 32))

        # Reference grid.
        grid = 120
        gx0 = int(cam_x - WORLD_W * 0.5)
        for gx in range(gx0 - gx0 % grid, int(cam_x + WORLD_W * 0.5) + grid, grid):
            x, _ = w2s(gx, 0)
            pygame.draw.line(screen, (38, 41, 50), (x, 0), (x, RH))
        gy0 = int(cam_y - WORLD_H * 0.5)
        for gy in range(gy0 - gy0 % grid, int(cam_y + WORLD_H * 0.5) + grid, grid):
            _, y = w2s(0, gy)
            pygame.draw.line(screen, (38, 41, 50), (0, y), (RW, y))

        # Worm segments (rotated OBBs).
        corners = [(-WORM_HW, -WORM_HH), (WORM_HW, -WORM_HH),
                   (WORM_HW, WORM_HH), (-WORM_HW, WORM_HH)]
        for bdy in range(NUM_WORM):
            c, s = math.cos(sim.worm.rot[bdy]), math.sin(sim.worm.rot[bdy])
            pts = []
            for lx, ly in corners:
                wx = sim.worm.px[bdy] + c * lx - s * ly
                wy = sim.worm.py[bdy] + s * lx + c * ly
                pts.append(w2s(wx, wy))
            pygame.draw.polygon(screen, WORM_COLORS[bdy], pts)
            pygame.draw.polygon(screen, (15, 15, 20), pts, 2)

        # Eyes on the head segment (segment nr 8).
        c, s = math.cos(sim.worm.rot[8]), math.sin(sim.worm.rot[8])
        for ey in (-12, 12):
            wx = sim.worm.px[8] + c * 14 - s * ey
            wy = sim.worm.py[8] + s * 14 + c * ey
            ex, eyy = w2s(wx, wy)
            pygame.draw.circle(screen, (255, 255, 255), (int(ex), int(eyy)), max(3, int(6 * scale)))
            pygame.draw.circle(screen, (0, 0, 0), (int(ex), int(eyy)), max(1, int(3 * scale)))

        # Reward-direction arrow (top center).
        ax, ay = RW * 0.5, 70
        L = 46
        tipx, tipy = ax + sim.eval_nx * L, ay + sim.eval_ny * L
        pygame.draw.line(screen, (255, 210, 30), (ax - sim.eval_nx * L, ay - sim.eval_ny * L),
                         (tipx, tipy + 10), 6)
        perp = (-sim.eval_ny, sim.eval_nx)
        head = 16
        pygame.draw.polygon(screen, (255, 210, 30), [
            (tipx, tipy),
            (tipx - sim.eval_nx * head + perp[0] * 10, tipy - sim.eval_ny * head + perp[1] * 10),
            (tipx - sim.eval_nx * head - perp[0] * 10, tipy - sim.eval_ny * head - perp[1] * 10),
        ])

        # Reward flash border.
        reward_ema = 0.88 * reward_ema + 0.12 * (1.0 if sim.rew_prob > 0.05 else 0.0)
        if reward_ema > 0.02:
            a = int(min(reward_ema, 1.0) * 160)
            flash = pygame.Surface((RW, RH), pygame.SRCALPHA)
            pygame.draw.rect(flash, (255, 200, 0, a), flash.get_rect(), 12)
            screen.blit(flash, (0, 0))

        # HUD text.
        t = sim.frame / FPS
        speed = sim.eval_nx * sim.ema_vx + sim.eval_ny * sim.ema_vy
        info = [
            f"reward dir: {dir_name}",
            f"speed: {speed:+.1f} px/s",
            f"reward: {sim.reward:.2f}",
            f"spikes/step: {sim.spike_count}",
            f"t: {t:.1f}s  frame: {sim.frame}",
        ]
        for i, ln in enumerate(info):
            draw_text(screen, 12, 12 + i * 22, ln, 3, (200, 205, 215))
        # Headline: distance moved in the reward direction.
        draw_text(screen, 12, RH - 64, f"{dir_name} movement: {sim.progress():+.0f} px",
                  6, (255, 255, 255))
        if paused:
            draw_text(screen, 12, RH - 110, "PAUSED (space)", 6, (255, 120, 120))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


def main():
    ap = argparse.ArgumentParser(description="Run a saved brain's directional sim in real time.")
    ap.add_argument("genome", nargs="?", default="brains/moving_worm_brain.bin",
                    help="path to brain genome .bin (default: brains/moving_worm_brain.bin)")
    ap.add_argument("--dir", choices=list(DIRS), default="up", help="reward direction (default: up)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    ap.add_argument("--scale", type=float, default=0.5, help="window scale vs 1080x1920 world")
    args = ap.parse_args()

    data = np.fromfile(args.genome, dtype=np.float32)
    if data.size < N_PARAMS:
        sys.exit(f"genome {args.genome!r} has {data.size} floats, expected {N_PARAMS}")
    genome = data[:N_PARAMS]

    rng = np.random.default_rng(args.seed)
    brain = Brain(genome, rng)
    worm = Worm()
    sim = Sim(brain, worm, DIRS[args.dir])

    print(f"Loaded {args.genome}  lr={brain.lr:.2e}  w_clip={brain.w_clip:.3f}  "
          f"exc={int(brain.exc.sum())}/{N_NEURONS}  synapses={int(brain.topo.sum())}")
    print(f"Reward direction: {args.dir}.  SPACE=pause  R=restart  ESC/Q=quit")
    run(sim, args.scale, args.dir)


if __name__ == "__main__":
    main()
