# wormdesign

A tiny spiking-neural-network worm that learns to swim, plus a pixel-art design
proposal for what the worm could look like.

## Files

| File | What it is |
| --- | --- |
| `realtime_brain.py` | The real-time simulation. Loads a brain genome and runs a learning worm on screen. |
| `worm_brain.model` | A saved brain genome (raw `float32` weights/parameters) loaded by the simulation. |
| `worm_demo.py` | A standalone visual *design proposal* for the worm — animation only, no brain. |

## Requirements

```sh
pip install numpy pygame
```

Python 3 is required.

## Running the simulation — `realtime_brain.py`

This is the main program. It loads a saved brain (`worm_brain.model` by default)
and runs the worm live in a pygame window.

```sh
python3 realtime_brain.py
```

The worm is a 9-segment chain that swims by undulation (anisotropic drag, no
gravity, no contacts). Its synaptic weights start random and are shaped *online*
by reward-modulated STDP: the worm is rewarded for moving in the chosen
direction (default: **up**), so over the first ~30–60 seconds it should
gradually learn to swim that way.

### Options

```sh
python3 realtime_brain.py [genome] [--dir up] [--seed 42] [--scale 0.5]
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `genome` | `worm_brain.model` | Path to the brain genome file to load. |
| `--dir` | `up` | Reward direction: `up`, `down`, `left`, or `right`. |
| `--seed` | `42` | RNG seed for weight init + input spikes. |
| `--scale` | `0.5` | Window scale relative to the 1080×1920 world. |

### Controls

| Key | Action |
| --- | --- |
| `SPACE` | Pause / resume |
| `R` | Restart the simulation (re-randomizes weights) |
| `ESC` / `Q` | Quit |

The HUD shows the reward direction, current speed along it, the instantaneous
reward, spikes per step, elapsed time, and total distance moved in the reward
direction. A yellow border flashes whenever the worm is being rewarded.

## The brain — `worm_brain.model`

`worm_brain.model` is a **genome**: a flat array of `float32` values (4546 of
them) that describes a 64-neuron spiking network. It is read with
`numpy.fromfile` and decoded by the `Brain` class in `realtime_brain.py`.

The genome encodes, per neuron, the parameters of a leaky integrate-and-fire
(LIF) model — membrane time constant, rest / threshold / reset voltages,
membrane resistance, presynaptic-trace decay, and an excitatory/inhibitory flag
(Dale's law) — plus two global scalars (learning rate and weight clip) and a
64×64 connectivity matrix (which neuron connects to which).

Of the 64 neurons:

- **0–7** — input neurons encoding the 8 joint angles.
- **8–11** — input neurons encoding movement direction (right/left/down/up).
- **12** — input neuron encoding the reward signal.
- **48–63** — output neurons, antagonist pairs driving the 8 joints.

Note that the genome only fixes the *structure* and per-neuron parameters. The
actual synaptic **weights start random** each run (seeded by `--seed`) and are
learned live via reward-modulated STDP — the model file is not a pre-trained
set of weights.

## The design proposal — `worm_demo.py`

`worm_demo.py` is **not** part of the simulation. It is a standalone,
purely cosmetic *design proposal* showing how the worm could look: a pixel-art
caterpillar of 9 round green segments that wriggles with a sine-wave gait, with
eyes, antennae, and a mouth on the head. There is no brain, physics, or
learning here — just the animation.

```sh
python3 worm_demo.py
```

Press `Q` or `ESC` to quit.
