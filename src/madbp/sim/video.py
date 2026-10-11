"""Top-down episode videos (matplotlib; GIF via Pillow, MP4 via ffmpeg).

Works with any arena (point-mass, diff-drive, MuJoCo) because it renders from
:class:`madbp.sim.record.Recorder` snapshots, so it needs no OpenGL. Robots are
drawn as discs with a heading tick; goals as stars in the robot's colour; the
current planned window as a dotted line; fading trails show executed paths.
"""

from __future__ import annotations

import numpy as np


def save_video(rec, scenario, path: str, fps: int = 20, trail: int = 40, show_plan: bool = True,
               dpi: int = 90, title: str = "") -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Circle

    frames = rec.frames
    if not frames:
        raise ValueError("recorder has no frames")
    B, r, ro = scenario.bound, scenario.agent_radius, scenario.obstacle_radius
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=dpi)
    if path.lower().endswith(".mp4"):
        writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
    else:
        writer = animation.PillowWriter(fps=fps)
    hist: dict[int, list] = {}
    with writer.saving(fig, path, dpi):
        for fr in frames:
            ax.clear()
            ax.set_xlim(-B, B)
            ax.set_ylim(-B, B)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.add_patch(plt.Rectangle((-B, -B), 2 * B, 2 * B, fill=False, lw=2, ec="0.2"))
            for c in scenario.obstacles:
                ax.add_patch(Circle(c, ro, fc="0.35", ec="0.15"))
            for i in np.flatnonzero(fr["active"]):
                col = cmap(i % 10)
                hist.setdefault(i, []).append(fr["pos"][i].copy())
                h = np.asarray(hist[i][-trail:])
                if len(h) > 1:
                    ax.plot(h[:, 0], h[:, 1], "-", color=col, lw=1.2, alpha=0.5)
                if show_plan and fr["window"] is not None:
                    w = fr["window"][i]
                    ax.plot(w[:, 0], w[:, 1], ":", color=col, lw=1.6)
                ax.plot(*fr["goals"][i], marker="*", ms=11, color=col, mec="k", mew=0.5, ls="")
                ax.add_patch(Circle(fr["pos"][i], r, fc=col, ec="k", lw=0.8, zorder=3))
                th = fr["theta"][i]
                p = fr["pos"][i]
                ax.plot([p[0], p[0] + 1.3 * r * np.cos(th)], [p[1], p[1] + 1.3 * r * np.sin(th)],
                        "-", color="k", lw=1.6, zorder=4)
            for i in hist:  # drop history of removed agents
                if not fr["active"][i]:
                    hist[i] = []
            ax.set_title(f"{title}  t={fr['t']}  coll(a-a/a-o)={fr['aa']}/{fr['ao']}", fontsize=8)
            writer.grab_frame()
    plt.close(fig)
    return path


def save_pixels_video(pixels, path: str, fps: int = 20) -> str:
    """Write native MuJoCo renderer frames (``[H,W,3]`` uint8) to GIF/MP4."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    h, w = pixels[0].shape[:2]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    im = ax.imshow(pixels[0])
    writer = animation.FFMpegWriter(fps=fps) if path.lower().endswith(".mp4") else animation.PillowWriter(fps=fps)
    with writer.saving(fig, path, 100):
        for p in pixels:
            im.set_data(p)
            writer.grab_frame()
    plt.close(fig)
    return path
