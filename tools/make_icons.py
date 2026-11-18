#!/usr/bin/env python3
"""Draw the Boots dashboard icons (2026-11-19) as PNG/ICO with Pillow.

The design is bootsui/icons/icon.svg ("boot buddy"); this redraws the same
shapes so no SVG renderer is needed. Output goes to bootsui/icons/:

  favicon.ico            16/32/48, rounded dark tile
  icon-192.png, -512     rounded tile, transparent corners (manifest "any")
  icon-maskable-512.png  full-bleed square, boot in the safe zone (Android)
  apple-touch-icon.png   180, full-bleed (iOS rounds the corners itself)

    python tools/make_icons.py
"""
import os

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "bootsui", "icons")

BG, GREEN, PALE, INK, WHITE = "#111412", "#5CCB8A", "#A6E5BF", "#0B2416", "#FFFFFF"
SS = 8                                     # supersampling factor


def quad(p0, p1, p2, n=24):
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1])
            for t in (i / n for i in range(1, n + 1))]


# M19 9 H38 V32 Q38 36 42 36 H45 Q55 36 55 46 V50 H14 V44 Q19 40 19 34 Z
BOOT = ([(19, 9), (38, 9), (38, 32)] + quad((38, 32), (38, 36), (42, 36)) + [(45, 36)]
        + quad((45, 36), (55, 36), (55, 46)) + [(55, 50), (14, 50), (14, 44)]
        + quad((14, 44), (19, 40), (19, 34)))
SMILE = [(24, 31)] + quad((24, 31), (29.5, 36), (35, 31))
LACES = [[(40, 40), (44, 37)], [(45, 41), (49, 38)]]


def draw(size, full):
    W = size * SS
    img = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    u = W / 64                              # one SVG unit in pixels
    if full:
        d.rectangle([0, 0, W, W], fill=BG)
        k, ox, oy = 0.78, 34.5, 32          # matches icon-full.svg's transform
        T = lambda x, y: (((x - ox) * k + 32) * u, ((y - oy) * k + 32) * u)
        s = u * k
    else:
        d.rounded_rectangle([0, 0, W - 1, W - 1], radius=14 * u, fill=BG)
        T = lambda x, y: (x * u, y * u)
        s = u

    def stroke_line(pts, w, close=False):
        px = [T(*p) for p in pts] + ([T(*pts[0])] if close else [])
        d.line(px, fill=INK, width=max(1, round(w * s)), joint="curve")
        r = w * s / 2
        for x, y in (px[0], px[-1]):        # round caps
            d.ellipse([x - r, y - r, x + r, y + r], fill=INK)

    def circle(cx, cy, r, fill, w=0):
        x, y = T(cx, cy)
        R = (r + w / 2) * s
        d.ellipse([x - R, y - R, x + R, y + R], fill=fill,
                  outline=INK if w else None, width=round(w * s) if w else 0)

    d.polygon([T(*p) for p in BOOT], fill=GREEN)
    stroke_line(BOOT, 2.5, close=True)
    x0, y0 = T(12 - 1.25, 49 - 1.25)
    x1, y1 = T(57 + 1.25, 55 + 1.25)
    d.rounded_rectangle([x0, y0, x1, y1], radius=(3 + 1.25) * s, fill=PALE,
                        outline=INK, width=round(2.5 * s))
    circle(25, 22, 5, WHITE, 2)
    circle(34, 22, 5, WHITE, 2)
    circle(26, 23, 2.2, INK)
    circle(35, 23, 2.2, INK)
    stroke_line(SMILE, 2.2)
    for seg in LACES:
        stroke_line(seg, 2)
    return img.resize((size, size), Image.LANCZOS)


def main():
    os.makedirs(OUT, exist_ok=True)
    draw(192, False).save(os.path.join(OUT, "icon-192.png"), optimize=True)
    draw(512, False).save(os.path.join(OUT, "icon-512.png"), optimize=True)
    draw(512, True).save(os.path.join(OUT, "icon-maskable-512.png"), optimize=True)
    draw(180, True).convert("RGB").save(os.path.join(OUT, "apple-touch-icon.png"), optimize=True)
    big = draw(256, False)
    big.save(os.path.join(OUT, "favicon.ico"), sizes=[(16, 16), (32, 32), (48, 48)])
    print("icons written to", OUT)


if __name__ == "__main__":
    main()
