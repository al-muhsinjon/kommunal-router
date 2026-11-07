"""Small PNG loss plot writer: NumPy pixels + standard-library PNG encoding."""
import struct
import zlib
from pathlib import Path
import numpy as np

# 3x5 bitmap glyphs keep plotting dependencies inside the assignment's allowlist.
FONT = dict(zip("0123456789.CEPOCHLS+TRAIN G",
    ("111101101101111", "010110010010111", "111001111100111", "111001111001111",
     "101101111001001", "111100111001111", "111100111101111", "111001010010010",
     "111101111101111", "111101111001111", "000000000000010", "111100100100111",
     "111100110100111", "111101111100100", "111101101101111", "111100100100111",
     "101101111101101", "100100100100111", "111100111001111", "000010111010000",
     "111010010010010", "110101110101101", "010101111101101", "111010010010111",
     "101111111111101", "000000000000000", "111100101101111")))


def save_loss(losses, path):
    width, height = 960, 540
    pixels = np.full((height, width, 3), 255, dtype=np.uint8)
    def line(x0, y0, x1, y1, color):
        n = max(abs(x1 - x0), abs(y1 - y0)) + 1
        x = np.rint(np.linspace(x0, x1, n)).astype(int)
        y = np.rint(np.linspace(y0, y1, n)).astype(int)
        pixels[y, x] = color
    def text(x, y, value, scale=3):
        for char in value:
            glyph = FONT[char]
            for index, bit in enumerate(glyph):
                if bit == "1":
                    row, col = divmod(index, 3)
                    pixels[y + row * scale:y + (row + 1) * scale,
                           x + col * scale:x + (col + 1) * scale] = (40, 50, 65)
            x += 4 * scale
    top = max(losses) * 1.05
    text(300, 25, "TRAINING LOSS")
    text(18, 65, "CE+L2", 2)
    text(445, 500, "EPOCH")
    for tick in range(5):
        y = 460 - tick * 90
        line(100, y, 920, y, (225, 230, 237))
        text(15, y - 7, f"{top * tick / 4:.2f}")
        x = 100 + tick * 205
        text(x - 12, 475, str(round((len(losses) - 1) * tick / 4)), 2)
    line(100, 100, 100, 460, (40, 50, 65))
    line(100, 460, 920, 460, (40, 50, 65))
    points = [(100 + round(i * 820 / (len(losses) - 1)), 460 - round(value * 360 / top))
              for i, value in enumerate(losses)]
    for start, end in zip(points, points[1:]):
        line(*start, *end, (25, 100, 210))
    def chunk(kind, data):
        return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data))
    raw = b"".join(b"\x00" + row.tobytes() for row in pixels)
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack("!2I5B", width, height, 8, 2, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
