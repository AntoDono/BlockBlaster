"""Pygame app for collecting real, labelled piece crops from a live device.

Shows the phone screen on the left, the padded slot crops the CNN would see in
the middle, and a clickable palette of every piece on the right. The CNN's
guess pre-fills each slot's label; correct any wrong ones, then press ENTER to
save. Re-captures of the same tray are de-duplicated automatically, so you can
just keep playing and collect one fresh sample per new tray.
"""

from __future__ import annotations

from typing import Optional

import pygame

from blockblaster.assist.vision.detection import (
    detect_interactables,
    estimate_background_bgr,
    split_roles,
)
from blockblaster.assist.vision.piece_recognizer import PieceRecognizer, pad_to_slot
from blockblaster.assist.render.phone import bgr_to_surface, draw_phone_panel
from blockblaster.control.device import Device
from blockblaster.game.pieces import PIECES
from blockblaster.gui.render import PIECE_COLORS, draw_piece_preview
from blockblaster.piece_cnn.realdata import RealPieceStore, dhash, hamming

# Same tray re-detected if every slot stays within this Hamming distance; a
# larger jump means the queue changed and labels should be re-predicted.
_TRAY_CHANGE_DIST = 12
# A changed tray must persist this many consecutive frames before it replaces
# what's on screen — debounces detection jitter so the panel doesn't flicker.
_STABLE_FRAMES = 4
_MAX_SLOTS = 3
_TARGET_FPS = 60

_PAD        = 16
_PHONE_H    = 760
_PHONE_W    = round(_PHONE_H * 9 / 19.5)
_SLOT_W     = 210
_PALETTE_W  = 470
_STATUS_H   = 44

_WIN_W = _PAD + _PHONE_W + _PAD + _SLOT_W + _PAD + _PALETTE_W + _PAD
_WIN_H = _PAD + _PHONE_H + _PAD + _STATUS_H

_BG          = (14, 14, 20)
_PANEL_BG    = (20, 20, 30)
_PANEL_LINE  = (55, 55, 75)
_TEXT        = (220, 220, 235)
_DIM         = (120, 120, 140)
_LABEL       = (160, 160, 185)
_SELECT_COL  = (90, 200, 255)
_EDITED_COL  = (240, 200, 90)
_OK_COL      = (90, 220, 110)
_MID_COL     = (240, 210, 80)
_LOW_COL     = (235, 90, 90)


class CollectorApp:
    def __init__(self, device: Device, store: RealPieceStore) -> None:
        self._device = device
        self._store = store
        self._recognizer = PieceRecognizer()

        self._crops: list = []
        self._labels: list[Optional[str]] = []
        self._preds: list[tuple] = []
        self._committed_sigs: list[int] = []
        self._pending_sigs: list[int] = []
        self._pending_crops: list = []
        self._pending_count = 0
        self._selected = 0
        self._saved: set[int] = set()
        self._message = "Correct labels, then [ENTER] to collect."
        self._palette_rects: dict[str, pygame.Rect] = {}

    # ── Detection / labelling ─────────────────────────────────────────────────

    @staticmethod
    def _same_tray(a: list[int], b: list[int]) -> bool:
        return len(a) == len(b) and all(
            hamming(x, y) <= _TRAY_CHANGE_DIST for x, y in zip(a, b)
        )

    def _update(self, frame) -> None:
        """Detect the tray, but only commit a *stable* change to the display.

        Raw per-frame detection jitters (boxes wobble, a piece is missed for a
        frame), which is what made the panel flicker. We debounce: a candidate
        tray must persist for several frames and differ from what's shown
        before it replaces the crops/labels.
        """
        _, pieces = split_roles(detect_interactables(frame, detect_board=False))
        pieces = pieces[:_MAX_SLOTS]
        bg = estimate_background_bgr(frame)
        crops = [pad_to_slot(frame, p.bbox, bg) for p in pieces]
        sigs = [dhash(c) for c in crops]

        # Already showing this tray → nothing to do (keeps the GUI rock-steady
        # and preserves any label corrections the user has made).
        if self._same_tray(sigs, self._committed_sigs):
            self._pending_count = 0
            return

        if self._same_tray(sigs, self._pending_sigs):
            self._pending_count += 1
        else:
            self._pending_sigs = sigs
            self._pending_crops = crops
            self._pending_count = 1

        if self._pending_count >= _STABLE_FRAMES:
            self._commit(self._pending_crops, self._pending_sigs)

    def _commit(self, crops: list, sigs: list[int]) -> None:
        self._crops = crops
        self._committed_sigs = sigs
        self._pending_count = 0
        self._saved.clear()
        if crops:
            self._preds = self._recognizer.recognize_crops(crops)
            self._labels = [pc.name if pc is not None else None
                            for pc, _ in self._preds]
        else:
            self._preds = []
            self._labels = []
        self._selected = min(self._selected, max(len(crops) - 1, 0))

    def _collect(self) -> None:
        saved = skipped = 0
        for i, (crop, label) in enumerate(zip(self._crops, self._labels)):
            if label is None:
                skipped += 1
                continue
            self._store.save(label, crop)
            self._saved.add(i)
            saved += 1
        parts = [f"saved {saved}"]
        if skipped:
            parts.append(f"{skipped} unlabelled")
        self._message = ", ".join(parts) + f"  ·  total {self._store.total()}"

    # ── Events ─────────────────────────────────────────────────────────────────

    def _handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_q, pygame.K_ESCAPE):
                return False
            if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                self._collect()
            elif event.key == pygame.K_BACKSPACE and self._selected < len(self._labels):
                self._labels[self._selected] = None
                self._saved.discard(self._selected)
            elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                idx = event.key - pygame.K_1
                if idx < len(self._crops):
                    self._selected = idx
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._handle_click(event.pos)
        return True

    def _handle_click(self, pos: tuple[int, int]) -> None:
        for i in range(len(self._crops)):
            if self._slot_rect(i).collidepoint(pos):
                self._selected = i
                return
        for name, rect in self._palette_rects.items():
            if rect.collidepoint(pos):
                if self._selected < len(self._labels):
                    self._labels[self._selected] = name
                    self._saved.discard(self._selected)
                return

    # ── Geometry ───────────────────────────────────────────────────────────────

    @staticmethod
    def _phone_rect() -> pygame.Rect:
        return pygame.Rect(_PAD, _PAD, _PHONE_W, _PHONE_H)

    @staticmethod
    def _slots_rect() -> pygame.Rect:
        return pygame.Rect(_PAD + _PHONE_W + _PAD, _PAD, _SLOT_W, _PHONE_H)

    @staticmethod
    def _palette_rect() -> pygame.Rect:
        x = _PAD + _PHONE_W + _PAD + _SLOT_W + _PAD
        return pygame.Rect(x, _PAD, _PALETTE_W, _PHONE_H)

    def _slot_rect(self, i: int) -> pygame.Rect:
        outer = self._slots_rect()
        gap = 10
        h = (outer.height - gap * (_MAX_SLOTS - 1)) // _MAX_SLOTS
        return pygame.Rect(outer.x, outer.y + i * (h + gap), outer.width, h)

    # ── Rendering ────────────────────────────────────────────────────────────────

    def _draw(self, screen, frame, fonts) -> None:
        title, normal, small, tiny = fonts
        screen.fill(_BG)

        draw_phone_panel(
            screen, frame=frame, elements=[], rect=self._phone_rect(),
            error_msg=self._device.last_error, small_font=small,
        )
        self._draw_slots(screen, (title, normal, small, tiny))
        self._draw_palette(screen, (title, normal, small, tiny))
        self._draw_status(screen, small)

    def _draw_slots(self, screen, fonts) -> None:
        _title, normal, small, tiny = fonts
        for i in range(_MAX_SLOTS):
            rect = self._slot_rect(i)
            selected = i == self._selected and i < len(self._crops)
            is_saved = i in self._saved
            pygame.draw.rect(screen, _PANEL_BG, rect, border_radius=8)
            if is_saved:
                border, bw = _OK_COL, 3
            elif selected:
                border, bw = _SELECT_COL, 2
            else:
                border, bw = _PANEL_LINE, 1
            pygame.draw.rect(screen, border, rect, width=bw, border_radius=8)

            color = PIECE_COLORS[i % len(PIECE_COLORS)]
            hdr = small.render(f"#{i + 1}", True, color)
            screen.blit(hdr, (rect.x + 8, rect.y + 6))
            if is_saved:
                badge = small.render("SAVED \u2713", True, _OK_COL)
                screen.blit(badge, (rect.right - badge.get_width() - 8, rect.y + 6))

            if i >= len(self._crops):
                msg = small.render("—", True, _DIM)
                screen.blit(msg, (rect.centerx - msg.get_width() // 2,
                                  rect.centery - msg.get_height() // 2))
                continue

            img_box = pygame.Rect(rect.x + 8, rect.y + 28, rect.width - 16,
                                  rect.height - 78)
            surf, _, bx, by = bgr_to_surface(self._crops[i], img_box)
            screen.blit(surf, (bx, by))

            pred_piece, conf = (self._preds[i] if i < len(self._preds)
                                else (None, 0.0))
            pred_name = pred_piece.name if pred_piece is not None else "?"
            pred_txt = tiny.render(f"cnn: {pred_name} p={conf:.2f}", True,
                                   _conf_color(conf))
            screen.blit(pred_txt, (rect.x + 8, rect.bottom - 44))

            label = self._labels[i] if i < len(self._labels) else None
            edited = label != pred_name
            lab_col = _EDITED_COL if edited else _OK_COL
            lab_str = label if label is not None else "(none)"
            count = self._store.count(label) if label else 0
            lab_txt = normal.render(f"{lab_str}", True,
                                    lab_col if label else _DIM)
            screen.blit(lab_txt, (rect.x + 8, rect.bottom - 28))
            cnt_txt = tiny.render(f"have {count}", True, _DIM)
            screen.blit(cnt_txt, (rect.right - cnt_txt.get_width() - 8,
                                  rect.bottom - 26))

    def _draw_palette(self, screen, fonts) -> None:
        _title, _normal, small, tiny = fonts
        rect = self._palette_rect()
        pygame.draw.rect(screen, _PANEL_BG, rect, border_radius=8)
        pygame.draw.rect(screen, _PANEL_LINE, rect, width=1, border_radius=8)
        hdr = small.render("PIECE PALETTE  (click to label slot)", True, _LABEL)
        screen.blit(hdr, (rect.x + 10, rect.y + 8))

        cols = 5
        rows = (len(PIECES) + cols - 1) // cols
        grid_y = rect.y + 32
        gap = 6
        bw = (rect.width - gap * (cols + 1)) // cols
        bh = (rect.height - (grid_y - rect.y) - gap * (rows + 1)) // rows

        current = (self._labels[self._selected]
                   if self._selected < len(self._labels) else None)

        self._palette_rects = {}
        for idx, piece in enumerate(PIECES):
            r, c = divmod(idx, cols)
            bx = rect.x + gap + c * (bw + gap)
            by = grid_y + gap + r * (bh + gap)
            btn = pygame.Rect(bx, by, bw, bh)
            self._palette_rects[piece.name] = btn

            chosen = piece.name == current
            pygame.draw.rect(screen, (32, 32, 46), btn, border_radius=6)
            pygame.draw.rect(screen, _SELECT_COL if chosen else _PANEL_LINE,
                             btn, width=2 if chosen else 1, border_radius=6)

            cell = max(4, min((bw - 12) // 5, (bh - 18) // 5, 10))
            pw, ph = piece.cols * cell, piece.rows * cell
            px = btn.x + (btn.width - pw) // 2
            py = btn.y + 4 + (btn.height - 16 - ph) // 2
            draw_piece_preview(screen, piece, px, py,
                               color=(150, 190, 240), cell_size=cell)
            name_txt = tiny.render(piece.name, True, _TEXT if chosen else _DIM)
            screen.blit(name_txt, (btn.centerx - name_txt.get_width() // 2,
                                   btn.bottom - 13))

    def _draw_status(self, screen, small) -> None:
        rect = pygame.Rect(0, _PAD + _PHONE_H + _PAD, _WIN_W, _STATUS_H)
        pygame.draw.rect(screen, (10, 10, 16), rect)

        connected = self._device.last_error is None
        dev_col = _OK_COL if connected else _LOW_COL
        dev_txt = small.render(
            f"total {self._store.total()}  ·  session {self._store.session_saved}",
            True, dev_col,
        )
        screen.blit(dev_txt, (rect.x + 12,
                              rect.y + (rect.height - dev_txt.get_height()) // 2))

        msg = small.render(self._message, True, _LABEL)
        screen.blit(msg, (rect.centerx - msg.get_width() // 2,
                          rect.y + (rect.height - msg.get_height()) // 2))

        hint = small.render("[1/2/3] slot  [ENTER] collect  [⌫] clear  [Q] quit",
                            True, _DIM)
        screen.blit(hint, (rect.right - hint.get_width() - 12,
                           rect.y + (rect.height - hint.get_height()) // 2))

    # ── Main loop ────────────────────────────────────────────────────────────────

    def run(self) -> None:
        pygame.init()
        pygame.font.init()
        title  = pygame.font.SysFont("monospace", 22, bold=True)
        normal = pygame.font.SysFont("monospace", 17, bold=True)
        small  = pygame.font.SysFont("monospace", 14)
        tiny   = pygame.font.SysFont("monospace", 11)
        fonts  = (title, normal, small, tiny)

        screen = pygame.display.set_mode((_WIN_W, _WIN_H))
        pygame.display.set_caption("Block Blast – Data Collector")
        clock = pygame.time.Clock()

        self._device.start()
        last_fid = -1
        running = True
        try:
            while running:
                frame, fid = self._device.get_latest_with_id()
                if frame is not None and fid != last_fid:
                    last_fid = fid
                    self._update(frame)

                for event in pygame.event.get():
                    if not self._handle_event(event):
                        running = False

                self._draw(screen, frame, fonts)
                pygame.display.flip()
                clock.tick(_TARGET_FPS)
        finally:
            self._device.stop()
            pygame.quit()


def _conf_color(conf: float) -> tuple[int, int, int]:
    if conf >= 0.90:
        return _OK_COL
    if conf >= 0.70:
        return _MID_COL
    return _LOW_COL


def run(
    device: Optional[Device] = None,
    out_dir: str = "data/pieces",
) -> None:
    """Launch the data-collection GUI."""
    if device is None:
        from blockblaster.control.ios_readonly import IosReadOnlyDevice
        device = IosReadOnlyDevice()
    store = RealPieceStore(out_dir)
    print(f"[collect] writing to {store.root.resolve()}  "
          f"(already have {store.total()} samples)")
    CollectorApp(device, store).run()
