"""Pygame application loop: auto-plays Block Blast using the trained agent."""

from __future__ import annotations

from typing import Optional

import pygame

import param
from blockblaster.agent.policy import select_action
from blockblaster.game.env import BlockBlastEnv
from blockblaster.gui.render import (
    BG_COLOR,
    CELL_SIZE,
    GRID_PADDING,
    draw_board,
    draw_info,
    draw_queue,
    make_window,
)
from blockblaster.model.value_net import ValueNet

FPS = 30
STEPS_PER_SECOND = 4        # how many agent moves per second when running


def run(net: Optional[ValueNet] = None, seed: int = 0) -> None:
    """
    Launch the pygame window and play Block Blast using `net` (if provided).
    Controls:
        SPACE  – pause / resume
        R      – reset game
        Q / ESC – quit
    """
    pygame.init()
    pygame.font.init()
    font       = pygame.font.SysFont("monospace", 20, bold=True)
    small_font = pygame.font.SysFont("monospace", 16)

    screen, layout = make_window()
    clock = pygame.time.Clock()

    env = BlockBlastEnv(seed=seed)
    paused    = False
    game_over = False

    step_interval_ms = 1000 // STEPS_PER_SECOND
    last_step_time   = pygame.time.get_ticks()

    running = True
    while running:
        now = pygame.time.get_ticks()

        # ── Events ──────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    last_step_time = now
                elif event.key == pygame.K_r:
                    env.reset(seed=seed)
                    game_over = False
                    paused    = False
                    last_step_time = now

        # ── Agent step ───────────────────────────────────────────────────
        if not paused and not game_over:
            if now - last_step_time >= step_interval_ms:
                if env.is_over():
                    game_over = True
                else:
                    action = select_action(env, net, epsilon=0.0)
                    env.step(*action)
                    if env.is_over():
                        game_over = True
                last_step_time = now

        # ── Render ───────────────────────────────────────────────────────
        screen.fill(BG_COLOR)

        draw_board(
            screen, env.board,
            layout["board_x"], layout["board_y"],
        )
        draw_queue(
            screen, env.queue,
            layout["queue_x"], layout["queue_y"],
            layout["win_h"],
            font, small_font,
        )
        draw_info(
            screen,
            score=env.total_score,
            steps=env.steps,
            game_over=game_over,
            paused=paused,
            x0=layout["board_x"],
            y0=layout["info_y"],
            win_w=layout["win_w"],
            font=font,
            small_font=small_font,
        )

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
