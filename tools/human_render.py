import os

os.environ["SDL_VIDEODRIVER"] = "x11"  # pygame's Wayland backend crashes on Hyprland

import numpy as np
import pygame
from pysc2.lib import remote_controller
from pysc2.lib.features import Feature
from pysc2.run_configs import platforms
from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import debug_pb2 as d_pb
from s2clientprotocol import raw_pb2 as r_pb
from s2clientprotocol import sc2api_pb2 as sc_pb

# ------------------------------Z-------------------------
# SC2 keeps the bottom of the screen for its HUD (console). It is not drawn
# in the rendered interface, so it shows up as a black band: we crop it.
VIEW_HEIGHT_RATIO = 546 / 768

_rgb_enabled = False


def enable_rgb(resolution=(1024, 768), minimap=(128, 128)):
    """Make every SC2 process launched afterwards render RGB frames."""
    global _rgb_enabled
    if _rgb_enabled:
        return
    _rgb_enabled = True

    # 1. Load the GL library (EGL) when the game process starts.
    original_start = platforms.Linux.start

    def start_with_rgb(self, want_rgb=True, **kwargs):
        return original_start(self, want_rgb=True, **kwargs)

    platforms.Linux.start = start_with_rgb

    # 2. Request rendered images when joining the game.
    original_join = remote_controller.RemoteController.join_game

    def join_with_rgb(self, req_join_game):
        render = req_join_game.options.render
        render.resolution.x, render.resolution.y = resolution
        render.minimap_resolution.x, render.minimap_resolution.y = minimap
        return original_join(self, req_join_game)

    remote_controller.RemoteController.join_game = join_with_rgb


class HumanRender:
    """Environment wrapper displaying the real SC2 rendering in a window."""

    def __init__(self, env, resolution=(1600, 1200), fps=60, camera_offset=0.0):
        self.env = env
        self.sc2_env = env.env  # unwrap StarCraftCapabilityEnvWrapper
        if self.sc2_env._sc2_proc is not None:
            raise RuntimeError("HumanRender must wrap the env before its first reset().")

        enable_rgb(resolution)
        self.fps = fps
        self.camera_offset = camera_offset
        self.size = (resolution[0], int(resolution[1] * VIEW_HEIGHT_RATIO))
        self.screen = None
        self.clock = None
        self.open = True

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):

        result = self.env.reset()

        controller = self.sc2_env._controller
        if not hasattr(controller, "step_one_loop"):
            controller.step_one_loop = controller.step
            def step(count=1):
                for _ in range(count):
                    print("loop")
                    controller.step_one_loop(1)
                    self.sc2_env._obs = controller.observe()
                    self._draw()
            controller.step = step

        self._reveal_map()
        self._draw()
        return result

    def step(self, actions):
        self._center_camera()
        result = self.env.step(actions)
        self._draw()
        return result

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
        return self.env.close()

    def _reveal_map(self):
        """Remove the fog of war (unexplored areas are rendered black)."""
        self.sc2_env._controller.debug(d_pb.DebugCommand(game_state=d_pb.show_map))

    def _center_camera(self):
        """Move the camera to the center of all living units.

        The camera target is drawn at the center of the full screen, which is
        partly hidden by the HUD band: ``camera_offset`` (world units) shifts
        it so that the battle sits in the middle of the visible area.
        """
        units = list(self.sc2_env.agents.values()) + list(self.sc2_env.enemies.values())
        alive = [u for u in units if u.health > 0]
        if not alive:
            return
        x = float(np.mean([u.pos.x for u in alive]))
        y = float(np.mean([u.pos.y for u in alive])) - self.camera_offset
        move = r_pb.ActionRawCameraMove(center_world_space=sc_common.Point(x=x, y=y))
        self.sc2_env._controller.act(sc_pb.Action(action_raw=r_pb.ActionRaw(camera_move=move)))

    def _frame(self):
        """Return the last rendered frame as a (H, W, 3) uint8 array, or None."""
        obs = self.sc2_env._obs
        if obs is None or not obs.observation.HasField("render_data"):
            return None
        frame = np.asarray(Feature.unpack_rgb_image(obs.observation.render_data.map))
        return frame[: int(frame.shape[0] * VIEW_HEIGHT_RATIO)]

    def _draw(self):
        if not self.open:
            return
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(self.size)
            pygame.display.set_caption("SINT - SMACv2")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # closing the window stops rendering only
                self.open = False
                pygame.quit()
                self.screen = None
                return

        frame = self._frame()
        if frame is not None:
            surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            self.screen.blit(surface, (0, 0))
            pygame.display.flip()
        self.clock.tick(self.fps)
# ------------------------------Z-------------------------
