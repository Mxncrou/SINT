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
    """
    Environment wrapper displaying the real SC2 rendering in a window.

    Every game loop is drawn (not only one frame per agent step), so ``fps`` is the game speed:
    22.4 is real time ("faster" speed), and agents act every ``step_mul`` loops (8 by default).

    Controls (click the window first):
        arrows : move the camera        F : follow the battle / fixed camera
        + / -  : zoom in / out          C : reset camera and zoom
    """

    def __init__(self, env, resolution=(1600, 1200), fps=22.4, camera_offset=0.0, pan_speed=0.5):
        self.env = env
        self.sc2_env = env.env  # unwrap StarCraftCapabilityEnvWrapper
        if self.sc2_env._sc2_proc is not None:
            raise RuntimeError("HumanRender must wrap the env before its first reset().")

        enable_rgb(resolution)
        self.fps = fps
        self.size = (resolution[0], int(resolution[1] * VIEW_HEIGHT_RATIO))
        self.screen = None
        self.clock = None
        self.open = True

        # Camera state
        self.camera_offset = camera_offset  # shifts the target up, above the hidden HUD band
        self.pan_speed = pan_speed          # world units per drawn frame while an arrow is held
        self.pan = [0.0, 0.0]               # manual offset from the anchor, in world units
        self.anchor = (0.0, 0.0)            # followed point: center of the living units
        self.follow = True
        self.zoom = 1.0                     # digital zoom: crop the frame center, then upscale

    def __getattr__(self, name):
        return getattr(self.env, name)

    # ------------------------------------------------------------------
    # Environment interface
    # ------------------------------------------------------------------

    def reset(self):
        result = self.env.reset()
        self._draw_every_loop()  # SMACv2 may relaunch SC2: patch the current controller
        self._reveal_map()
        self._center_camera()
        self._draw()
        return result

    def step(self, actions):
        return self.env.step(actions)  # frames are drawn inside, one per game loop

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
        return self.env.close()

    # ------------------------------------------------------------------
    # SC2 control
    # ------------------------------------------------------------------

    def _draw_every_loop(self):
        """
        SMACv2 advances SC2 by ``step_mul`` loops in one call, so only one frame per agent step
        would be rendered. Replace the controller's step so that it advances one loop at a time,
        moving the camera and drawing each loop.
        """
        controller = self.sc2_env._controller
        if hasattr(controller, "step_one_loop"):  # already patched
            return
        controller.step_one_loop = controller.step

        def step(count=1):
            if not self.open:  # window closed: run at full speed, without rendering
                return controller.step_one_loop(count)
            for _ in range(count):
                self._center_camera()
                controller.step_one_loop(1)
                self.sc2_env._obs = controller.observe()
                self._draw()

        controller.step = step

    def _reveal_map(self):
        """Remove the fog of war (unexplored areas are rendered black)."""
        self.sc2_env._controller.debug(d_pb.DebugCommand(game_state=d_pb.show_map))

    def _center_camera(self):
        """
        Move the camera to the anchor plus the manual pan.
        In follow mode, the anchor is the center of all living units.
        """
        if self.follow:
            units = list(self.sc2_env.agents.values()) + list(self.sc2_env.enemies.values())
            alive = [u for u in units if u.health > 0]
            if alive:
                self.anchor = (float(np.mean([u.pos.x for u in alive])),
                               float(np.mean([u.pos.y for u in alive])) - self.camera_offset)
        x, y = self.anchor[0] + self.pan[0], self.anchor[1] + self.pan[1]
        move = r_pb.ActionRawCameraMove(center_world_space=sc_common.Point(x=x, y=y))
        self.sc2_env._controller.act(sc_pb.Action(action_raw=r_pb.ActionRaw(camera_move=move)))

    # ------------------------------------------------------------------
    # Window
    # ------------------------------------------------------------------

    def _handle_input(self):
        """Read the keyboard. Return False if the window was closed."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # closing the window stops rendering only
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    self.follow = not self.follow
                elif event.key == pygame.K_c:
                    self.pan, self.zoom = [0.0, 0.0], 1.0
                elif event.key in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS):
                    self.zoom = min(self.zoom * 1.25, 4.0)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.zoom = max(self.zoom / 1.25, 1.0)

        keys = pygame.key.get_pressed()  # held keys: continuous movement
        self.pan[0] += self.pan_speed * (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT])
        self.pan[1] += self.pan_speed * (keys[pygame.K_UP] - keys[pygame.K_DOWN])
        return True

    def _frame(self):
        """Return the last rendered frame as a (H, W, 3) uint8 array, or None."""
        obs = self.sc2_env._obs
        if obs is None or not obs.observation.HasField("render_data"):
            return None
        frame = np.asarray(Feature.unpack_rgb_image(obs.observation.render_data.map))
        frame = frame[: int(frame.shape[0] * VIEW_HEIGHT_RATIO)]
        if self.zoom > 1.0:  # crop the center; it is upscaled to the window size when drawn
            h, w = frame.shape[:2]
            ch, cw = int(h / self.zoom), int(w / self.zoom)
            frame = frame[(h - ch) // 2:(h + ch) // 2, (w - cw) // 2:(w + cw) // 2]
        return frame

    def _draw(self):
        if not self.open:
            return
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(self.size)
            pygame.display.set_caption("SINT - SMACv2   [arrows] move  [F] follow  [+/-] zoom  [C] reset")
            self.clock = pygame.time.Clock()

        if not self._handle_input():
            self.open = False
            pygame.quit()
            self.screen = None
            return

        frame = self._frame()
        if frame is not None:
            surface = pygame.surfarray.make_surface(np.ascontiguousarray(frame.swapaxes(0, 1)))
            if surface.get_size() != self.size:
                surface = pygame.transform.smoothscale(surface, self.size)
            self.screen.blit(surface, (0, 0))
            pygame.display.flip()
        self.clock.tick(self.fps)
# ------------------------------Z-------------------------