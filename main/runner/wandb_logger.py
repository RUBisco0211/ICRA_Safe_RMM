import os
import queue
from pathlib import Path

import numpy as np

try:
    import wandb
except ImportError:
    wandb = None


def get_next_run_dir(run_root):
    run_root = Path(run_root)
    if not run_root.exists():
        os.makedirs(str(run_root))

    run_nums = [
        int(str(folder.name).split("run")[1])
        for folder in run_root.iterdir()
        if str(folder.name).startswith("run")
    ]
    curr_run = "run1" if len(run_nums) == 0 else "run%i" % (max(run_nums) + 1)
    run_dir = run_root / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    return run_dir


def init_wandb(all_args, run_dir):
    if wandb is None:
        raise ImportError("wandb is not installed. Install requirements.txt or run with --no_wandb.")

    run_name = "{}_{}_{}_seed{}".format(
        all_args.algorithm_name,
        all_args.env_name,
        all_args.experiment_name,
        all_args.seed,
    )
    return wandb.init(
        project=all_args.env_name,
        entity=os.environ.get("WANDB_ENTITY"),
        group=all_args.user_name if all_args.user_name else None,
        name=run_name,
        config=vars(all_args),
        dir=str(run_dir),
        sync_tensorboard=False,
    )


def log(data, step=None):
    if wandb is not None and wandb.run is not None:
        wandb.log(data, step=step)


def save(path, base_path=None):
    if wandb is not None and wandb.run is not None and os.path.exists(path):
        wandb.save(path, base_path=base_path)


def finish():
    if wandb is not None and wandb.run is not None:
        wandb.finish()


class EvalVideoRecorder:
    def __init__(self, env, video_dir, episode, width=1280, height=720, fps=20):
        self.env = env
        self.video_dir = Path(video_dir)
        self.episode = episode
        self.width = width
        self.height = height
        self.fps = fps
        self.sensor = None
        self.frame_queue = queue.Queue()
        self.frames = []
        self.path = self.video_dir / ("eval_episode_%04d.mp4" % episode)

    def start(self):
        if not all(hasattr(self.env, attr) for attr in ("world", "blueprint_library", "spectator")):
            return False

        camera_bp = self.env.blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.width))
        camera_bp.set_attribute("image_size_y", str(self.height))
        camera_bp.set_attribute("fov", "90")
        transform = self.env.spectator.get_transform()
        self.sensor = self.env.world.spawn_actor(camera_bp, transform)
        self.sensor.listen(self.frame_queue.put)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        return True

    def capture(self):
        if self.sensor is None:
            return

        if hasattr(self.env, "spectator"):
            self.sensor.set_transform(self.env.spectator.get_transform())

        image = None
        while True:
            try:
                image = self.frame_queue.get_nowait()
            except queue.Empty:
                break

        if image is None:
            return

        frame = np.frombuffer(image.raw_data, dtype=np.uint8)
        frame = frame.reshape((image.height, image.width, 4))[:, :, :3]
        frame = frame[:, :, ::-1]
        self.frames.append(frame)

    def close(self, wandb_key=None, step=None):
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor.destroy()
            self.sensor = None

        if len(self.frames) == 0:
            return None

        try:
            import imageio

            imageio.mimsave(str(self.path), self.frames, fps=self.fps)
        except Exception:
            fallback_path = self.path.with_suffix(".npz")
            np.savez_compressed(str(fallback_path), frames=np.asarray(self.frames))
            self.path = fallback_path

        save(str(self.path), base_path=str(self.video_dir.parent))
        if wandb is not None and wandb.run is not None and wandb_key is not None:
            if self.path.suffix == ".mp4":
                wandb.log({wandb_key: wandb.Video(str(self.path), fps=self.fps, format="mp4")}, step=step)
            else:
                wandb.log({wandb_key: str(self.path)}, step=step)
        return str(self.path)
