"""Isaac Sim 5.0: original warehouse + physical Nova Carter mapping circuit.

Run from repository root with Isaac Python. No model training or navigation
policy evaluation: a deterministic waypoint controller supplies wheel commands.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--navigation-root", type=Path)
    parser.add_argument(
        "--navigation-task", choices=("landmark", "explore", "follow", "instruction")
    )
    parser.add_argument(
        "--person-asset",
        default="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/People/Characters/male_adult_construction_01_new/male_adult_construction_01_new.usd",
    )
    parser.add_argument("--max-steps", type=int, default=18000)
    parser.add_argument("--capture-hz", type=float, default=1.0)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--live-map", type=Path, help="Incremental map output directory"
    )
    parser.add_argument("--render-hz", type=int, default=30)
    parser.add_argument(
        "--live-capture-mode",
        choices=("synchronized", "replicator"),
        default="synchronized",
    )
    parser.add_argument(
        "--realtime", action="store_true", help="Pace simulation against wall clock"
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Export the labeled warehouse and robot without starting a circuit",
    )
    from DynaNav.warehouse_loop.landmark_variants import VARIANTS

    parser.add_argument("--landmark-variant", choices=VARIANTS)
    parser.add_argument("--scene-person", action="store_true")
    parser.add_argument(
        "--scene-layout", choices=("warehouse", "junction-fixture"), default="warehouse"
    )
    args = parser.parse_args()
    if (
        not 0 < args.capture_hz <= 60
        or not 0 < args.render_hz <= 60
        or 60 % args.render_hz
    ):
        parser.error("capture-hz must be in (0,60]; render-hz must divide 60")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": not args.gui,
            "width": 960,
            "height": 720,
            "renderer": "RayTracedLighting",
        }
    )
    capture = None
    perf_stream = None
    try:
        import omni.usd
        import omni.replicator.core as rep
        from pxr import Gf, UsdGeom, Usd
        from isaacsim.core.api import World
        from isaacsim.core.utils.rotations import quat_to_euler_angles
        from isaacsim.core.utils.viewports import set_camera_view
        from isaacsim.robot.wheeled_robots.robots import WheeledRobot
        from isaacsim.robot.wheeled_robots.controllers.differential_controller import (
            DifferentialController,
        )
        from DynaNav.warehouse_loop.scene import WAREHOUSE, ROBOT, WAYPOINTS, add_labels
        from ticvla.warehouse.collect import EpisodeRecorder, camera_calibration

        print("LOAD_WAREHOUSE", flush=True)
        # New local overlay references the original stage, preserving its original prim paths.
        context = omni.usd.get_context()
        context.new_stage()
        stage = context.get_stage()
        if args.scene_layout == "warehouse":
            stage.GetRootLayer().subLayerPaths.append(WAREHOUSE)
        stage.SetEditTarget(stage.GetRootLayer())
        for _ in range(20):
            app.update()
        if args.scene_layout == "warehouse":
            labels = add_labels(stage, root, args.landmark_variant)
        else:
            from DynaNav.warehouse_loop.navigation_scene import build_junction_fixture

            labels = build_junction_fixture(stage, root)
        navigation = pedestrian = None
        if args.navigation_root:
            from DynaNav.warehouse_loop.navigation_control import (
                NavigationControl,
                Pedestrian,
            )

            navigation = NavigationControl(args.navigation_root)
            if args.navigation_task == "follow" or args.scene_person:
                pedestrian = Pedestrian(stage, args.navigation_root, args.person_asset)
        if args.landmark_variant == "lighting":
            # Change scene lighting in USD, not RGB post-processing. Depth,
            # calibration and evaluation masks remain synchronized.
            for prim in stage.Traverse():
                if prim.GetTypeName().endswith("Light"):
                    intensity = prim.GetAttribute("inputs:intensity")
                    if intensity and intensity.Get() is not None:
                        intensity.Set(float(intensity.Get()) * 0.2)
        labels_only = root / "warehouse-labels-only.usda"
        stage.GetRootLayer().Export(str(labels_only))
        template = Path("DynaNav/configs/data_collection/warehouse.yaml").read_text()
        (root / "dynanav-warehouse.yaml").write_text(
            template.replace(WAREHOUSE, str(labels_only))
        )
        world = World(
            stage_units_in_meters=1.0,
            physics_dt=1 / 60,
            rendering_dt=1 / args.render_hz,
        )
        robot = world.scene.add(
            WheeledRobot(
                prim_path="/World/MainRobot",
                name="main_robot",
                create_robot=True,
                usd_path=ROBOT,
                wheel_dof_names=["joint_wheel_left", "joint_wheel_right"],
                position=np.array([*WAYPOINTS[0], 0.02]),
                orientation=np.array([math.sqrt(0.5), 0, 0, math.sqrt(0.5)]),
            )
        )
        for _ in range(10):
            app.update()
        body = "/World/MainRobot/chassis_link"
        if not stage.GetPrimAtPath(body).IsValid():
            raise ValueError("Nova Carter chassis_link not found")
        camera_path = body + "/MappingCamera"
        camera = UsdGeom.Camera.Define(stage, camera_path)
        camera.CreateFocalLengthAttr(16.0)
        camera.CreateHorizontalApertureAttr(24.0)
        camera.CreateVerticalApertureAttr(18.0)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 60.0))
        camera.AddTransformOp().Set(
            Gf.Matrix4d()
            .SetLookAt(
                Gf.Vec3d(0.15, 0, 1.45), Gf.Vec3d(8.15, 0, 1.95), Gf.Vec3d(0, 0, 1)
            )
            .GetInverse()
        )
        overview = UsdGeom.Camera.Define(stage, "/World/Overview")
        overview.CreateFocalLengthAttr(6.0)
        overview.CreateHorizontalApertureAttr(24.0)
        overview.CreateVerticalApertureAttr(18.0)
        overview.AddTransformOp().Set(
            Gf.Matrix4d()
            .SetLookAt(
                Gf.Vec3d(-10.5, 16.5, 8.7), Gf.Vec3d(-10.5, 16.5, 0), Gf.Vec3d(0, 1, 0)
            )
            .GetInverse()
        )
        set_camera_view(eye=[-13.0, 2.0, 3.4], target=[-13.0, 13.0, 2.0])
        # Save before physics modifies any poses. Can be opened directly in Isaac GUI.
        stage.GetRootLayer().Export(str(root / "warehouse-labeled.usda"))
        if args.build_only:
            print("BUILT", str(root / "warehouse-labeled.usda"), flush=True)
            return
        world.reset()
        drive = DifferentialController(
            name="loop_drive", wheel_radius=0.14, wheel_base=0.413
        )
        # Isaac 5.0 Nova Carter drive wheels have 0.14 m radius; baseline legacy
        # behavior uses .152 m. Closed-loop pose control tolerates small slip.
        for _ in range(60):
            world.step(render=True)
        metadata = {
            "layout_family": "dynanav-simple-warehouse-labeled",
            "landmark_variant": args.landmark_variant,
            "episode_id": root.parent.name if args.live_map else root.name,
            "task": {
                "task_id": "warehouse-loop",
                "instruction": "Map the shelf markers during one closed circuit.",
                "hard_constraints": [],
            },
            "camera_prim": camera_path,
            "body_prim": body,
            "resolution": [960, 720],
            "capture_hz": args.capture_hz,
            "capture_oracle": not bool(args.live_map),
            "png_compress_level": 0 if args.live_map else 6,
            "live_capture_mode": args.live_capture_mode if args.live_map else None,
            "supervision": {"actions_task_aligned": False},
        }
        if navigation and (args.navigation_root / "task.json").exists():
            metadata["task"] = json.loads(
                (args.navigation_root / "task.json").read_text()
            )
        config = root / "capture.json"
        config.write_text(json.dumps(metadata, indent=2))
        rep.orchestrator.set_capture_on_play(False)
        recorder_class = EpisodeRecorder
        if args.live_map:
            from DynaNav.warehouse_loop.live_recording import AsyncEpisodeRecorder

            recorder_class = AsyncEpisodeRecorder
        capture = recorder_class(
            root / "raw",
            {
                **metadata,
                "pose_source": (
                    "synchronized_render_camera_pose_checked_against_physx"
                    if args.live_map and args.live_capture_mode == "synchronized"
                    else "synchronized_simulated_odometry"
                ),
                "timestamp_source": "frozen_physics_time",
            },
        )
        product = rep.create.render_product(camera_path, (960, 720))
        overview_rgb = overview_product = None
        if not args.live_map:
            overview_product = rep.create.render_product(
                str(overview.GetPath()), (960, 720)
            )
            overview_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
            overview_rgb.attach(overview_product)
        (root / "overview").mkdir()
        annotators = {
            name: rep.AnnotatorRegistry.get_annotator(name)
            for name in (
                "rgb",
                "distance_to_image_plane",
                "camera_params",
            )
            + (() if args.live_map else ("instance_segmentation",))
        }
        for annotator in annotators.values():
            annotator.attach(product)
        transforms = UsdGeom.XformCache(Usd.TimeCode.Default())
        twb = np.array(transforms.GetLocalToWorldTransform(stage.GetPrimAtPath(body))).T
        twc = np.array(
            transforms.GetLocalToWorldTransform(camera.GetPrim())
        ).T @ np.diag([1.0, -1.0, -1.0, 1.0])
        extrinsic = np.linalg.inv(twb) @ twc
        origin = None
        from DynaNav.warehouse_loop.live_metrics import atomic_json, rate_report

        perf_stream = (root / "performance.jsonl").open("w", buffering=1)
        capture_perf, render_perf, physics_perf, panel_perf = [], [], [], []
        panel = None
        if args.live_map and args.gui:
            from DynaNav.warehouse_loop.live_panel import LiveMapPanel
            from omni.kit.viewport.utility import get_active_viewport

            panel = None if args.navigation_root else LiveMapPanel(args.live_map)
            get_active_viewport().camera_path = "/World/Overview"

        def metric(module, rows, started, **extra):
            row = dict(
                wall=time.time(),
                service_s=time.perf_counter() - started,
                t_obs=world.current_time,
                **extra,
            )
            rows.append(row)
            perf_stream.write(json.dumps(dict(module=module, **row)) + "\n")

        def snapshot():
            nonlocal origin
            started = time.perf_counter()
            # Freeze physics while rendering RGB, optical-Z depth and calibration.
            # Render reference time is NOT physics time when rendering sparsely.
            stamp = world.current_time
            source_monotonic = time.monotonic()
            direct_render = (
                bool(args.live_map)
                and args.live_capture_mode == "synchronized"
                and origin is not None
            )
            if direct_render:
                # Keep the timeline playing, but explicitly schedule and await
                # a zero-delta capture. Plain render/get_data can return a stale
                # annotator buffer even after multiple renders at frozen time.
                world.render()
                rep.orchestrator.step(
                    rt_subframes=1,
                    delta_time=0.0,
                    pause_timeline=False,
                    wait_for_render=True,
                )
            else:
                world.pause()
                world.render()
                rep.orchestrator.step(
                    rt_subframes=1 if args.live_map else 3,
                    delta_time=0.0,
                    pause_timeline=True,
                )
            render_done = time.perf_counter()
            data = {key: value.get_data() for key, value in annotators.items()}
            readback_done = time.perf_counter()
            K, camera_pose = camera_calibration(data["camera_params"])
            body_pose = camera_pose @ np.linalg.inv(extrinsic)
            position_error = float(
                np.linalg.norm(body_pose[:2, 3] - robot.get_world_pose()[0][:2])
            )
            if args.live_map and position_error > 0.03:
                raise RuntimeError(
                    f"RGB-D source pose lags current physics by {position_error:.4f} m"
                )
            if origin is None:
                origin = np.linalg.inv(body_pose)
                (root / "evaluation-origin.json").write_text(
                    json.dumps({"T_world_odom": body_pose.tolist()})
                )
            privileged = None
            if overview_rgb is not None:
                from PIL import Image

                Image.fromarray(overview_rgb.get_data()[..., :3]).save(
                    root / "overview" / f"{capture.count:08d}.png"
                )
                instance = data["instance_segmentation"]
                privileged = {
                    "instance_ids": instance["data"],
                    "instance_labels_json": np.frombuffer(
                        json.dumps(instance["info"]["idToLabels"]).encode(),
                        dtype=np.uint8,
                    ),
                }
            capture.append(
                rgb=data["rgb"],
                depth=data["distance_to_image_plane"],
                K=K,
                T_body_camera=extrinsic,
                T_odom_body=origin @ body_pose,
                t_obs=stamp,
                odom_epoch=metadata["episode_id"],
                privileged=privileged,
                source_monotonic=source_monotonic,
            )
            if abs(world.current_time - stamp) > 1e-6:
                raise RuntimeError("Physics advanced during synchronized capture")
            if not direct_render:
                world.play()
            metric(
                "capture",
                capture_perf,
                started,
                source_position_error_m=position_error,
                render_s=render_done - started,
                readback_s=readback_done - render_done,
                write_s=time.perf_counter() - readback_done,
            )

        snapshot()
        run_wall_start = time.time()
        run_clock_start = time.perf_counter()
        run_sim_start = world.current_time
        trace = (root / "trajectory.jsonl").open("w")
        target_index = 1
        reached = []
        final_heading = math.pi / 2
        finished = False
        initial = None
        distance = 0.0
        previous = None
        print("DRIVING", flush=True)
        for step in range(args.max_steps):
            if not app.is_running():
                raise RuntimeError("Isaac window closed before capture finished")
            step_started = time.perf_counter()
            pos, quat = robot.get_world_pose()
            yaw = float(quat_to_euler_angles(quat)[2])
            if initial is None:
                initial = pos.copy()
            if previous is not None:
                distance += float(np.linalg.norm(pos[:2] - previous[:2]))
            previous = pos.copy()
            if navigation:
                # Replace fixed-loop output completely: only source-checked external commands execute.
                pose = np.eye(4)
                pose[:2, :2] = [
                    [math.cos(yaw), -math.sin(yaw)],
                    [math.sin(yaw), math.cos(yaw)],
                ]
                pose[:3, 3] = pos
                if step % 2 == 0:
                    nav_v, nav_w = navigation.command(
                        origin @ pose, metadata["episode_id"], world.current_time
                    )
                v, w = nav_v, nav_w
                finished = (args.navigation_root / "task-finished.json").exists() or (
                    args.navigation_root.parent / "cancel.json"
                ).exists()
                if pedestrian:
                    pedestrian.update(world.current_time)
            else:
                if target_index < len(WAYPOINTS):
                    delta = np.asarray(WAYPOINTS[target_index]) - pos[:2]
                    error = float(np.linalg.norm(delta))
                    heading = math.atan2(delta[1], delta[0])
                    if error < 0.14:
                        reached.append(
                            {
                                "index": target_index,
                                "time": world.current_time,
                                "position": pos.tolist(),
                            }
                        )
                        target_index += 1
                        print("WAYPOINT", target_index - 1, pos.tolist(), flush=True)
                if target_index == len(WAYPOINTS):
                    heading, error = final_heading, 0.0
                angular_error = math.atan2(
                    math.sin(heading - yaw), math.cos(heading - yaw)
                )
                v = (
                    min(0.65, error * 1.2) * max(0.0, math.cos(angular_error))
                    if abs(angular_error) < 0.45
                    else 0.0
                )
                w = float(np.clip(2.0 * angular_error, -0.8, 0.8))
                if target_index == len(WAYPOINTS) and abs(angular_error) < 0.035:
                    v, w = 0.0, 0.0
                    finished = True
            robot.apply_wheel_actions(drive.forward(command=np.array([v, w])))
            world.step(render=False)
            metric("physics", physics_perf, step_started)
            capture_due = (
                world.current_time - capture.last_t >= 1 / args.capture_hz - 1e-6
            )
            captured = False
            if args.live_map and (step + 1) % (60 // args.render_hz) == 0:
                render_started = time.perf_counter()
                if panel:
                    panel_started = time.perf_counter()
                    if panel.update():
                        metric("map_panel", panel_perf, panel_started)
                # A synchronized sensor capture already renders the viewport.
                # Share its scheduled slot instead of rendering it twice.
                if capture_due:
                    snapshot()
                    captured = True
                else:
                    world.render()
                metric("viewport_render", render_perf, render_started)
            if capture_due and not captured:
                snapshot()
            if args.realtime:
                delay = (world.current_time - run_sim_start) - (
                    time.perf_counter() - run_clock_start
                )
                if delay > 0:
                    time.sleep(delay)
            if step % 6 == 0:
                trace.write(
                    json.dumps(
                        {
                            "step": step,
                            "t": world.current_time,
                            "position": pos.tolist(),
                            "yaw": yaw,
                            "v_command": v,
                            "w_command": w,
                            "target": (
                                args.navigation_task if navigation else target_index
                            ),
                        }
                    )
                    + "\n"
                )
                trace.flush()
            if step % 600 == 0:
                print(
                    "PROGRESS",
                    step,
                    target_index,
                    pos.tolist(),
                    "frames",
                    capture.count,
                    flush=True,
                )
            if finished:
                for _ in range(60):
                    world.step(render=False)
                snapshot()
                break
        trace.close()
        if navigation:
            navigation.close()
        run_wall_end = time.time()
        robot.apply_wheel_actions(drive.forward(command=np.zeros(2)))
        duration = world.current_time
        for annotator in annotators.values():
            annotator.detach(product)
        if overview_rgb is not None:
            overview_rgb.detach(overview_product)
        world.pause()
        rep.orchestrator.stop()
        frame_count = capture.count
        capture.close()
        writer_perf = getattr(capture, "metrics", [])
        capture = None
        report = {
            "isaac_version": Path("/data/isaacsim/VERSION").read_text().strip(),
            "scene_asset": WAREHOUSE,
            "robot_asset": ROBOT,
            "shelf_count": len(labels),
            "waypoints": [] if navigation else WAYPOINTS,
            "navigation_task": args.navigation_task,
            "execution_source": (
                "navigation_command_watchdog" if navigation else "fixed_loop"
            ),
            "reached": reached,
            "loop_complete": finished if not navigation else False,
            "navigation_finished": finished if navigation else None,
            "distance_m": distance,
            "return_error_m": float(np.linalg.norm(pos[:2] - initial[:2])),
            "frames": frame_count,
            "simulation_seconds": duration,
            "motion": "PhysX articulation; DifferentialController wheel velocities; no pose teleport after initialization",
            "odometry": "synchronized simulated pose, not wheel odometry or SLAM",
        }
        active_capture = [r for r in capture_perf if r["wall"] >= run_wall_start]
        perf = {
            "run_wall_start": run_wall_start,
            "run_wall_end": run_wall_end,
            "wall_seconds": run_wall_end - run_wall_start,
            "simulation_elapsed_seconds": duration - run_sim_start,
            "real_time_factor": (duration - run_sim_start)
            / (run_wall_end - run_wall_start),
            "physics": rate_report(physics_perf, 60, run_wall_start, run_wall_end),
            "viewport_render": rate_report(
                render_perf, args.render_hz, run_wall_start, run_wall_end
            ),
            "capture": rate_report(
                active_capture, args.capture_hz, run_wall_start, run_wall_end
            ),
            "map_panel": rate_report(
                panel_perf, args.capture_hz, run_wall_start, run_wall_end
            ),
            "record_write": rate_report(
                writer_perf, args.capture_hz, run_wall_start, run_wall_end
            ),
            "capture_source_hz": (
                (len(capture_perf) - 1)
                / (capture_perf[-1]["t_obs"] - capture_perf[0]["t_obs"])
                if len(capture_perf) > 1
                else 0.0
            ),
            "viewport_metric": "Scheduled viewport update completions, with synchronized capture sharing its slot; not GPU presentation FPS",
        }
        atomic_json(root / "performance-report.json", perf)
        (root / "capture-report.json").write_text(json.dumps(report, indent=2))
        print("LOOP_REPORT", json.dumps(report), flush=True)
    except BaseException:
        import traceback

        failure = traceback.format_exc()
        print(failure, flush=True)
        (root / "capture-error.txt").write_text(failure)
        raise
    finally:
        if capture:
            capture.close()
        if perf_stream:
            perf_stream.close()
        app.close()


if __name__ == "__main__":
    main()
