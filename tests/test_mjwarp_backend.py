"""Numerical and lifecycle tests; optional when the MJWarp environment is absent."""

import importlib.util
import os
import subprocess
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("mujoco", "mujoco_warp", "warp"))
NATIVE_XML = """
<mujoco>
  <compiler autolimits="true"/>
  <option timestep="0.001" gravity="0 0 0" integrator="Euler"/>
  <extension><plugin plugin="mujoco.cable.unilateral"><instance name="cable">
    <config key="stiffness" value="100"/>
    <config key="damping" value="2"/>
    <config key="pretension_offset" value="0.02"/>
    <config key="max_tension" value="20"/>
  </instance></plugin></extension>
  <worldbody>
    <site name="anchor" pos="0 0 0"/>
    <body pos="1 0 0"><joint name="slide" type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.01" mass="1" contype="0" conaffinity="0"/>
      <site name="end"/>
    </body>
  </worldbody>
  <tendon><spatial name="route"><site site="anchor"/><site site="end"/></spatial></tendon>
  <actuator><plugin name="pull" instance="cable" tendon="route" ctrlrange="-0.2 0.2"/></actuator>
  <sensor><plugin name="state" instance="cable"/><actuatorfrc name="force" actuator="pull"/></sensor>
</mujoco>
"""


@unittest.skipUnless(AVAILABLE, "install requirements-warp.txt in a separate environment")
class MJWarpBackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mujoco
        import warp as wp
        from mujoco_cable_warp.model import load_model, UnsupportedCableError
        from mujoco_cable_warp.simulation import CableSimulation

        cls.mujoco = mujoco
        cls.wp = wp
        cls.load_model = staticmethod(load_model)
        cls.UnsupportedCableError = UnsupportedCableError
        cls.Simulation = CableSimulation
        cls.device = os.environ.get("CABLE_WARP_DEVICE", "cpu")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def model_file(self, xml=NATIVE_XML, **config):
        root = ET.fromstring(xml)
        instance = root.find("./extension/plugin/instance")
        for key, value in config.items():
            existing = instance.find(f"config[@key='{key}']")
            if existing is None:
                existing = ET.SubElement(instance, "config", key=key)
            existing.set("value", str(value))
        path = Path(self.temp.name) / "scene.xml"
        ET.ElementTree(root).write(path)
        return path

    def sim(self, path=None, nworld=1, **config):
        return self.Simulation(path or self.model_file(**config), device=self.device, nworld=nworld, nconmax=8, njmax=32)

    def test_model_conversion_preserves_controls_and_sensor_layout(self):
        loaded = self.load_model(self.model_file())
        self.assertEqual(loaded.model.nplugin, 0)
        self.assertEqual(loaded.model.nu, 1)
        self.assertEqual(loaded.model.sensor_dim.tolist(), [8, 1])
        self.assertEqual(loaded.model.sensor_adr.tolist(), [0, 8])
        self.assertEqual(loaded.cables[0].actuator_name, "pull")
        np.testing.assert_allclose(loaded.model.actuator_ctrlrange, [[-0.2, 0.2]])

    def test_native_law_force_sensor_clipping_slack_and_saturation(self):
        sim = self.sim(nworld=4)
        sim.set_control([[0.10], [-0.10], [0.5], [0.0]])
        sim.data.qvel.assign(np.array([[0.3], [0.3], [0.0], [-2.0]], dtype=np.float32))
        sim.forward()
        expected = [12.6, 0.0, 20.0, 0.0]
        np.testing.assert_allclose(sim.state.values.numpy()[:, 0, 5], expected, atol=2e-5)
        np.testing.assert_allclose(sim.data.qfrc_actuator.numpy()[:, 0], -np.array(expected), atol=2e-5)
        np.testing.assert_allclose(sim.data.sensordata.numpy()[:, 8], -np.array(expected), atol=2e-5)
        self.assertEqual(sim.state.values.numpy()[2, 0, 7], 1)
        np.testing.assert_array_equal(sim.data.sensordata.numpy()[:, :8], sim.state.values.numpy()[:, 0, :8])

    def test_native_trajectory_matches_independent_cpu_law(self):
        sim = self.sim()
        m = sim.cpu_model
        d = self.mujoco.MjData(m)
        sim.set_control([0.03])
        for _ in range(100):
            # Independent scalar mass/spring law for this one-DOF fixture.
            extension = 1.0 + float(d.qpos[0]) - (1.0 - 0.03 - 0.02)
            force = min(20.0, max(0.0, 100.0 * extension + 2.0 * float(d.qvel[0]))) if extension > 0 else 0.0
            d.qfrc_applied[0] = -force
            self.mujoco.mj_step(m, d)
            sim.step()
        np.testing.assert_allclose(sim.data.qpos.numpy()[0], d.qpos, atol=2e-6)
        np.testing.assert_allclose(sim.data.qvel.numpy()[0], d.qvel, atol=2e-5)

    def test_native_actuator_force_limit_is_preserved(self):
        root = ET.fromstring(NATIVE_XML)
        root.find("./actuator/plugin").set("forcerange", "-3 3")
        sim = self.sim(self.model_file(ET.tostring(root, encoding="unicode")))
        sim.set_control([0.1])
        sim.forward()
        self.assertAlmostEqual(sim.state.values.numpy()[0, 0, 5], 12, delta=2e-5)
        self.assertEqual(sim.data.actuator_force.numpy()[0, 0], -3)
        self.assertEqual(sim.data.qfrc_actuator.numpy()[0, 0], -3)

    def test_velocity_state_advances_once_and_reset_is_per_world(self):
        sim = self.sim(nworld=2, ctrl_mode="spool_velocity", max_contraction_rate="0.1")
        sim.set_control([[0.2], [0.05]])
        sim.forward()
        sim.forward()
        np.testing.assert_array_equal(sim.state.integrated.numpy(), 0)
        sim.step(5)
        np.testing.assert_allclose(sim.state.integrated.numpy()[:, 0], [0.0005, 0.00025], atol=1e-9)
        before = sim.data.qpos.numpy().copy()
        time_before = sim.data.time.numpy().copy()
        sim.reset([0])
        np.testing.assert_array_equal(sim.data.qpos.numpy()[1], before[1])
        self.assertEqual(sim.data.time.numpy()[1], time_before[1])
        self.assertEqual(sim.data.time.numpy()[0], 0)
        np.testing.assert_allclose(sim.state.integrated.numpy()[:, 0], [0, 0.00025], atol=1e-9)

    def test_filtered_control_and_taut_hysteresis(self):
        sim = self.sim(control_timeconstant="0.04", max_contraction_rate="0.025")
        sim.set_control([0.1])
        sim.step(4)
        np.testing.assert_allclose(sim.state.filtered.numpy(), [[0.0001]], atol=1e-9)
        sim = self.sim(pretension_offset="0", taut_hysteresis="0.01")
        sim.set_control([0.005])
        sim.forward()
        self.assertEqual(sim.state.values.numpy()[0, 0, 5], 0)
        sim.set_control([0.015])
        sim.forward()
        self.assertGreater(sim.state.values.numpy()[0, 0, 5], 1)
        sim.set_control([0.005])
        sim.forward()
        self.assertGreater(sim.state.values.numpy()[0, 0, 5], 0)

    def test_signed_joint_spool_applies_reaction_torque(self):
        root = ET.fromstring(NATIVE_XML)
        body = ET.SubElement(root.find("worldbody"), "body", pos="0 0 1")
        ET.SubElement(body, "joint", name="spool", type="hinge", axis="0 0 1")
        ET.SubElement(body, "geom", type="sphere", size="0.02", mass="1", contype="0", conaffinity="0")
        sim = self.sim(self.model_file(ET.tostring(root, encoding="unicode"), ctrl_mode="joint_spool_angle", spool_joint="spool", spool_radius="0.025", spool_reserve_length="0.03", spool_reaction_torque="true"))
        joint = self.mujoco.mj_name2id(sim.cpu_model, self.mujoco.mjtObj.mjOBJ_JOINT, "spool")
        qpos = sim.data.qpos.numpy().copy()
        qpos[0, sim.cpu_model.jnt_qposadr[joint]] = -2.0
        sim.data.qpos.assign(qpos)
        sim.forward()
        values = sim.state.values.numpy()[0, 0]
        self.assertAlmostEqual(values[3], -0.01, delta=1e-7)
        self.assertAlmostEqual(values[5], 1.0, delta=2e-5)
        torque = sim.data.qfrc_passive.numpy()[0, sim.cpu_model.jnt_dofadr[joint]]
        self.assertAlmostEqual(torque, 0.025, delta=1e-6)

    def test_guide_friction_transmits_capstan_tension(self):
        xml = """<mujoco><size nuser_site="1"/><option gravity="0 0 0" integrator="Euler"/>
        <extension><plugin plugin="mujoco.cable.unilateral"><instance name="cable">
          <config key="route_mode" value="surface"/><config key="route_tendon" value="route"/>
          <config key="stiffness" value="100"/><config key="pretension_offset" value="0.1"/>
          <config key="guide_friction_mu" value="0.45"/>
        </instance></plugin></extension><worldbody>
          <body pos="-1 0 0"><freejoint/><geom size="0.01" mass="1" contype="0" conaffinity="0"/><site name="start" user="1"/></body>
          <site name="guide" user="3"/>
          <body pos="0 -1 0"><freejoint/><geom size="0.01" mass="1" contype="0" conaffinity="0"/><site name="end" user="1"/></body>
        </worldbody><tendon><spatial name="route"><site site="start"/><site site="guide"/><site site="end"/></spatial></tendon>
        <sensor><plugin instance="cable" name="state"/></sensor></mujoco>"""
        sim = self.sim(self.model_file(xml))
        incoming, outgoing = sim.state.tensions.numpy()[0, 0, :2]
        self.assertAlmostEqual(incoming, 10, delta=2e-5)
        self.assertAlmostEqual(outgoing / incoming, np.exp(-0.45 * np.pi / 2), delta=1e-6)
        forces = sim.state.force.numpy()[0]
        self.assertAlmostEqual(forces[0], incoming, delta=1e-6)
        self.assertAlmostEqual(forces[7], outgoing, delta=1e-6)

    def test_single_cylinder_geometry_force_and_world_independence(self):
        sim = self.sim(ROOT / "cable_plugin_demos/15_cpp_plugin_surface_single_pulley.xml", nworld=2)
        initial = sim.state.values.numpy().copy()
        self.assertTrue(np.all(initial[:, 0, 9] < 2e-5))
        np.testing.assert_allclose(initial[:, 0, 5], 1.875, atol=2e-4)
        self.assertIsNone(sim._visual_points)
        self.assertEqual(sim.state.points.shape[2], 4)
        points, counts = sim.get_route_points(0)
        points, count = points[0], counts[0]
        host = sim.get_data()
        geom = sim.loaded.cables[0].geom_id
        local = (points[1:count - 1] - host.geom_xpos[geom]) @ host.geom_xmat[geom].reshape(3, 3)
        np.testing.assert_allclose(np.linalg.norm(local[:, :2], axis=1), 0.04, atol=1e-7)
        # Virtual-work check: vertical payload force equals -T dL/dz.
        base = sim.data.qpos.numpy().copy()
        force_z = sim.state.force.numpy()[0, 2]
        lengths = []
        for offset in (-1e-4, 1e-4):
            qpos = base.copy()
            qpos[0, 2] += offset
            sim.data.qpos.assign(qpos)
            sim.forward()
            lengths.append(sim.state.values.numpy()[0, 0, 0])
        derivative = (lengths[1] - lengths[0]) / 2e-4
        self.assertAlmostEqual(force_z, -initial[0, 0, 5] * derivative, delta=0.003)
        sim.data.qpos.assign(base)
        sim.set_control([[0.01], [-0.01]])
        sim.step(20)
        sim.check()
        self.assertGreater(sim.data.qpos.numpy()[0, 2], sim.data.qpos.numpy()[1, 2])
        visual, counts = sim.get_route_points(1)
        np.testing.assert_allclose(visual[0, counts[0] - 1], sim.state.points.numpy()[1, 0, 3], atol=1e-7)
        from mujoco_cable_warp.cli import draw_cables

        scene = self.mujoco.MjvScene(sim.cpu_model, maxgeom=1000)
        draw_cables(sim, scene, 0)
        self.assertGreater(scene.ngeom, 10)

    def test_surface_invalid_route_disables_force(self):
        sim = self.sim(ROOT / "cable_plugin_demos/15_cpp_plugin_surface_single_pulley.xml")
        qpos = sim.data.qpos.numpy()
        qpos[0, :3] = [0, 0, 0.266]  # hook lies on the cylinder axis
        sim.data.qpos.assign(qpos)
        sim.forward()
        self.assertEqual(sim.state.values.numpy()[0, 0, 8], 2)
        self.assertEqual(sim.state.values.numpy()[0, 0, 5], 0)
        np.testing.assert_array_equal(sim.state.force.numpy(), 0)
        with self.assertRaisesRegex(RuntimeError, "Invalid surface route"):
            sim.check()
        _, counts = sim.get_route_points()
        self.assertEqual(counts[0], 0)

    def test_analytic_helix_virtual_work_and_display_isolation(self):
        xml = """<mujoco><size nuser_site="1"/><option gravity="0 0 0" integrator="Euler"/>
        <extension><plugin plugin="mujoco.cable.unilateral"><instance name="cable">
          <config key="route_mode" value="surface"/><config key="route_tendon" value="route"/>
          <config key="wrap_geoms" value="wheel"/><config key="stiffness" value="1000"/>
          <config key="pretension_offset" value="0.01"/><config key="damping" value="0"/>
        </instance></plugin></extension><worldbody>
          <body pos="-1 0 0.06"><freejoint/><geom size="0.03" mass="1" contype="0" conaffinity="0"/>
            <site name="start" pos="0.01 0.02 0.03" user="1"/></body>
          <body pos="1 0 0.12"><freejoint/><geom size="0.03" mass="1" contype="0" conaffinity="0"/>
            <site name="end" pos="-0.02 0.01 -0.02" user="1"/></body>
          <body quat="0.984807753 0.173648178 0 0"><freejoint/>
            <geom name="wheel" type="cylinder" size="0.2 0.8" mass="1" contype="0" conaffinity="0"/>
            <site name="hint" pos="0 0.3 0" user="2"/>
          </body>
        </worldbody><tendon><spatial name="route"><site site="start"/><site site="hint"/><site site="end"/></spatial></tendon>
        <sensor><plugin instance="cable" name="state"/></sensor></mujoco>"""
        sim = self.sim(self.model_file(xml), nworld=2)
        m = sim.cpu_model
        base = sim.data.qpos.numpy().copy()
        force = sim.state.force.numpy()[0].copy()
        tension = sim.state.values.numpy()[0, 0, 5]
        # Differentiate length in every translational/rotational DOF, including
        # tilted cylinder motion. This independently checks reaction torques.
        gradient = []
        epsilon = 0.001
        for dof in range(m.nv):
            lengths = []
            for sign in (-1, 1):
                pose = base[0].astype(np.float64)
                direction = np.zeros(m.nv)
                direction[dof] = sign
                self.mujoco.mj_integratePos(m, pose, direction, epsilon)
                qpos = base.copy()
                qpos[0] = pose
                sim.data.qpos.assign(qpos)
                sim.forward()
                lengths.append(float(sim.state.values.numpy()[0, 0, 0]))
            gradient.append((lengths[1] - lengths[0]) / (2 * epsilon))
        np.testing.assert_allclose(force, -tension * np.array(gradient), atol=0.004)
        sim.data.qpos.assign(base)
        velocities = np.linspace(-0.2, 0.3, m.nv).astype(np.float32)
        sim.data.qvel.assign(np.stack([velocities, -velocities]))
        sim.forward()
        values = sim.state.values.numpy().copy()
        np.testing.assert_allclose(values[:, 0, 1], [np.dot(gradient, velocities), -np.dot(gradient, velocities)], atol=1e-4)
        np.testing.assert_allclose(np.sum(sim.state.force.numpy() * sim.data.qvel.numpy(), axis=1), -values[:, 0, 5] * values[:, 0, 1], atol=1e-5)

        # Check exact helix length in the cylinder frame, independently of
        # display resolution. Four physical points include both tangent points.
        host = sim.get_data()
        geom = sim.loaded.cables[0].geom_id
        points = sim.state.points.numpy()[0, 0]
        local = (points - host.geom_xpos[geom]) @ host.geom_xmat[geom].reshape(3, 3)
        cross_z = local[1, 0] * local[2, 1] - local[1, 1] * local[2, 0]
        sweep = abs(np.arctan2(cross_z, np.dot(local[1, :2], local[2, :2])))
        exact = np.linalg.norm(local[1] - local[0]) + np.hypot(0.2 * sweep, local[2, 2] - local[1, 2]) + np.linalg.norm(local[3] - local[2])
        self.assertAlmostEqual(values[0, 0, 0], exact, delta=5e-7)
        snapshots = {name: getattr(sim.state, name).numpy().copy() for name in ("values", "points", "force", "integrated", "filtered", "latch", "branch")}
        for world in (1, 0, 1):
            visual, counts = sim.get_route_points(world)
            self.assertGreater(counts[0], 10)
            np.testing.assert_allclose(visual[0, 0], points[0], atol=1e-7)
            np.testing.assert_allclose(visual[0, counts[0] - 1], points[3], atol=1e-7)
        for name, expected in snapshots.items():
            np.testing.assert_array_equal(getattr(sim.state, name).numpy(), expected)

    def test_moving_cylinder_reaction_and_friction_dissipation(self):
        xml = """<mujoco><size nuser_site="1"/><option gravity="0 0 0" integrator="Euler"/>
        <extension><plugin plugin="mujoco.cable.unilateral"><instance name="cable">
          <config key="route_mode" value="surface"/><config key="route_tendon" value="route"/>
          <config key="wrap_geoms" value="wheel"/><config key="stiffness" value="1000"/>
          <config key="pretension_offset" value="0.01"/>
        </instance></plugin></extension><worldbody>
          <site name="start" pos="-1 0 0" user="1"/><site name="end" pos="1 0 0" user="1"/>
          <body><freejoint/><geom name="wheel" type="cylinder" size="0.2 0.2" mass="1"/>
            <site name="hint" pos="0 0.3 0" user="2"/>
          </body>
        </worldbody><tendon><spatial name="route"><site site="start"/><site site="hint"/><site site="end"/></spatial></tendon>
        <sensor><plugin instance="cable" name="state"/></sensor></mujoco>"""
        sim = self.sim(self.model_file(xml))
        tension = sim.state.values.numpy()[0, 0, 5]
        initial_force = sim.state.force.numpy()[0].copy()
        np.testing.assert_allclose(initial_force, [0, -0.4 * tension, 0, 0, 0, 0], atol=2e-5)
        sim.data.qvel.assign(np.array([[0, 0.2, 0, 0, 0, 1.3]], dtype=np.float32))
        sim.forward()
        values = sim.state.values.numpy()[0, 0]
        power = np.dot(sim.state.force.numpy()[0], sim.data.qvel.numpy()[0])
        self.assertAlmostEqual(power, -values[5] * values[1], delta=2e-5)
        rough = self.sim(self.model_file(xml, capstan_mu="0.2", capstan_direction="velocity"))
        self.assertEqual(rough.state.points.shape[2], 98)
        self.assertGreater(rough.state.count.numpy()[0, 0], 4)
        for speed in (-1.0, 1.0):
            rough.data.qvel.assign(np.array([[0, 0, 0, 0, 0, speed]], dtype=np.float32))
            rough.forward()
            self.assertLess(rough.state.force.numpy()[0, 5] * speed, -0.01)

    def test_unsupported_models_fail_explicitly(self):
        for demo, message in [("09", "composite"), ("14", "mesh")]:
            path = next((ROOT / "cable_plugin_demos").glob(f"{demo}_*.xml"))
            with self.subTest(demo=demo), self.assertRaisesRegex(self.UnsupportedCableError, message):
                self.load_model(path)
        with self.assertRaisesRegex(self.UnsupportedCableError, "implicit_compliant"):
            self.load_model(self.model_file(integration_mode="implicit_compliant"))

    def test_tensegrity_demo_runs_without_cpp_library(self):
        sim = self.sim(ROOT / "cable_plugin_demos/17_cpp_plugin_three_strut_nine_cable.xml", nworld=2)
        self.assertEqual(sim.ncable, 9)
        sim.step(5)
        sim.check()
        np.testing.assert_allclose(sim.data.qpos.numpy()[0], sim.data.qpos.numpy()[1], atol=1e-7)

    def test_unified_launcher_dispatches_without_plugin_library(self):
        result = subprocess.run(["bash", str(ROOT / "scripts/run_demo.sh"), "15", "--backend", "mjwarp", "--headless", "--device", "cpu"], env={**os.environ, "MUJOCO_PYTHON": "/bin/echo"}, capture_output=True, text=True, check=True)
        self.assertIn("run_mjwarp_demo.py", result.stdout)
        self.assertIn("--headless --device cpu", result.stdout)

    def test_cuda_graph_matches_uncaptured_steps(self):
        if not self.wp.get_device(self.device).is_cuda:
            self.skipTest("CUDA device required")
        reference = self.sim(nworld=2, ctrl_mode="spool_velocity")
        captured = self.sim(nworld=2, ctrl_mode="spool_velocity")
        reference.step()
        captured.step()
        reference.reset()
        captured.reset()
        captured.capture_step()
        for controls in ([[0.03], [0.06]], [[-0.02], [0.04]]):
            reference.set_control(controls)
            captured.set_control(controls)
            reference.step(20)
            captured.step(20)
            for name in ("qpos", "qvel", "time"):
                np.testing.assert_allclose(getattr(captured.data, name).numpy(), getattr(reference.data, name).numpy(), atol=1e-6)
            for name in ("values", "integrated", "filtered", "latch"):
                np.testing.assert_allclose(getattr(captured.state, name).numpy(), getattr(reference.state, name).numpy(), atol=1e-6)
            # Replaying an existing graph must also respect selective resets.
            reference.reset([1])
            captured.reset([1])

    def test_analytic_cuda_graph_survives_display_and_partial_reset(self):
        if not self.wp.get_device(self.device).is_cuda:
            self.skipTest("CUDA device required")
        path = ROOT / "cable_plugin_demos/15_cpp_plugin_surface_single_pulley.xml"
        reference = self.sim(path, nworld=2)
        captured = self.sim(path, nworld=2)
        for sim in (reference, captured):
            sim.step()
            sim.reset()
            sim.set_control([[0.01], [-0.01]])
        captured.capture_step()
        for _ in range(2):
            reference.step(20)
            captured.step(20)
            captured.get_route_points(1)
            captured.get_route_points(0)
            for name in ("qpos", "qvel", "time"):
                np.testing.assert_allclose(getattr(captured.data, name).numpy(), getattr(reference.data, name).numpy(), atol=1e-6)
            np.testing.assert_allclose(captured.state.values.numpy(), reference.state.values.numpy(), atol=1e-6)
            for sim in (reference, captured):
                sim.reset([1])
                sim.set_control([[0.02], [0.01]])


if __name__ == "__main__":
    unittest.main()
