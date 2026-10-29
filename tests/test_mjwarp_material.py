"""Conservative cable material history on a synthetic two-cylinder mechanism.

Optional CABLE_WARP_REFERENCE_PLUGIN and CABLE_WARP_HINGE_REFERENCE_PLUGIN
enable comparisons with C++ libraries built against this interpreter's MuJoCo.
"""

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/two_cylinder_material.xml"
AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("mujoco", "mujoco_warp", "warp"))


@unittest.skipUnless(AVAILABLE, "install the warp extra")
class CableMaterialWarpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mujoco
        import warp as wp
        from mujoco_cable_warp import CableSimulation

        cls.mj, cls.wp, cls.Simulation = mujoco, wp, CableSimulation
        cls.device = os.environ.get("CABLE_WARP_DEVICE", "cpu")
        cls.cable_reference = bool(os.environ.get("CABLE_WARP_REFERENCE_PLUGIN"))
        cls.reference = cls.cable_reference and bool(os.environ.get("CABLE_WARP_HINGE_REFERENCE_PLUGIN"))
        for name in ("CABLE_WARP_REFERENCE_PLUGIN", "CABLE_WARP_HINGE_REFERENCE_PLUGIN"):
            if os.environ.get(name):
                mujoco.mj_loadPluginLibrary(str(Path(os.environ[name]).resolve()))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def model_file(self, *, native=True, hinges=True, **options):
        root = ET.parse(FIXTURE).getroot()
        if not hinges:
            extension = root.find("extension")
            extension.remove(extension.find("plugin[@plugin='mjhinge.hinge_friction']"))
        instance = root.find("./extension/plugin/instance[@name='cable']")
        for key, value in options.items():
            node = instance.find(f"config[@key='{key}']")
            if node is None:
                node = ET.SubElement(instance, "config", key=key)
            node.set("value", str(value))
        if not native:
            tendon = root.find("./tendon/spatial")
            for child in list(tendon):
                tendon.remove(child)
            for site in ("end", "guide_a", "guide_b", "anchor"):
                ET.SubElement(tendon, "site", site=site)
        path = Path(self.temp.name) / "model.xml"
        ET.ElementTree(root).write(path)
        return path

    def sim(self, path=None, nworld=1, **options):
        return self.Simulation(path or self.model_file(**options), device=self.device, nworld=nworld, nconmax=32, njmax=128)

    def test_sensor_layout_and_independent_friction_coefficients(self):
        sim = self.sim(nworld=2)
        self.assertEqual(sim.cpu_model.nplugin, 0)
        np.testing.assert_array_equal(sim.cpu_model.sensor_dim, [12, 3 * sim.cpu_model.nbody, 8])
        np.testing.assert_allclose(sim.topology.contact_mu.numpy()[0], [.2, .4])
        self.assertGreater(sim.state.values.numpy()[0, 0, 0], 0.2)
        self.assertEqual(sim.state.count.numpy()[0, 0], 7)
        for sensor in sim.loaded.cables[0].sensor_ids:
            self.assertIn(sim.cpu_model.sensor(sensor).name, ("state", "body_forces", "capstans"))

    def test_49_poses_lengths_contacts_and_frictionless_virtual_work(self):
        sim = self.sim(hinges=False, guide_friction_mus="0 0", home_length=".100")
        m = sim.cpu_model
        host = self.mj.MjData(m)
        for a in np.linspace(0, 90, 7):
            for b in np.linspace(0, 90, 7):
                q = np.deg2rad([a, b]).astype(np.float32)
                sim.reset()
                sim.data.qpos.assign(q[None])
                sim.forward()
                sim.check()
                host.qpos[:] = q
                self.mj.mj_forward(m, host)
                length = sim.state.values.numpy()[0, 0, 0]
                self.assertAlmostEqual(length, host.ten_length[0], delta=8e-8)
        for q in ((0, 0), (25, 40), (70, 65), (90, 90)):
            q = np.deg2rad(q).astype(np.float32)
            sim.reset()
            sim.data.qpos.assign(q[None])
            sim.forward()
            host.qpos[:] = q
            self.mj.mj_forward(m, host)
            jacobian = []
            for axis in range(2):
                host.qpos[:] = q
                host.qpos[axis] += 1e-6
                self.mj.mj_forward(m, host)
                plus = host.ten_length[0]
                host.qpos[axis] -= 2e-6
                self.mj.mj_forward(m, host)
                jacobian.append((plus - host.ten_length[0]) / 2e-6)
            tension = sim.state.values.numpy()[0, 0, 5]
            np.testing.assert_allclose(sim.state.force.numpy()[0], -tension * np.array(jacobian), atol=3e-5, rtol=2e-4)

    def test_material_conservation_cones_memory_and_forward_isolation(self):
        for native in (False, True):
            with self.subTest(native=native):
                sim = self.sim(native=native, home_length=".200", guide_friction_mus=".2 .4")
                mus = np.array([.2, .4])
                sliding = False
                for control in np.r_[np.linspace(0, .003, 80), np.full(40, .003), np.linspace(.003, 0, 80)]:
                    sim.data.qpos.zero_()
                    sim.data.qvel.zero_()
                    sim.set_control([control])
                    sim.step()
                    history = sim.state.material.numpy()[0, 0]
                    self.assertAlmostEqual(history[:3].sum(), history[3], delta=1e-11)
                    cap = sim.state.capstan.numpy()[0, 0]
                    ratio = np.exp(mus * cap[:, 0])
                    self.assertTrue(np.all(cap[:, 1:3] >= 0))
                    self.assertTrue(np.all(cap[:, 2] <= ratio * cap[:, 1] + 2e-5))
                    self.assertTrue(np.all(cap[:, 1] <= ratio * cap[:, 2] + 2e-5))
                    sliding |= bool(np.any((cap[:, 0] > 1e-3) & (cap[:, 1] > 1) & (np.abs(np.abs(cap[:, 3]) - mus * cap[:, 0]) < 1e-5)))
                    np.testing.assert_allclose(sim.state.body_forces.numpy()[0, 0].sum(axis=0), 0, atol=2e-5)
                self.assertTrue(sliding)
                # Repeated force queries cannot commit material or bearing loads.
                history = sim.state.material.numpy().copy()
                loads = sim.hinges.loads.numpy().copy()
                sim.forward()
                before = sim.state.capstan.numpy().copy()
                sim.forward()
                np.testing.assert_array_equal(sim.state.material.numpy(), history)
                np.testing.assert_array_equal(sim.hinges.loads.numpy(), loads)
                np.testing.assert_array_equal(sim.state.capstan.numpy(), before)

    def test_small_wrap_angles_do_not_create_spurious_material_arc(self):
        sim = self.sim(hinges=False)
        m = sim.cpu_model
        host = self.mj.MjData(m)
        geom = m.geom("cylinder_a").id

        def contact_indices(angle):
            host.qpos[:] = [0, angle]
            self.mj.mj_forward(m, host)
            return np.flatnonzero(host.wrap_obj.reshape(-1)[:host.ten_wrapnum[0]] == geom)

        # Locate lift-off independently with the native double-precision route.
        lower, upper = -.5, .5
        self.assertEqual(len(contact_indices(lower)), 0)
        self.assertEqual(len(contact_indices(upper)), 2)
        for _ in range(40):
            midpoint = (lower + upper) / 2
            if len(contact_indices(midpoint)):
                upper = midpoint
            else:
                lower = midpoint
        for offset in (1e-5, 1e-4, 1e-3, -1e-3):
            q = np.array([[0, upper + offset]], dtype=np.float32)
            sim.data.qpos.assign(q)
            sim.forward()
            host.qpos[:] = q[0]
            self.mj.mj_forward(m, host)
            count = host.ten_wrapnum[0]
            objects = host.wrap_obj.reshape(-1)[:count]
            indices = np.flatnonzero(objects == geom)
            expected = 0.0
            if len(indices):
                a, b = host.wrap_xpos.reshape(-1, 3)[indices] - host.geom_xpos[geom]
                expected = np.arctan2(np.linalg.norm(np.cross(a, b)), a @ b)
            self.assertLess(expected, .001)
            self.assertAlmostEqual(sim.state.capstan.numpy()[0, 0, 0, 0], expected, delta=5e-5)
            sim.check()

    def test_batch_reset_preserves_other_worlds_and_lagged_loads(self):
        sim = self.sim(nworld=3, home_length=".200")
        sim.set_control([[.001], [.002], [.003]])
        sim.step(12)
        before = {name: getattr(sim.state, name).numpy().copy() for name in ("material", "filtered", "material_initialized")}
        loads, qpos = sim.hinges.loads.numpy().copy(), sim.data.qpos.numpy().copy()
        self.assertTrue(np.any(loads > 0))
        sim.reset([1])
        for name, values in before.items():
            np.testing.assert_array_equal(getattr(sim.state, name).numpy()[[0, 2]], values[[0, 2]])
        np.testing.assert_array_equal(sim.hinges.loads.numpy()[[0, 2]], loads[[0, 2]])
        np.testing.assert_array_equal(sim.data.qpos.numpy()[[0, 2]], qpos[[0, 2]])
        np.testing.assert_array_equal(sim.state.material.numpy()[1], 0)
        np.testing.assert_array_equal(sim.hinges.loads.numpy()[1], 0)

    def test_checkpoint_replays_material_and_bearing_history(self):
        sim = self.sim(home_length=".200", nworld=2)
        sim.set_control([[.002], [.003]])
        sim.step(20)
        checkpoint = sim.save_state()
        sim.step(15)
        expected = sim.save_state()
        sim.restore_state(checkpoint)
        sim.step(15)
        replay = sim.save_state()
        for name in expected:
            np.testing.assert_allclose(replay[name], expected[name], atol=2e-7, rtol=2e-5, err_msg=name)

    def test_invalid_material_and_sensor_configuration_is_rejected(self):
        from mujoco_cable_warp.model import UnsupportedCableError, load_model

        for options in (
            {"guide_friction_mus": ".2"}, {"guide_friction_mus": "nan .4"},
            {"capstan_sensor_guides": "guide_a guide_a"},
            {"body_force_sensor": "missing_sensor"}, {"damping": "1"},
        ):
            with self.subTest(options=options), self.assertRaises((ValueError, UnsupportedCableError)):
                load_model(self.model_file(**options))

    def test_hinge_zero_base_friction_and_per_world_bounds(self):
        path = self.model_file(home_length=".200")
        root = ET.parse(path).getroot()
        for joint in root.findall(".//worldbody//joint"):
            joint.set("frictionloss", "0")
        ET.ElementTree(root).write(path)
        sim = self.sim(path, nworld=2)
        sim.data.qvel.assign(np.array([[.1, .2], [.3, .4]], dtype=np.float32))
        sim.forward()
        np.testing.assert_array_equal(sim.hinges.loads.numpy(), 0)
        before = sim.data.efc.frictionloss.numpy().copy()
        sim.step(3)
        self.assertTrue(np.any(sim.hinges.loads.numpy() > 0))
        self.assertTrue(np.any(sim.data.efc.frictionloss.numpy() != before))
        np.testing.assert_array_equal(sim.cpu_model.dof_frictionloss, 0)
        sim.check()

    def test_body_force_sensor_is_physical_and_not_a_joint_torque(self):
        sim = self.sim(home_length=".200")
        m = sim.cpu_model
        host = self.mj.MjData(m)
        sim.forward()
        self.mj.mj_forward(m, host)
        forces = sim.state.body_forces.numpy()[0, 0]
        expected = np.zeros(m.nv)
        # All native contacts on a cylinder lie in the planar route;
        # accumulate the actual Cartesian forces at each contact independently.
        points = sim.state.points.numpy()[0, 0]
        bodies = sim.state.bodies.numpy()[0, 0]
        tensions = sim.state.tensions.numpy()[0, 0]
        directions = sim.state.directions.numpy()[0, 0]
        for node in range(7):
            force = np.zeros(3)
            if node:
                force -= tensions[node - 1] * directions[node - 1]
            if node < 6:
                force += tensions[node] * directions[node]
            self.mj.mj_applyFT(m, host, force, np.zeros(3), points[node].astype(float), int(bodies[node]), expected)
        np.testing.assert_allclose(sim.state.force.numpy()[0], expected, atol=2e-6)
        sensor = m.sensor("body_forces")
        np.testing.assert_array_equal(sim.data.sensordata.numpy()[0, sensor.adr[0]:sensor.adr[0] + sensor.dim[0]], forces.ravel())

    def test_cpp_reference_material_cycle_and_free_trajectory(self):
        if not self.reference:
            self.skipTest("set matching C++ reference library paths")
        path = self.model_file(home_length=".200", guide_friction_mus=".2 .4")
        sim = self.sim(path)
        m = self.mj.MjModel.from_xml_path(str(path))
        host = self.mj.MjData(m)
        self.mj.mj_forward(m, host)
        for control in np.r_[np.linspace(0, .003, 80), np.full(40, .003), np.linspace(.003, 0, 80)]:
            host.qpos[:] = 0
            host.qvel[:] = 0
            sim.data.qpos.zero_()
            sim.data.qvel.zero_()
            host.ctrl[:] = control
            sim.set_control([control])
            self.mj.mj_step(m, host)
            sim.step()
            actual = sim.state.capstan.numpy()[0, 0].ravel()
            np.testing.assert_allclose(actual, host.sensor("capstans").data, atol=3e-3, rtol=3e-3)
            np.testing.assert_allclose(sim.state.body_forces.numpy()[0, 0].ravel(), host.sensor("body_forces").data, atol=3e-3, rtol=3e-3)
        # A genuine coupled rollout, without overriding poses between steps.
        self.mj.mj_resetData(m, host)
        sim.reset()
        host.ctrl[:] = .003
        sim.set_control([.003])
        for _ in range(300):
            self.mj.mj_step(m, host)
            sim.step()
        sim.check()
        np.testing.assert_allclose(sim.data.qpos.numpy()[0], host.qpos, atol=2e-4)
        np.testing.assert_allclose(sim.data.qvel.numpy()[0], host.qvel, atol=3e-3)

    def test_cpp_reference_cable_without_external_hinge_plugin(self):
        if not self.cable_reference:
            self.skipTest("set the matching C++ cable reference library path")
        path = self.model_file(hinges=False, home_length=".200", guide_friction_mus=".2 .4")
        sim = self.sim(path)
        m = self.mj.MjModel.from_xml_path(str(path))
        host = self.mj.MjData(m)
        for q in ((0, 0), (25, 40), (70, 65), (90, 90)):
            self.mj.mj_resetData(m, host)
            sim.reset()
            for control in np.r_[np.linspace(0, .003, 30), np.linspace(.003, 0, 30)]:
                host.qpos[:] = np.deg2rad(q)
                host.qvel[:] = 0
                host.ctrl[:] = control
                sim.data.qpos.assign(host.qpos.astype(np.float32)[None])
                sim.data.qvel.zero_()
                sim.set_control([control])
                self.mj.mj_step(m, host)
                sim.step()
                np.testing.assert_allclose(sim.state.capstan.numpy()[0, 0].ravel(), host.sensor("capstans").data, atol=3e-3, rtol=3e-3)

    def test_cuda_graph_carries_material_and_hinge_history(self):
        if not self.wp.get_device(self.device).is_cuda:
            self.skipTest("requires a CUDA device")
        plain, graphed = self.sim(nworld=2), self.sim(nworld=2)
        for sim in (plain, graphed):
            sim.step()
            sim.reset()
        graphed.capture_step()
        for sim in (plain, graphed):
            sim.set_control([[.003], [.004]])
            sim.step(30)
            sim.get_route_points(1)
            sim.reset([0])
            sim.step(10)
            sim.check()
        np.testing.assert_allclose(plain.data.qpos.numpy(), graphed.data.qpos.numpy(), atol=2e-6)
        np.testing.assert_allclose(plain.state.material.numpy(), graphed.state.material.numpy(), atol=2e-8)
        np.testing.assert_allclose(plain.hinges.loads.numpy(), graphed.hinges.loads.numpy(), atol=2e-4)


if __name__ == "__main__":
    unittest.main()
