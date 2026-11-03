"""Analytical stepping and lifecycle checks for cable implicit elasticity."""

import os
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from mujoco_cable_warp import CableSimulation
from mujoco_cable_warp.model import UnsupportedCableError

XML = '''<mujoco>
<option timestep="0.002" gravity="0 0 0" integrator="implicitfast"/>
<extension><plugin plugin="mujoco.cable.unilateral"><instance name="cable">
<config key="stiffness" value="10000"/>
<config key="damping" value="0"/>
<config key="pretension_offset" value="0.02"/>
<config key="integration_mode" value="implicit_elastic"/>
</instance></plugin></extension>
<worldbody><site name="anchor"/>
<body pos="1 0 0"><joint type="slide" axis="1 0 0"/>
<geom type="sphere" size=".01" mass="1" contype="0" conaffinity="0"/>
<site name="end"/></body></worldbody>
<tendon><spatial name="route"><site site="anchor"/>
<site site="end"/></spatial></tendon>
<actuator><plugin instance="cable" tendon="route"/></actuator>
</mujoco>'''


class ImplicitElasticTest(unittest.TestCase):
    def make(self, xml=XML, nworld=1):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / 'model.xml'
        path.write_text(xml)
        return CableSimulation(path, device=os.environ.get('CABLE_WARP_DEVICE', 'cpu'),
                               nworld=nworld,
                               nconmax=8, njmax=32)

    def test_matches_backward_euler_spring_with_nonzero_velocity(self):
        sim = self.make(nworld=2)
        sim.data.qvel.assign(np.array([[.1], [-.1]], dtype=np.float32))
        sim.step()
        h, k = .002, 10000.
        old_v = np.array([.1, -.1])
        expected_v = (old_v - h * k * .02) / (1 + h * h * k)
        np.testing.assert_allclose(sim.data.qvel.numpy()[:, 0], expected_v,
                                   atol=1e-6)
        np.testing.assert_allclose(sim.data.qpos.numpy()[:, 0], h * expected_v,
                                   atol=1e-7)

    def test_forward_is_physical_and_does_not_accumulate_metric(self):
        sim = self.make()
        for _ in range(3):
            sim.forward()
            np.testing.assert_allclose(sim.data.M.numpy(), [[1.]])
            force = (sim.data.qfrc_passive.numpy()
                     + sim.data.qfrc_actuator.numpy())
            np.testing.assert_allclose(force, [[-200.]], atol=.002)
        sim.step()
        sim.forward()
        np.testing.assert_allclose(sim.data.M.numpy(), [[1.]])

    def test_taut_transition_uses_extension_tangent(self):
        xml = XML.replace(
            '<config key="integration_mode"',
            '<config key="taut_transition" value="0.04"/>'
            '<config key="integration_mode"',
        )
        sim = self.make(xml)
        sim.step()
        # T = k*x*x/(2*width), dT/dx = k*x/width.
        tension = 10000. * .02**2 / (.04 * 2)
        tangent = 10000. * .02 / .04
        expected = -.002 * tension / (1 + .002**2 * tangent)
        np.testing.assert_allclose(sim.data.qvel.numpy(), [[expected]],
                                   atol=1e-6)

    def test_slack_cable_does_not_add_stiffness(self):
        sim = self.make(XML.replace('value="0.02"', 'value="-0.02"'))
        sim.data.qvel.assign(np.array([[.1]], dtype=np.float32))
        sim.step()
        np.testing.assert_allclose(sim.data.qvel.numpy(), [[.1]], atol=1e-7)
        np.testing.assert_allclose(sim.data.M.numpy(), [[1.]])

    def test_smaller_steps_converge_to_undamped_spring(self):
        exact = .02 * np.cos(1.) - .02
        errors = []
        for h in (.002, .001, .0005):
            sim = self.make(XML.replace('timestep="0.002"',
                                       f'timestep="{h}"'))
            sim.step(round(.01 / h))
            errors.append(abs(float(sim.data.qpos.numpy()[0, 0]) - exact))
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[2], errors[1])

    def test_cuda_capture_replays_step_and_reset(self):
        sim = self.make(nworld=2)
        if not sim.device.is_cuda:
            self.skipTest('CUDA device required')
        sim.step(3)
        expected = sim.data.qpos.numpy().copy()
        sim.reset()
        sim.capture_step()
        sim.step(3)
        np.testing.assert_allclose(sim.data.qpos.numpy(), expected, atol=1e-7)
        sim.reset()
        sim.step(3)
        np.testing.assert_allclose(sim.data.qpos.numpy(), expected, atol=1e-7)

    def test_force_clamp_does_not_add_elastic_tangent(self):
        xml = XML.replace('instance="cable" tendon="route"',
                          'instance="cable" tendon="route" '
                          'forcelimited="true" forcerange="-10 10"')
        sim = self.make(xml)
        sim.step()
        np.testing.assert_allclose(sim.data.qvel.numpy(), [[-.02]], atol=1e-7)
        np.testing.assert_allclose(sim.data.M.numpy(), [[1.]])

    def test_rejects_runge_kutta(self):
        with self.assertRaisesRegex(UnsupportedCableError, 'Euler.*implicitfast'):
            self.make(XML.replace('integrator="implicitfast"', 'integrator="RK4"'))

    def test_rejects_unimplemented_damping(self):
        with self.assertRaisesRegex(UnsupportedCableError, 'zero cable damping'):
            self.make(XML.replace('key="damping" value="0"',
                                  'key="damping" value="1"'))

    def test_material_forward_and_reset_keep_history_semantics(self):
        fixture = Path(__file__).parent / 'fixtures/two_cylinder_material.xml'
        root = ET.parse(fixture).getroot()
        instance = root.find('./extension/plugin/instance')
        ET.SubElement(instance, 'config', key='integration_mode',
                      value='implicit_elastic')
        sim = self.make(ET.tostring(root, encoding='unicode'), nworld=2)
        old = sim.state.material.numpy().copy()
        sim.forward()
        np.testing.assert_array_equal(old, sim.state.material.numpy())
        sim.step(3)
        self.assertTrue(np.isfinite(sim.data.qpos.numpy()).all())
        state = sim.state.material.numpy().copy()
        sim.forward()
        np.testing.assert_array_equal(state, sim.state.material.numpy())
        sim.reset()
        np.testing.assert_array_equal(old, sim.state.material.numpy())


if __name__ == '__main__':
    unittest.main()
