"""CPU capstan material history and native two-cylinder regressions.

Run with CABLE_PLUGIN_LIBRARY pointing to the matching MuJoCo build.
Optional MJHINGE_PLUGIN_LIBRARY enables the cross-plugin load regression.
"""
import os
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def fixture(native=False, mus='.2 .4', home=.040, hinge=False):
    route = ('''<site site="end"/><geom geom="dip_wrap" sidesite="dip_side"/>
      <site site="middle"/><geom geom="pip_wrap" sidesite="pip_side"/><site site="anchor"/>'''
      if native else '<site site="end"/><site site="dip_guide"/><site site="pip_guide"/><site site="anchor"/>')
    root=ET.fromstring(f'''<mujoco><compiler angle="radian"/><size nuser_site="1"/>
      <option timestep=".0002" gravity="0 0 0" integrator="implicitfast"/>
      <extension><plugin plugin="mujoco.cable.unilateral"><instance name="cable">
        <config key="route_mode" value="surface"/><config key="route_tendon" value="route"/>
        <config key="site_role_user_index" value="0"/>
        <config key="capstan_direction" value="stick_slip"/>
        <config key="ctrl_mode" value="target_contraction"/>
        <config key="stiffness" value="20000"/><config key="damping" value="0"/>
        <config key="home_length" value="{home}"/><config key="slack" value="0"/>
        <config key="control_timeconstant" value=".002"/>
        <config key="guide_friction_mus" value="{mus}"/>
        <config key="body_force_sensor" value="body_forces"/>
        <config key="capstan_sensor" value="capstans"/>
        <config key="capstan_sensor_guides" value="dip_guide pip_guide"/>
      </instance></plugin></extension>
      <worldbody><body name="pp">
        <site name="anchor" pos=".03463 .00783 0" user="1"/>
        <body name="mp" pos=".04166 -.00102 0">
          <joint name="pip" axis="0 0 1" damping="0" frictionloss=".00001"/>
          <geom type="sphere" size=".002" mass=".005" contype="0" conaffinity="0"/>
          <site name="pip_guide" pos=".01061 .0063 0" user="3"/>
          <site name="dip_guide" pos=".01822 .0062 0" user="3"/>
          <geom name="pip_wrap" type="cylinder" pos=".0106132848675 .00654997841836 0" size=".00025 .003" mass="0" contype="0" conaffinity="0"/>
          <geom name="dip_wrap" type="cylinder" pos=".0182232848675 .00644997841836 0" size=".00025 .003" mass="0" contype="0" conaffinity="0"/>
          <site name="pip_side" pos=".010606058159 .00600002589797 0"/>
          <site name="dip_side" pos=".018216058159 .00590002589797 0"/>
          <site name="middle" pos=".014415 .00625 0"/>
          <body name="dp" pos=".025 0 0">
            <joint name="dip" axis="0 0 1" damping="0" frictionloss=".00001"/>
            <geom type="sphere" size=".002" mass=".005" contype="0" conaffinity="0"/>
            <site name="end" pos=".00999 .00633 0" user="1"/>
          </body>
        </body>
      </body></worldbody>
      <tendon><spatial name="route">{route}</spatial></tendon>
      <actuator><plugin name="pull" tendon="route" instance="cable" gear="0" ctrllimited="true" ctrlrange="0 .025"/></actuator>
      <sensor><plugin name="cable_state" instance="cable"/>
        <plugin name="body_forces" instance="cable"/><plugin name="capstans" instance="cable"/></sensor>
    </mujoco>''')
    if hinge:
        group=ET.SubElement(root.find('extension'),'plugin',plugin='mjhinge.hinge_friction')
        for name in ('pip','dip'):
            inst=ET.SubElement(group,'instance',name=name+'_friction')
            for k,v in [('joint',name),('radial_gain','.001'),('body_force_sensors','body_forces')]:
                ET.SubElement(inst,'config',key=k,value=v)
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    return model,mujoco.MjData(model)


class StickSlipHistoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=os.environ.get('CABLE_PLUGIN_LIBRARY')
        if not path or not Path(path).is_file():
            raise unittest.SkipTest('Set CABLE_PLUGIN_LIBRARY to the CPU plugin')
        mujoco.mj_loadPluginLibrary(str(Path(path).resolve()))
        cls.hinge=os.environ.get('MJHINGE_PLUGIN_LIBRARY')
        if cls.hinge:mujoco.mj_loadPluginLibrary(str(Path(cls.hinge).resolve()))

    def fixed_step(self,m,d,control):
        d.qpos[:]=0;d.qvel[:]=0;d.ctrl[0]=control
        mujoco.mj_step(m,d)

    def test_material_conservation_friction_cones_and_memory(self):
        for native in (False,True):
            with self.subTest(native=native):
                m,d=fixture(native)
                adr=m.plugin_stateadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_PLUGIN,'cable')]
                sliding=False
                for u in np.r_[np.linspace(0,.003,150),np.full(100,.003),np.linspace(.003,0,150)]:
                    self.fixed_step(m,d,u)
                    state=d.plugin_state[adr:adr+10]
                    self.assertAlmostEqual(sum(state[5:8]),state[8],places=11)
                    for (angle,tin,tout,logratio),mu in zip(d.sensor('capstans').data.reshape(2,4),[.2,.4]):
                        self.assertGreaterEqual(min(tin,tout),0)
                        self.assertLessEqual(tout,np.exp(mu*angle)*tin+1e-6)
                        self.assertLessEqual(tin,np.exp(mu*angle)*tout+1e-6)
                        sliding|=tin>1 and abs(abs(logratio)-mu*angle)<1e-5
                    np.testing.assert_allclose(d.sensor('body_forces').data.reshape(-1,3).sum(axis=0),0,atol=1e-10)
                self.assertTrue(sliding)
                for _ in range(200):self.fixed_step(m,d,.003)
                before=d.sensor('capstans').data.copy()
                for _ in range(300):self.fixed_step(m,d,.003)
                np.testing.assert_allclose(before,d.sensor('capstans').data,atol=1e-8)
                self.assertGreater(np.max(abs(before[[3,7]])),.01)
                snapshot=d.plugin_state.copy()
                mujoco.mj_forward(m,d);mujoco.mj_forward(m,d)
                np.testing.assert_array_equal(d.plugin_state[adr+5:adr+10],snapshot[adr+5:adr+10])
                clone=mujoco.MjData(m);mujoco.mj_copyData(clone,m,d)
                self.fixed_step(m,d,.002);self.fixed_step(m,clone,.002)
                np.testing.assert_array_equal(d.plugin_state,clone.plugin_state)
                mujoco.mj_resetData(m,d);self.assertEqual(d.plugin_state[adr+9],0)

    def test_zero_friction_equalizes_tensions(self):
        for native in (False,True):
            m,d=fixture(native,'0 0')
            for _ in range(200):self.fixed_step(m,d,.003)
            c=d.sensor('capstans').data
            np.testing.assert_allclose(c[[1,2,5,6]],c[1],atol=1e-6)

    def test_native_geometry_and_frictionless_force_gradient(self):
        m,d=fixture(True,'0 0',home=.010)
        for a in np.linspace(0,90,7):
            for b in np.linspace(0,90,7):
                mujoco.mj_resetData(m,d);d.qpos[:]=np.deg2rad([a,b]);mujoco.mj_forward(m,d)
                self.assertEqual(d.sensor('cable_state').data[8],0)
                self.assertAlmostEqual(d.sensor('cable_state').data[0],d.ten_length[0],places=12)
        for q in [(0,0),(25,40),(70,65),(90,90)]:
            mujoco.mj_resetData(m,d);d.qpos[:]=np.deg2rad(q);mujoco.mj_forward(m,d)
            force=d.qfrc_passive.copy();tension=d.sensor('capstans').data[1]
            self.assertGreater(tension,0)
            jac=[]
            for axis in range(2):
                center=d.qpos[axis]
                d.qpos[axis]=center+1e-6;mujoco.mj_forward(m,d);plus=d.ten_length[0]
                d.qpos[axis]=center-1e-6;mujoco.mj_forward(m,d);minus=d.ten_length[0]
                d.qpos[axis]=center;jac.append((plus-minus)/2e-6)
            np.testing.assert_allclose(force,-tension*np.array(jac),atol=2e-7,rtol=2e-6)

    def test_optional_hinge_native_route_load_reporting(self):
        if not self.hinge:self.skipTest('Set MJHINGE_PLUGIN_LIBRARY for integration test')
        m,d=fixture(True,hinge=True)
        for _ in range(40):self.fixed_step(m,d,.003)
        for name in ('pip_friction','dip_friction'):
            adr=m.plugin_stateadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_PLUGIN,name)]
            self.assertGreater(d.plugin_state[adr],0)
        self.assertTrue(np.isfinite(d.qpos).all())
        self.assertEqual(int(d.warning.number.sum()),0)


if __name__=='__main__':unittest.main()
