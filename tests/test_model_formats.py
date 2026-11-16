"""Native interchange checks: six directions, physical frames, units and assets."""
import math
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial.transform import Rotation
import model_formats as f
from model_server import portable_path,unpack_usdz

URDF='''<robot name="format_fixture">
<link name="base"><inertial><mass value="2"/><inertia ixx=".12" iyy=".14" izz=".16"/></inertial><visual><origin xyz="0 0 .7"/><geometry><box size=".3 .2 .1"/></geometry></visual></link>
<link name="arm"><inertial><origin xyz="0 0 -.2" rpy=".1 .2 .3"/><mass value="1"/><inertia ixx=".03" iyy=".04" izz=".05"/></inertial><visual><origin xyz="0 0 -.2"/><geometry><cylinder radius=".04" length=".4"/></geometry><material name="red"><color rgba=".8 .1 .2 1"/></material></visual><collision><origin xyz="0 0 -.2"/><geometry><box size=".08 .08 .4"/></geometry></collision></link>
<joint name="hinge" type="revolute"><parent link="base"/><child link="arm"/><origin xyz=".2 .1 .7" rpy=".2 .3 .4"/><axis xyz="0 1 0"/><limit lower="-1" upper="1.2" effort="10" velocity="3"/><dynamics damping=".2" friction=".1"/></joint></robot>'''


def poses(model,q):
    root=model.validate();result={root:np.eye(4)}
    while len(result)<len(model.links):
        for j in model.joints:
            if j.parent in result and j.child not in result:
                delta=np.eye(4)
                if j.kind in ['revolute','continuous']:delta[:3,:3]=Rotation.from_rotvec(j.axis*q.get(j.name,0)).as_matrix()
                elif j.kind=='prismatic':delta[:3,3]=j.axis*q.get(j.name,0)
                result[j.child]=result[j.parent]@j.pose@delta
    return result


def physical(model,q):
    world=poses(model,q);out={}
    for name,link in model.links.items():
        if not link.mass:continue
        t=world[name]
        centers=[]
        for shape in link.visuals:
            mesh=f.shape_mesh(shape);mesh.apply_transform(t@shape.pose)
            centers.append(mesh.bounds.mean(axis=0))
        out[name]=(link.mass,(t@np.r_[link.com,1])[:3],t[:3,:3]@link.inertia@t[:3,:3].T,centers)
    return out


class Formats(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='format-unit-');self.root=Path(self.temp.name)
        self.source=self.root/'robot.urdf';self.source.write_text(URDF)
        self.model=f.read_model(self.root,self.source)[0]
    def tearDown(self):self.temp.cleanup()
    def equivalent(self,a,b):
        ja=next(j for j in a.joints if j.name=='hinge');jb=next(j for j in b.joints if j.name=='hinge')
        self.assertEqual(ja.kind,jb.kind)
        np.testing.assert_allclose([ja.lower,ja.upper,ja.effort,ja.velocity],[jb.lower,jb.upper,jb.effort,jb.velocity],atol=2e-6)
        for angle in [-.7,0,.8]:
            pa,pb=physical(a,{'hinge':angle}),physical(b,{'hinge':angle})
            self.assertEqual(set(pa),set(pb))
            for name in pa:
                for i in range(3):np.testing.assert_allclose(pa[name][i],pb[name][i],atol=3e-6)
                np.testing.assert_allclose(pa[name][3],pb[name][3],atol=3e-6)
    def test_six_directions_and_binary_usd(self):
        models={'urdf':self.model}
        for target in ['mjcf','usd']:
            path=f.write_model(self.model,target,self.root/('source_'+target))
            if target=='usd':self.assertEqual(path.read_bytes()[:8],b'PXR-USDC')
            models[target]=f.read_model(self.root,path)[0]
            self.equivalent(self.model,models[target])
        for source,model in models.items():
            for target in models:
                if source==target:continue
                with self.subTest(source=source,target=target):
                    path=f.write_model(model,target,self.root/(source+'_to_'+target))
                    result=f.read_model(self.root,path)[0]
                    self.equivalent(self.model,result)
    def test_mjcf_defaults_includes_degree_limits_and_joint_anchor(self):
        folder=self.root/'mjc';folder.mkdir()
        (folder/'defaults.xml').write_text('<mujocoinclude><default><joint damping=".3"/><geom density="300" rgba=".1 .4 .8 1"/></default></mujocoinclude>')
        path=folder/'main.xml';path.write_text('''<mujoco model="anchor"><include file="defaults.xml"/><compiler angle="degree"/><worldbody><body name="root" pos=".2 .1 1"><geom type="box" size=".1 .1 .1"/><body name="tip" pos=".4 0 0"><joint name="hinge" axis="0 0 1" pos=".1 .02 0" ref="10" range="-60 80"/><geom type="capsule" fromto="0 0 0 0 0 -.4" size=".04"/></body></body></worldbody></mujoco>''')
        model=f.read_model(self.root,path)[0];j=next(j for j in model.joints if j.name=='hinge')
        np.testing.assert_allclose([j.lower,j.upper],np.deg2rad([-70,70]),atol=1e-8)
        self.assertAlmostEqual(j.damping,.3)
        import mujoco
        native=mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(native)
        for angle in [0,.5]:
            data.qpos[:]=native.qpos0;data.qpos[0]+=angle;mujoco.mj_forward(native,data)
            bid=mujoco.mj_name2id(native,mujoco.mjtObj.mjOBJ_BODY,'tip')
            t=poses(model,{'hinge':angle})['tip'];link=model.links['tip']
            np.testing.assert_allclose((t@np.r_[link.com,1])[:3],data.xipos[bid],atol=1e-8)
    def test_usd_units_y_up_static_geometry_and_layer_reference(self):
        from pxr import Usd,UsdGeom,Gf
        sub=Usd.Stage.CreateNew(str(self.root/'geometry.usda'));g=UsdGeom.Cube.Define(sub,'/Box');g.CreateSizeAttr(20);g.AddTranslateOp().Set(Gf.Vec3d(0,100,0));sub.SetDefaultPrim(g.GetPrim());sub.GetRootLayer().Save()
        stage=Usd.Stage.CreateNew(str(self.root/'scene.usda'));UsdGeom.SetStageMetersPerUnit(stage,.01);UsdGeom.SetStageUpAxis(stage,'Y');p=stage.DefinePrim('/World');p.GetReferences().AddReference('geometry.usda');stage.SetDefaultPrim(p);stage.GetRootLayer().Save()
        model=f.read_model(self.root,self.root/'scene.usda')[0];shape=next(iter(model.links.values())).visuals[0]
        np.testing.assert_allclose(shape.mesh.bounds.mean(axis=0),[0,0,1],atol=1e-7)
        np.testing.assert_allclose(np.ptp(shape.mesh.bounds,axis=0),[.2,.2,.2],atol=1e-7)
        self.assertTrue(any('no rigid-body' in note for note in model.notes))
    def test_obj_package_resource_and_negative_scale(self):
        folder=self.root/'package/meshes';folder.mkdir(parents=True)
        f.trimesh.creation.box(extents=[.1,.2,.3]).export(folder/'box.obj')
        text=URDF.replace('<box size=".3 .2 .1"/>','<mesh filename="package://package/meshes/box.obj" scale="2 1 -1"/>')
        self.source.write_text(text);model=f.read_model(self.root,self.source)[0]
        np.testing.assert_allclose(model.links['base'].visuals[0].mesh.extents,[.2,.2,.3])
        path=f.write_model(model,'usd',self.root/'mesh_usd');roundtrip=f.read_model(self.root,path)[0]
        self.equivalent(model,roundtrip)
    def test_missing_resources_closed_loops_and_path_escape_rejected(self):
        self.source.write_text(URDF.replace('<box size=".3 .2 .1"/>','<mesh filename="missing.obj"/>'))
        with self.assertRaisesRegex(ValueError,'Missing'):f.read_model(self.root,self.source)
        for path in ['../escape.usd','/tmp/escape.usd','a/../../secret','a\\b','http://remote/model.usd']:
            with self.assertRaises(ValueError):portable_path(path)
        self.model.joints.append(f.Joint('cycle','fixed','arm','base'))
        with self.assertRaises(ValueError):self.model.validate()
        (self.root/'bad.usda').write_text('#usda 1.0\n( subLayers = [@/etc/passwd@] )\n')
        with self.assertRaises(ValueError):f.read_model(self.root,self.root/'bad.usda')
    def test_zero_axis_fixed_link_does_not_gain_inferred_mass(self):
        text=URDF.replace('</robot>','<link name="camera"><visual><geometry><box size=".1 .1 .1"/></geometry></visual></link><joint name="camera_fixed" type="fixed"><parent link="base"/><child link="camera"/><axis xyz="0 0 0"/></joint></robot>')
        self.source.write_text(text);model=f.read_model(self.root,self.source)[0]
        native=f.write_model(model,'mjcf',self.root/'camera')
        result=f.read_model(self.root,native)[0]
        self.assertEqual(result.links['camera'].mass,0.)
        self.assertAlmostEqual(sum(link.mass for link in result.links.values()),3.)
        self.assertIn('camera_fixed',[j.name for j in result.joints])
    def test_floating_world_root_roundtrip(self):
        text=URDF.replace('</robot>','<link name="__world"/><joint name="free_base" type="floating"><parent link="__world"/><child link="base"/><origin xyz=".2 .1 .8"/></joint></robot>')
        self.source.write_text(text);model=f.read_model(self.root,self.source)[0]
        for target in ['mjcf','usd']:
            native=f.write_model(model,target,self.root/('floating_'+target));result=f.read_model(self.root,native)[0]
            self.assertEqual(set(model.links),set(result.links))
            self.assertEqual(next(j for j in result.joints if j.kind=='floating').name,'free_base')
            self.equivalent(model,result)
    def test_usdz_and_archive_escape(self):
        import zipfile
        stage=f.write_model(self.model,'usd',self.root/'binary')
        package=self.root/'robot.usdz'
        with zipfile.ZipFile(package,'w') as z:z.writestr('model.usd',stage.read_bytes())
        unpacked=unpack_usdz(package,self.root)
        result=f.read_model(self.root,unpacked)[0];self.equivalent(self.model,result)
        directory=self.root/'bad_archive';directory.mkdir();bad=directory/'bad.usdz'
        with zipfile.ZipFile(bad,'w') as z:z.writestr('../outside.usda','#usda 1.0')
        with self.assertRaises(ValueError):unpack_usdz(bad,directory)
        self.assertFalse((directory/'outside.usda').exists())
    def test_multi_joint_expansion_and_unsupported_ball(self):
        p=self.root/'multi.xml';p.write_text('<mujoco><worldbody><body name="body"><joint name="x" axis="1 0 0"/><joint name="y" axis="0 1 0" pos=".1 0 0"/><geom type="box" size=".1 .1 .1"/></body></worldbody></mujoco>')
        model=f.read_model(self.root,p)[0];self.assertEqual(len([j for j in model.joints if j.kind=='continuous']),2)
        p.write_text('<mujoco><worldbody><body><joint type="ball"/><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
        with self.assertRaisesRegex(ValueError,'Ball'):f.read_model(self.root,p)

if __name__=='__main__':unittest.main()
