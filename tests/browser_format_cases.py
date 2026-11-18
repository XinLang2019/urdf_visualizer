"""Boundary cases and independent native checks for the static browser engine."""
import json
from pathlib import Path
import tempfile
import io
import zipfile
import base64
import numpy as np
import model_formats as formats
from test_model_formats import physical


def browser_import(client, files, entry):
    expression='''(async()=>{const files=new Map(INPUT.map(([name,data])=>[name,new File([data],name.split('/').pop())]));try{return await(await URDFBrowserFormats.process('import',ENTRY,files)).json()}catch(e){return {error:e.message}}})()'''
    return client.js(expression.replace('INPUT',json.dumps(list(files.items()))).replace('ENTRY',json.dumps(entry)))


def compare_native(client,files,entry,joint=None):
    result=browser_import(client,files,entry)
    assert 'error' not in result,result
    with tempfile.TemporaryDirectory(prefix='static-format-cases-') as folder:
        root=Path(folder);native=root/'native';browser=root/'browser';native.mkdir();browser.mkdir()
        for name,data in files.items():
            path=native/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(data)
        (browser/'model.urdf').write_text(result['urdf'])
        import base64
        for asset in result['assets']:
            path=browser/asset['path'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(base64.b64decode(asset['data']))
        a=formats.read_model(native,native/entry)[0];b=formats.read_model(browser,browser/'model.urdf')[0]
        for angle in [-.4,0,.6] if joint else [0]:
            poses={joint:angle} if joint else {}
            pa,pb=physical(a,poses),physical(b,poses)
            assert set(pa)==set(pb),(set(pa),set(pb))
            for name in pa:
                for i in range(4):np.testing.assert_allclose(pa[name][i],pb[name][i],atol=5e-6)
    return result


def check_browser_cases(client):
    # XML entities and USD string punctuation must survive source formatting.
    name='Robot "[{}]" & <test>'
    from xml.sax.saxutils import quoteattr
    source='<robot name='+quoteattr(name)+'><link name="base"><visual><geometry><box size=".2 .3 .4"/></geometry></visual></link></robot>'
    for target in ['urdf','mjcf','usd']:
        expression='''(async()=>{const file=new File([INPUT],'robot.urdf');const r=await URDFBrowserFormats.process('convert','robot.urdf',new Map([['robot.urdf',file]]),TARGET);const bytes=new Uint8Array(await r.arrayBuffer());return btoa(String.fromCharCode(...bytes))})()'''
        packed=base64.b64decode(client.js(expression.replace('INPUT',json.dumps(source)).replace('TARGET',json.dumps(target))))
        with zipfile.ZipFile(io.BytesIO(packed)) as archive:
            entry=json.loads(archive.read('conversion-report.json'))['entry']
            result=browser_import(client,{entry:archive.read(entry).decode()},entry)
        assert 'error' not in result,result
        import xml.etree.ElementTree as ET
        assert ET.fromstring(result['urdf']).get('name')==name
    print('PASS formatted URDF, MJCF and USDA preserve XML entities, quoted strings, brackets and braces',flush=True)
    mjcf='''<mujoco model="frames"><compiler angle="degree"/><default><geom density="500" rgba=".2 .4 .6 1"/><default class="leg"><joint damping=".2"/><geom size=".1 .2 .3"/></default></default><include file="parts/bodies.xml"/></mujoco>'''
    bodies='''<mujocoinclude><worldbody><body name="base" pos=".1 .2 .7" euler="10 20 30"><geom type="box" size=".1 .15 .2"/><body name="arm" pos=".2 .1 .3" childclass="leg" euler="5 15 20"><joint name="hinge" type="hinge" axis=".2 1 .3" pos=".04 .02 -.1" range="-60 70" ref="10"/><geom type="box" pos=".03 .1 -.2" euler="20 30 40"/><body name="tip" pos=".1 .2 -.3"><geom type="sphere" size=".04" mass=".3"/></body></body></body></worldbody></mujocoinclude>'''
    compare_native(client,{'robot.xml':mjcf,'parts/bodies.xml':bodies},'robot.xml','hinge')
    print('PASS browser MJCF includes, inherited defaults, degree rotations, anchor rebasing, ref and density match MuJoCo',flush=True)
    for kind,size in [('capsule','.08 .2'),('ellipsoid','.1 .2 .3'),('cylinder','.1 .2')]:
        text=f'<mujoco><worldbody><body name="body"><geom type="{kind}" size="{size}" pos=".1 .2 .3" quat=".8 .1 .2 .3" density="600"/></body></worldbody></mujoco>'
        result=compare_native(client,{'shape.xml':text},'shape.xml')
        assert 'inertial' in result['urdf']
    print('PASS browser capsule, ellipsoid and cylinder mass / COM / inertia match native analytic values',flush=True)
    usd='''#usda 1.0
(metersPerUnit = 0.01\nkilogramsPerUnit = 2\nupAxis = "Y")
def Xform "Robot" {
def Xform "base" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {
float physics:mass = 2
point3f physics:centerOfMass = (1, 2, 3)
float3 physics:diagonalInertia = (100, 200, 250)
quatf physics:principalAxes = (0.9238795325, 0, 0, 0.3826834324)
double3 xformOp:translate = (20, 70, 10)
float3 xformOp:rotateXYZ = (10, 20, 30)
uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
def Cube "box" {
double size = 2
double3 xformOp:translate = (5, 10, 15)
float3 xformOp:scale = (10, 15, 20)
uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
}
}
}'''
    compare_native(client,{'centimeters.usda':usd},'centimeters.usda')
    print('PASS browser USDA centimetres, mass units, Y-up, ordered transforms and principal inertia match OpenUSD',flush=True)
    floating='''<robot name="floating"><link name="__world"/><link name="base"><inertial><mass value="1"/><inertia ixx=".1" iyy=".1" izz=".1"/></inertial><visual><geometry><box size=".2 .2 .2"/></geometry></visual></link><joint name="floating_base" type="floating"><parent link="__world"/><child link="base"/><origin xyz=".1 .2 .7"/></joint></robot>'''
    expression='''(async()=>{const file=new File([INPUT],'robot.urdf');const r=await URDFBrowserFormats.process('convert','robot.urdf',new Map([['robot.urdf',file]]),'usd');const bytes=new Uint8Array(await r.arrayBuffer());return btoa(String.fromCharCode(...bytes))})()'''
    packed=base64.b64decode(client.js(expression.replace('INPUT',json.dumps(floating))))
    with zipfile.ZipFile(io.BytesIO(packed)) as archive:
        result=browser_import(client,{'robot.usda':archive.read('model.usda').decode()},'robot.usda')
    assert 'error' not in result,result
    import xml.etree.ElementTree as ET
    robot=ET.fromstring(result['urdf']);assert len(robot.findall('link'))==2
    joints=robot.findall('joint');assert len(joints)==1 and joints[0].get('name')=='floating_base' and joints[0].get('type')=='floating'
    print('PASS browser USD floating roots preserve the real __world link and floating joint name without extra virtual links',flush=True)
    bad_cases=[
        ({'binary.usd':'PXR-USDC\u0000\u0000'},'binary.usd','USDA'),
        ({'reference.usda':'#usda 1.0\n(subLayers = [@other.usda@])'},'reference.usda','flatten'),
        ({'cycle.xml':'<mujoco><include file="cycle.xml"/></mujoco>'},'cycle.xml','Cyclic'),
        ({'ball.xml':'<mujoco><worldbody><body><joint type="ball"/></body></worldbody></mujoco>'},'ball.xml','Ball'),
        ({'multi.xml':'<mujoco><worldbody><body><joint/><joint/></body></worldbody></mujoco>'},'multi.xml','one joint'),
    ]
    for files,entry,expected in bad_cases:
        result=browser_import(client,files,entry);assert expected.lower() in result.get('error','').lower(),result
    print('PASS browser rejects binary USD, unflattened layers, cyclic includes, ball and multi-joint bodies explicitly',flush=True)
