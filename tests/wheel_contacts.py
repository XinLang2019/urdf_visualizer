"""Wheel contacts lie in tread midplanes, independent of mirror orientation and spin."""
import json
from model_interaction import rendered_frame

WHEELS=[(.5,.4,1),(.5,-.4,-1),(-.5,.4,-1),(-.5,-.4,1)]
FIXTURE='<robot name="wheel_contact_fixture"><link name="base"><inertial><origin xyz="0 0 .8"/><mass value="10"/><inertia ixx="1" iyy="1" izz="1"/></inertial><visual><origin xyz="0 0 .8"/><geometry><box size=".8 .5 .25"/></geometry></visual></link>'
for i,(x,y,sign) in enumerate(WHEELS):
    FIXTURE+=f'''<link name="wheel_{i}"><inertial><origin xyz="0 {sign*.03} 0"/><mass value="1"/><inertia ixx=".01" iyy=".01" izz=".01"/></inertial><visual><origin xyz="0 {sign*.03} 0" rpy="{sign*1.5707963267948966} 0 0"/><geometry><cylinder radius=".2" length=".12"/></geometry></visual><collision><origin xyz="0 {sign*.03} 0" rpy="{sign*1.5707963267948966} 0 0"/><geometry><cylinder radius=".2" length=".12"/></geometry></collision></link>
<joint name="wheel_{i}_joint" type="continuous"><origin xyz="{x} {y} .2"/><parent link="base"/><child link="wheel_{i}"/><axis xyz="0 {sign} 0"/><limit effort="20" velocity="10"/></joint>'''
FIXTURE+='</robot>'


def check_wheel_contacts(client,output,check):
    # Render-independent geometry checks cover imported STL-like triangle buffers.
    checks=client.js('''(()=>{const T=parcelRequire('../node_modules/three/build/three.module.js'),G=new URDFStaticContacts(T),out={},near=(a,b)=>Math.abs(a-b)<1e-6;
    for(const sign of [-1,1]){
      const axle=new T.Group();axle.isURDFJoint=true;axle.jointType='continuous';axle.axis=new T.Vector3(0,sign,0);axle.rotation.x=Math.PI/2;
      const link=new T.Group();axle.add(link);
      const geometry=new T.CylinderGeometry(.2,.2,.12,31).toNonIndexed();geometry.translate(0,.17,0);
      const mesh=new T.Mesh(geometry),shape={link:'wheel',object:link,meshes:[mesh],fallback:false};link.add(mesh);axle.updateMatrixWorld(true);G.prepare(shape);
      const initial=G.bottom(shape),inverse=link.matrixWorld.clone().invert(),local=new T.Vector3(...initial.points[0]).applyMatrix4(inverse);
      out['meshMidplane'+sign]=!!shape.wheel&&initial.points.length===1&&near(local.y,.17)&&near(initial.min,-.2);
      axle.rotation.y=.8;link.rotation.y=.57;axle.updateMatrixWorld(true);const moved=G.bottom(shape),localMoved=new T.Vector3(...moved.points[0]).applyMatrix4(link.matrixWorld.clone().invert());
      out['meshSpin'+sign]=moved.points.length===1&&near(localMoved.y,.17)&&near(moved.min,-.2);
    }
    const sphere=new T.Mesh(new T.SphereGeometry(.1));sphere.position.set(.3,.2,-.4);sphere.scale.set(1,2,1);sphere.updateMatrixWorld(true);
    const foot={link:'point_foot',object:sphere,meshes:[sphere],fallback:false};G.prepare(foot);const point=G.bottom(foot);out.sphere=!foot.wheel&&point.points.length===1&&point.points[0].every((x,i)=>near(x,[.3,0,-.4][i]));
    const box=new T.Mesh(new T.BoxGeometry(.4,.1,.2));box.geometry.computeBoundingBox();box.updateMatrixWorld(true);const patch=G.bottom({link:'flat_foot',object:box,meshes:[box],fallback:false});out.box=patch.points.length===4&&near(patch.min,-.05);
    const cap=new T.Mesh(new T.CylinderGeometry(.2,.2,.1,32));cap.updateMatrixWorld(true);const capPatch=G.bottom({link:'cap',object:cap,meshes:[cap],fallback:false});out.cap=capPatch.points.length===4&&near(capPatch.min,-.05);
    // Imported rectangular feet must not be mistaken for circular wheel treads.
    const ankle=new T.Group();ankle.isURDFJoint=true;ankle.jointType='revolute';ankle.axis=new T.Vector3(0,1,0);const flat=new T.Group();ankle.add(flat);const flatMesh=new T.Mesh(new T.BoxGeometry(.4,.1,.2,8,8,8).toNonIndexed());flat.add(flatMesh);ankle.updateMatrixWorld(true);const flatShape={link:'flat',object:flat,meshes:[flatMesh],fallback:false};G.prepare(flatShape);out.rectangularMesh=!flatShape.wheel;
    const ballJoint=new T.Group();ballJoint.isURDFJoint=true;ballJoint.jointType='revolute';ballJoint.axis=new T.Vector3(0,1,0);const ballLink=new T.Group();ballJoint.add(ballLink);const ball=new T.Mesh(new T.SphereGeometry(.1,32,24).toNonIndexed());ballLink.add(ball);ballJoint.updateMatrixWorld(true);const ballShape={link:'mesh_ball',object:ballLink,meshes:[ball],fallback:false};G.prepare(ballShape);out.meshSphere=!ballShape.wheel;
    return out;})()''')
    for name,passed in checks.items():assert passed,('wheel support geometry',name)
    print('PASS imported mesh midplanes, mirrored axle directions, wheel spin, sphere feet, cap patches and rectangular-foot rejection',flush=True)
    client.js('document.getElementById("source").value='+json.dumps(FIXTURE)+';document.getElementById("source").dispatchEvent(new Event("input"));document.getElementById("reload").click()')
    client.until('urdfWorkbench.viewer.robot?.robotName==="wheel_contact_fixture"&&!urdfWorkbench.loading')
    client.js('if(!urdfWorkbench.statics.enabled)document.getElementById("statics-toggle").click()');rendered_frame(client)
    check(client,'(()=>{const s=urdfWorkbench.statics;return s.valid&&s.contacts.length===4&&s.contacts.every(c=>c.kind==="wheelMidplane"&&Math.abs(c.position[1])<1e-7)&&s.forceResidual<1e-6&&s.momentResidual<1e-6})()','four mirrored cylinder wheels each supply one midplane contact with balanced forces')
    check(client,'(()=>{const T=parcelRequire("../node_modules/three/build/three.module.js"),r=urdfWorkbench.viewer.robot;return urdfWorkbench.statics.contacts.every(c=>{const collider=r.links[c.link].children.find(o=>o.isURDFCollider),mesh=collider.children.find(o=>o.isMesh),center=mesh.getWorldPosition(new T.Vector3());return Math.abs(c.position[0]-center.x)<1e-7&&Math.abs(c.position[2]-center.z)<1e-7})})()','contact positions match each wheel geometry center across thickness, including local origin offsets')
    client.js('window.wheelContactsBefore=urdfWorkbench.statics.contacts.map(c=>c.position);urdfWorkbench.viewer.setJointValue("wheel_0_joint",.7);urdfWorkbench.viewer.setJointValue("wheel_1_joint",-1.2)');rendered_frame(client)
    check(client,'urdfWorkbench.statics.valid&&urdfWorkbench.statics.contacts.every((c,i)=>c.position.every((x,j)=>Math.abs(x-wheelContactsBefore[i][j])<1e-7))','wheel rotation does not move the contact between inner and outer rim vertices')
    check(client,'document.querySelectorAll("#statics-overlay .statics-contact").length===4','3D support markers use the same four midplane points as the force solver')
    client.screenshot(str(output/'wheel-midplane-contacts.png'))
    client.js('document.getElementById("statics-toggle").click();document.getElementById("right-code-tab").click()')
