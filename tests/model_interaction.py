"""Pointer and rendered-stroke regression checks for the URDF workbench."""
import json

FIXTURE = '''<robot name="picking_fixture">
  <link name="base"><visual><origin xyz="0 0 1.3"/><geometry><box size="1 .3 .3"/></geometry></visual></link>
  <link name="left_leg"><visual><origin xyz="0 0 -.4"/><geometry><box size=".15 .15 .7"/></geometry></visual></link>
  <link name="badge"><visual><geometry><box size=".14 .14 .14"/></geometry></visual></link>
  <link name="foot"><visual><origin xyz=".1 0 -.04"/><geometry><box size=".35 .2 .12"/></geometry></visual></link>
  <link name="right_leg"><visual><origin xyz="0 0 -.4"/><geometry><box size=".15 .15 .7"/></geometry></visual></link>
  <joint name="left_joint" type="revolute"><parent link="base"/><child link="left_leg"/><origin xyz="-.55 0 1.15"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="10" velocity="2"/></joint>
  <joint name="badge_fixed" type="fixed"><parent link="left_leg"/><child link="badge"/><origin xyz="-.24 0 -.25"/></joint>
  <joint name="foot_joint" type="revolute"><parent link="left_leg"/><child link="foot"/><origin xyz="0 0 -.75"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="10" velocity="2"/></joint>
  <joint name="right_joint" type="prismatic"><parent link="base"/><child link="right_leg"/><origin xyz=".55 0 1.15"/><axis xyz="0 0 1"/><limit lower="0" upper=".3" effort="10" velocity="2"/></joint>
</robot>'''

# This independent oracle finds a projected point on a visible, named visual mesh.
POINT_ORACLE = '''window.pickTestPoint=name=>{
  const T=parcelRequire('../node_modules/three/build/three.module.js'),v=urdfWorkbench.viewer;
  v.scene.updateMatrixWorld(true);v.camera.updateMatrixWorld(true);
  const meshes=[];v.robot.traverseVisible(mesh=>{if(!mesh.isMesh)return;let parent=mesh.parent;while(parent&&!parent.isURDFVisual)parent=parent.parent;if(parent)meshes.push(mesh);});
  const owner=mesh=>{while(mesh&&!mesh.isURDFLink)mesh=mesh.parent;return mesh;};
  const link=v.robot.links[name],box=new T.Box3();
  for(const child of link.children)if(child.isURDFVisual)box.expandByObject(child);
  const center=box.getCenter(new T.Vector3()),size=box.getSize(new T.Vector3());
  for(const offset of [[0,0,0],[.2,0,0],[-.2,0,0],[0,.2,0],[0,-.2,0],[0,0,.2],[0,0,-.2]]){
    const p=center.clone().add(new T.Vector3(...offset).multiply(size)).project(v.camera),ray=new T.Raycaster();
    ray.setFromCamera(new T.Vector2(p.x,p.y),v.camera);
    const hit=ray.intersectObjects(meshes,false)[0];
    if(hit&&owner(hit.object)?.name===name)return {x:(p.x+1)*innerWidth/2,y:(1-p.y)*innerHeight/2,distance:hit.distance};
  }
  throw Error('No visible point for '+name);
}'''


def pointer(client, event_type, point, button='left', **extra):
    client.call('Input.dispatchMouseEvent', dict(type=event_type, x=point['x'], y=point['y'], button=button, **extra))


def rendered_frame(client):
    client.js('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))')


def check_model_interaction(client, output, check):
    client.js('document.getElementById("source").value='+json.dumps(FIXTURE)+';document.getElementById("source").dispatchEvent(new Event("input"));document.getElementById("reload").click()')
    client.until('urdfWorkbench.viewer.robot?.robotName==="picking_fixture" && !urdfWorkbench.loading')
    client.js(POINT_ORACLE)
    for key in ['axes', 'jointAxes']:
        client.js('if(!urdfWorkbench.display['+json.dumps(key)+'])document.querySelector('+json.dumps(f'[data-display={key}]')+').click()')
    rendered_frame(client)
    # Verify actual rendered world radii against perspective projection, not a lineWidth flag.
    stroke_check = '''(()=>{const T=parcelRequire('../node_modules/three/build/three.module.js'),v=urdfWorkbench.viewer,shafts=[];v.robot.traverseVisible(mesh=>{if(mesh.userData.axisShaft)shafts.push(mesh)});return shafts.length===18&&shafts.every(mesh=>{const p=mesh.getWorldPosition(new T.Vector3()).applyMatrix4(v.camera.matrixWorldInverse),s=mesh.parent.getWorldScale(new T.Vector3());const worldPerPixel=2*(-p.z)*Math.tan(T.MathUtils.degToRad(v.camera.fov/2))/(v.renderer.domElement.clientHeight*v.camera.zoom);return Math.abs(2*mesh.scale.x*s.x/worldPerPixel-3)<.1})})()'''
    check(client, stroke_check, 'coordinate and joint shafts render at three pixels')
    for name in ['base', 'left_leg', 'badge', 'foot', 'right_leg']:
        point=client.js('pickTestPoint('+json.dumps(name)+')')
        pointer(client, 'mouseMoved', point, button='none')
        client.until('urdfWorkbench.hoveredLink?.name==='+json.dumps(name))
        highlighted='''(()=>{const names=new Set();urdfWorkbench.viewer.robot.traverse(mesh=>{if(!mesh.isMesh)return;const materials=Array.isArray(mesh.material)?mesh.material:[mesh.material];if(materials.some(m=>m.emissive?.getHex()===0x268fff&&m.emissiveIntensity===.5)){while(mesh&&!mesh.isURDFLink)mesh=mesh.parent;if(mesh)names.add(mesh.name)}});return [...names]})()'''
        assert client.js(highlighted)==[name], (name, client.js(highlighted))
        pointer(client, 'mousePressed', point, clickCount=1)
        pointer(client, 'mouseReleased', point, clickCount=1)
        check(client, 'urdfWorkbench.selectedObject?.name==='+json.dumps(name), 'hover and click select the exact link: '+name)
        if name=='base':
            check(client, '(()=>{const T=parcelRequire("../node_modules/three/build/three.module.js");return !urdfWorkbench.viewer.scene.children.some(o=>o instanceof T.Box3Helper||o instanceof T.BoxHelper)})()', 'link selection does not create a bounding wireframe')
        if name=='badge':
            check(client, 'document.querySelector(`.structure-node[data-name="badge"]`).classList.contains("selected")', 'picking a fixed attachment reveals its exact topology node')
    # Simulate another link's auxiliary mesh in front of the root. It must not steal selection.
    base=client.js('pickTestPoint("base")')
    client.js('''(()=>{const T=parcelRequire('../node_modules/three/build/three.module.js'),v=urdfWorkbench.viewer,p=pickTestPoint('base'),ray=new T.Raycaster();ray.setFromCamera(new T.Vector2(p.x/innerWidth*2-1,1-p.y/innerHeight*2),v.camera);const mesh=new T.Mesh(new T.SphereGeometry(.05),new T.MeshBasicMaterial({color:0xffcc00}));v.robot.links.foot.add(mesh);mesh.position.copy(v.robot.links.foot.worldToLocal(ray.ray.at(p.distance-.12,new T.Vector3())));window.pickingTestHelper=mesh;v.redraw()})()''')
    pointer(client, 'mouseMoved', base, button='none')
    client.until('urdfWorkbench.hoveredLink?.name==="base"')
    check(client, 'urdfWorkbench.hoveredLink?.name==="base"', 'helper geometry cannot intercept model picking')
    client.js('pickingTestHelper.removeFromParent();pickingTestHelper.geometry.dispose();pickingTestHelper.material.dispose()')
    badge=client.js('pickTestPoint("badge")')
    client.js('urdfWorkbench.viewer.robot.links.badge.visible=false;urdfWorkbench.viewer.redraw()')
    pointer(client, 'mouseMoved', badge, button='none')
    check(client, 'urdfWorkbench.hoveredLink?.name!=="badge"', 'hidden links cannot be picked')
    client.js('urdfWorkbench.viewer.robot.links.badge.visible=true')
    client.js('urdfWorkbench.viewer.world.visible=false;urdfWorkbench.viewer.redraw()')
    pointer(client, 'mouseMoved', base, button='none')
    check(client, 'urdfWorkbench.hoveredLink===null', 'hidden ancestors suppress all descendant hits')
    client.js('urdfWorkbench.viewer.world.visible=true;urdfWorkbench.viewer.redraw()')
    # Joint drag uses the owning movable joint while selection continues to use the exact link.
    leg=client.js('pickTestPoint("left_leg")')
    initial=client.js('urdfWorkbench.viewer.robot.joints.left_joint.angle')
    pointer(client, 'mousePressed', leg, clickCount=1)
    check(client, 'urdfWorkbench.viewer.dragControls.manipulating?.name==="left_joint"&&!urdfWorkbench.viewer.controls.enabled', 'left drag grabs the correct joint')
    moved=dict(x=leg['x']+35,y=leg['y']+20)
    pointer(client, 'mouseMoved', moved, buttons=1)
    pointer(client, 'mouseReleased', moved)
    check(client, f'Math.abs(urdfWorkbench.viewer.robot.joints.left_joint.angle-({initial}))>.001 && urdfWorkbench.viewer.dragControls.manipulating===null && urdfWorkbench.viewer.controls.enabled', 'joint dragging moves and releases normally')
    leg=client.js('pickTestPoint("left_leg")')
    client.js('window.testAngle=urdfWorkbench.viewer.robot.joints.left_joint.angle;window.testCamera=urdfWorkbench.viewer.camera.position.clone()')
    pointer(client, 'mousePressed', leg, button='right', clickCount=1)
    pointer(client, 'mouseMoved', dict(x=leg['x']+30,y=leg['y']+12), button='right', buttons=2)
    pointer(client, 'mouseReleased', dict(x=leg['x']+30,y=leg['y']+12), button='right')
    check(client, 'urdfWorkbench.viewer.robot.joints.left_joint.angle===testAngle && urdfWorkbench.viewer.camera.position.distanceTo(testCamera)>.001', 'right drag pans the camera without moving the joint')
    leg=client.js('pickTestPoint("left_leg")')
    pointer(client, 'mousePressed', leg, clickCount=1)
    client.js('window.dispatchEvent(new Event("blur"))')
    check(client, 'urdfWorkbench.viewer.dragControls.manipulating===null&&urdfWorkbench.viewer.controls.enabled', 'focus loss releases the joint drag')
    pointer(client, 'mouseReleased', leg)
    client.js('(()=>{const v=urdfWorkbench.viewer;v.camera.position.sub(v.controls.target).multiplyScalar(.7).add(v.controls.target);v.controls.update();v.redraw()})()')
    rendered_frame(client)
    check(client, stroke_check, 'thick shafts keep their width after camera zoom')
    client.js('document.getElementById("fit-view").click()')
    rendered_frame(client)
    badge=client.js('pickTestPoint("badge")')
    pointer(client, 'mouseMoved', badge, button='none')
    pointer(client, 'mousePressed', badge, clickCount=1)
    pointer(client, 'mouseReleased', badge, clickCount=1)
    client.screenshot(str(output/'axes-and-link-picking.png'))
