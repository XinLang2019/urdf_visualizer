"""Motion ground invariance and real-pointer editing of a fixed chassis link."""
import json
from model_interaction import POINT_ORACLE, pointer, rendered_frame

FIXTURE='''<robot name="motion_body_fixture">
<link name="world"/>
<link name="chassis"><visual><origin xyz="0 0 .9"/><geometry><box size=".8 .5 .3"/></geometry></visual></link>
<joint name="body_fixed" type="fixed"><parent link="world"/><child link="chassis"/></joint>
<link name="arm"><visual><origin xyz="0 0 -.3"/><geometry><box size=".12 .12 .6"/></geometry></visual></link>
<joint name="arm_joint" type="revolute"><parent link="chassis"/><child link="arm"/><origin xyz=".5 0 .9"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="10" velocity="2"/></joint>
</robot>'''

GROUND='''(()=>{const v=urdfWorkbench.viewer,g=v.scene.children.find(o=>o.type==='GridHelper');return [v.plane,g].map(o=>({position:o.position.toArray(),quaternion:o.quaternion.toArray(),scale:o.scale.toArray()}))})()'''


def click_link(client,name,modifiers=0):
    point=client.js('pickTestPoint('+json.dumps(name)+')')
    pointer(client,'mouseMoved',point,button='none',modifiers=modifiers)
    pointer(client,'mousePressed',point,clickCount=1,modifiers=modifiers)
    pointer(client,'mouseReleased',point,clickCount=1,modifiers=modifiers)
    rendered_frame(client)


def change(client,id,value):
    client.js('document.getElementById('+json.dumps(id)+').value='+json.dumps(str(value))+';document.getElementById('+json.dumps(id)+').dispatchEvent(new Event("change"))')
    rendered_frame(client)


def check_motion_body(client,output,check):
    client.js('document.getElementById("source").value='+json.dumps(FIXTURE)+';document.getElementById("source").dispatchEvent(new Event("input"));document.getElementById("reload").click()')
    client.until('urdfWorkbench.viewer.robot?.robotName==="motion_body_fixture"&&!urdfWorkbench.loading')
    client.js('window.groundBeforeMotion='+GROUND+';window.cameraBeforeMotion=urdfWorkbench.viewer.camera.position.toArray();window.targetBeforeMotion=urdfWorkbench.viewer.controls.target.toArray();document.getElementById("animate").click()')
    check(client,GROUND+'&&JSON.stringify('+GROUND+')===JSON.stringify(groundBeforeMotion)&&JSON.stringify(urdfWorkbench.viewer.camera.position.toArray())===JSON.stringify(cameraBeforeMotion)&&JSON.stringify(urdfWorkbench.viewer.controls.target.toArray())===JSON.stringify(targetBeforeMotion)','entering motion freezes ground transforms without auto-refitting or moving the camera target')
    client.js(POINT_ORACLE)
    click_link(client,'chassis')
    check(client,'!!urdfWorkbench.motion.bodyEditor&&!document.getElementById("motion-body-editor").hidden&&document.querySelectorAll("#motion-body-editor input").length===6','clicking a fixed chassis below an empty root opens the six-axis body editor')
    client.js('window.bodyEditorBeforeOrbit=[document.getElementById("motion-body-editor").style.left,document.getElementById("motion-body-editor").style.top];(()=>{const T=parcelRequire("../node_modules/three/build/three.module.js"),v=urdfWorkbench.viewer,d=v.camera.position.clone().sub(v.controls.target);d.applyAxisAngle(new T.Vector3(0,1,0),.7);v.camera.position.copy(v.controls.target).add(d);v.controls.update();v.redraw()})()');rendered_frame(client)
    check(client,'!document.getElementById("motion-body-editor").hidden&&JSON.stringify([document.getElementById("motion-body-editor").style.left,document.getElementById("motion-body-editor").style.top])!==JSON.stringify(bodyEditorBeforeOrbit)','body editor follows its selected point while orbiting the camera')
    change(client,'motion-body-2',.4)
    change(client,'motion-body-4',720)
    check(client,'Math.abs(urdfWorkbench.viewer.robot.position.z-.4)<1e-8&&Number(document.getElementById("motion-root-4").value)===720&&urdfWorkbench.motion.project.keyframes.length===0','body popover edits apply live and synchronize with the sidebar without overwriting keyframes')
    change(client,'motion-body-0',0)
    client.js('document.getElementById("motion-body-record").click()')
    check(client,'urdfWorkbench.motion.project.keyframes.length===1&&Math.abs(urdfWorkbench.motion.project.keyframes[0].base.rpy[1]-4*Math.PI)<1e-8','recording from the body editor retains 720-degree winding')
    change(client,'motion-body-2',1.2)
    client.js('document.getElementById("motion-body-update").click()')
    check(client,'urdfWorkbench.motion.project.keyframes.length===1&&Math.abs(urdfWorkbench.motion.project.keyframes[0].base.position[2]-1.2)<1e-8','update from the body editor writes the current base pose to the selected key')
    client.js('document.getElementById("fit-view").click();window.targetBeforeEnvironment=urdfWorkbench.viewer.controls.target.toArray();urdfWorkbench.viewer._updateEnvironment()')
    check(client,'JSON.stringify('+GROUND+')===JSON.stringify(groundBeforeMotion)&&JSON.stringify(urdfWorkbench.viewer.controls.target.toArray())===JSON.stringify(targetBeforeEnvironment)','Fit view and the renderer environment update cannot move the ground or make the camera follow the airborne base')
    change(client,'motion-body-2',1.4)
    client.js('document.getElementById("motion-body-record").click();document.getElementById("motion-rewind").click();document.getElementById("motion-play").click()');rendered_frame(client)
    check(client,'JSON.stringify('+GROUND+')===JSON.stringify(groundBeforeMotion)','keyframe playback keeps the original plane and grid fixed')
    client.js('if(urdfWorkbench.motion.playing)document.getElementById("motion-play").click()')
    # Make both poses visible before exercising model clicks and measurement modifiers.
    client.js('document.getElementById("motion-fit").click()')
    client.screenshot(str(output/'motion-body-editor.png'))
    change(client,'motion-body-4',0)
    change(client,'motion-body-2',0)
    client.js('document.getElementById("motion-body-close").click();document.getElementById("fit-view").click()')
    click_link(client,'chassis',modifiers=2)
    click_link(client,'arm',modifiers=2)
    check(client,'!urdfWorkbench.motion.bodyEditor&&urdfWorkbench.linkMeasurement.links.length===2','Ctrl link measurement does not open the body pose editor')
    client.js('document.getElementById("link-measure-clear").click()')
    click_link(client,'chassis')
    click_link(client,'arm')
    check(client,'!urdfWorkbench.motion.bodyEditor','clicking a moving link closes the body editor and retains joint interaction')
    point=client.js('pickTestPoint("arm")')
    pointer(client,'mouseMoved',point,button='none')
    pointer(client,'mousePressed',point,clickCount=1)
    check(client,'!!urdfWorkbench.viewer.dragControls.manipulating','movable-link dragging still manipulates the joint in motion mode')
    pointer(client,'mouseReleased',point,clickCount=1)
    click_link(client,'chassis')
    client.call('Emulation.setDeviceMetricsOverride',{'width':390,'height':844,'deviceScaleFactor':1,'mobile':False});rendered_frame(client)
    check(client,'(()=>{const p=document.getElementById("motion-body-editor"),r=p.getBoundingClientRect();return !p.hidden&&r.left>=0&&r.right<=innerWidth&&r.top>=0&&r.bottom<=innerHeight&&p.scrollWidth<=p.clientWidth})()','direct body editor stays reachable on a phone screen')
    client.screenshot(str(output/'motion-body-mobile.png'))
    client.call('Emulation.setDeviceMetricsOverride',{'width':1897,'height':942,'deviceScaleFactor':1,'mobile':False});rendered_frame(client)
    client.js('document.getElementById("motion-body-4").focus()')
    client.call('Input.dispatchKeyEvent',{'type':'keyDown','key':'Escape','code':'Escape','windowsVirtualKeyCode':27})
    client.call('Input.dispatchKeyEvent',{'type':'keyUp','key':'Escape','code':'Escape','windowsVirtualKeyCode':27})
    check(client,'!urdfWorkbench.motion.bodyEditor','Escape dismisses the body editor')
    empty=client.js('(()=>{const p=urdfWorkbench.motion.project;p.keyframes=[];return p})()')
    client.js('(()=>{const d=new DataTransfer();d.items.add(new File(['+json.dumps(json.dumps(empty))+'],"empty.motion.json"));const i=document.getElementById("motion-import");i.files=d.files;i.dispatchEvent(new Event("change"));})()');rendered_frame(client)
    client.js('document.getElementById("motion-close").click();document.getElementById("example-select").value="urdf/go2/urdf/go2.urdf";document.getElementById("load-example").click()')
    client.until('urdfWorkbench.viewer.robot?.robotName==="go2_description"&&!urdfWorkbench.loading',55)
    # A slope must keep its original normal and offset when the base is previewed.
    client.js('for(const side of ["FL","FR","RL","RR"]){urdfWorkbench.viewer.setJointValue(side+"_thigh_joint",.8);urdfWorkbench.viewer.setJointValue(side+"_calf_joint",-1.6)}document.getElementById("statics-toggle").click();document.getElementById("statics-place-slope").click()');rendered_frame(client)
    check(client,'urdfWorkbench.statics.terrain.active','slope fixture is placed before testing motion ground locking')
    client.js('window.slopeGroundBeforeMotion='+GROUND+';window.slopeRootBeforeMotion={p:urdfWorkbench.viewer.robot.position.toArray(),q:urdfWorkbench.viewer.robot.quaternion.toArray()};document.getElementById("animate").click()')
    change(client,'motion-root-2',.5)
    client.js('document.getElementById("fit-view").click();urdfWorkbench.viewer._updateEnvironment()')
    check(client,'JSON.stringify('+GROUND+')===JSON.stringify(slopeGroundBeforeMotion)&&!urdfWorkbench.statics.enabled&&Math.abs(urdfWorkbench.motion.project.model.groundPlane.normal[0])>.01','previewing base motion preserves an existing slope normal and offset while suspending statics')
    client.js('document.getElementById("motion-close").click();document.getElementById("statics-flat").click();document.getElementById("right-code-tab").click();document.getElementById("reset-joints").click()')
    print('PASS fixed ground in flat/slope motion, direct body pose editing, winding, keyframe recording, measurement and joint-drag coexistence',flush=True)
