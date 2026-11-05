"""Geometric snapping and signed 3D measurements through real canvas input."""
import json
import math
from model_interaction import POINT_ORACLE, pointer, rendered_frame
from link_measurement import choose

FIXTURE='''<robot name="surface_fixture">
<link name="base"><visual><origin xyz="-.35 0 .65"/><geometry><box size=".4 .4 .4"/></geometry></visual></link>
<link name="drum"><visual><geometry><cylinder radius=".13" length=".32"/></geometry></visual></link>
<joint name="drum_joint" type="revolute"><parent link="base"/><child link="drum"/><origin xyz=".35 0 .65"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="10" velocity="2"/></joint>
</robot>'''


def hover(client,name):
    point=client.js('pickTestPoint('+json.dumps(name)+')')
    pointer(client,'mouseMoved',point,button='none',modifiers=8)
    client.until('!document.getElementById("measure-snap-hint").hidden&&!document.getElementById("measure-snap-hint").textContent.includes("正在")&&document.querySelectorAll(`#measure-snap-overlay .snap-point[data-link="'+name+'"]`).length>0')


def pick_world(client,point,modifiers=8):
    screen=client.js('(()=>{const T=parcelRequire("../node_modules/three/build/three.module.js"),p=new T.Vector3(...'+json.dumps(point)+').project(urdfWorkbench.viewer.camera);return {x:(p.x+1)*innerWidth/2,y:(1-p.y)*innerHeight/2}})()')
    pointer(client,'mouseMoved',screen,button='none',modifiers=modifiers)
    for event in ['mousePressed','mouseReleased']:
        pointer(client,event,screen,modifiers=modifiers,clickCount=1)
    rendered_frame(client)


def choose_model(client,name,modifiers=2):
    point=client.js('pickTestPoint('+json.dumps(name)+')')
    pointer(client,'mouseMoved',point,button='none',modifiers=modifiers)
    for event in ['mousePressed','mouseReleased']:
        pointer(client,event,point,modifiers=modifiers,clickCount=1)
    rendered_frame(client)


def drag_after_ctrl(client,check,count):
    client.call('Input.dispatchKeyEvent',{'type':'keyUp','key':'Control','code':'ControlLeft','windowsVirtualKeyCode':17,'modifiers':0})
    point=client.js('pickTestPoint("drum")')
    # No intervening mouse move: the first ordinary press after Ctrl must grab.
    pointer(client,'mousePressed',point,clickCount=1)
    check(client,'urdfWorkbench.viewer.dragControls.manipulating?.name==="drum_joint"&&urdfWorkbench.linkMeasurement.links.length==='+str(count),'releasing Ctrl immediately restores joint dragging with '+str(count)+' measured link(s)')
    moved={'x':point['x']+35,'y':point['y']+20}
    pointer(client,'mouseMoved',moved,buttons=1)
    pointer(client,'mouseReleased',moved)
    rendered_frame(client)
    check(client,'Math.abs(urdfWorkbench.viewer.robot.joints.drum_joint.angle)>.001&&urdfWorkbench.viewer.dragControls.manipulating===null&&urdfWorkbench.linkMeasurement.links.length==='+str(count),'joint movement preserves the existing measurement endpoints')
    client.js('urdfWorkbench.viewer.setJointValue("drum_joint",0)')
    rendered_frame(client)


def check_surface_measurement(client,output,check):
    client.js('document.getElementById("source").value='+json.dumps(FIXTURE)+';document.getElementById("source").dispatchEvent(new Event("input"));document.getElementById("reload").click()')
    client.until('urdfWorkbench.viewer.robot?.robotName==="surface_fixture"&&!urdfWorkbench.loading')
    client.js(POINT_ORACLE)
    choose_model(client,'base')
    check(client,'urdfWorkbench.linkMeasurement.links.length===1&&urdfWorkbench.linkMeasurement.links[0]==="base"&&urdfWorkbench.surfaceMeasurement.anchors.length===0&&document.getElementById("measure-snap-overlay").hasAttribute("hidden")','Ctrl + 3D click starts link measurement without selecting surface points')
    check(client,'document.getElementById("graph-measure").getAttribute("aria-pressed")==="false"','Ctrl selection does not latch the Measure button')
    drag_after_ctrl(client,check,1)
    choose_model(client,'base')
    check(client,'urdfWorkbench.linkMeasurement.links.length===1','3D link measurement rejects a duplicate link')
    blank={'x':client.js('innerWidth/2'),'y':80}
    for event in ['mousePressed','mouseReleased']:
        pointer(client,event,blank,modifiers=2,clickCount=1)
    check(client,'urdfWorkbench.linkMeasurement.links.length===1','Ctrl clicking empty 3D space preserves the first link')
    choose_model(client,'drum')
    check(client,'JSON.stringify(urdfWorkbench.linkMeasurement.links)===JSON.stringify(["base","drum"])&&urdfWorkbench.linkMeasurement.delta.every((n,i)=>Math.abs(n-[.35,.65,0][i])<1e-8)&&document.querySelectorAll(".structure-node.measured-link").length===2','Ctrl selects two actual 3D links and measures their origins with synchronized topology highlights')
    check(client,'urdfWorkbench.viewer.robot.joints.drum_joint.angle===0','Ctrl link selection does not drag the joint')
    drag_after_ctrl(client,check,2)
    choose_model(client,'base',modifiers=0)
    check(client,'urdfWorkbench.selectedObject?.name==="base"&&JSON.stringify(urdfWorkbench.linkMeasurement.links)===JSON.stringify(["base","drum"])','ordinary clicks select links normally while retaining Ctrl measurement results')
    choose(client,'drum')
    check(client,'urdfWorkbench.selectedObject?.name==="drum"&&urdfWorkbench.linkMeasurement.links.length===2','ordinary structure selection does not add measurement endpoints')
    choose_model(client,'base')
    choose(client,'drum',modifiers=2)
    check(client,'JSON.stringify(urdfWorkbench.linkMeasurement.links)===JSON.stringify(["base","drum"])','3D and structure Ctrl selections can be mixed in one measurement')
    client.js('document.getElementById("graph-measure").click()')
    choose_model(client,'base',modifiers=0)
    choose_model(client,'drum',modifiers=0)
    check(client,'urdfWorkbench.linkMeasurement.links.length===2&&urdfWorkbench.viewer.robot.joints.drum_joint.angle===0','active link measurement also accepts ordinary 3D clicks without joint dragging')
    camera=client.js('urdfWorkbench.viewer.camera.position.toArray()')
    pointer(client,'mousePressed',blank,buttons=1,clickCount=1)
    pointer(client,'mouseMoved',{'x':blank['x']+45,'y':blank['y']+25},buttons=1)
    pointer(client,'mouseReleased',{'x':blank['x']+45,'y':blank['y']+25},clickCount=1)
    rendered_frame(client)
    check(client,'urdfWorkbench.viewer.camera.position.toArray().some((n,i)=>Math.abs(n-'+json.dumps(camera)+'[i])>1e-6)&&urdfWorkbench.linkMeasurement.links.length===2','dragging empty 3D space still orbits the camera during link measurement')
    client.js('document.getElementById("fit-view").click()')
    rendered_frame(client)
    pointer(client,'mouseMoved',client.js('pickTestPoint("base")'),button='none')
    client.call('Input.dispatchKeyEvent',{'type':'keyDown','key':'Shift','code':'ShiftLeft','windowsVirtualKeyCode':16,'modifiers':8})
    client.until('!document.getElementById("measure-snap-overlay").hasAttribute("hidden")')
    client.call('Input.dispatchKeyEvent',{'type':'keyUp','key':'Shift','code':'ShiftLeft','windowsVirtualKeyCode':16,'modifiers':0})
    check(client,'document.getElementById("measure-snap-overlay").hasAttribute("hidden")','Shift key press/release shows and hides snap points without mouse movement')
    client.call('Input.dispatchKeyEvent',{'type':'keyDown','key':'Control','code':'ControlLeft','windowsVirtualKeyCode':17,'modifiers':2})
    check(client,'document.getElementById("measure-snap-overlay").hasAttribute("hidden")&&urdfWorkbench.surfaceMeasurement.anchors.length===0','Ctrl alone never enables surface snap points')
    client.call('Input.dispatchKeyEvent',{'type':'keyUp','key':'Control','code':'ControlLeft','windowsVirtualKeyCode':17,'modifiers':0})
    hover(client,'base')
    check(client,'document.querySelectorAll("#measure-snap-overlay .snap-point").length>0&&document.querySelectorAll("#measure-snap-overlay .snap-point").length<=5','Shift hover shows at most five box snap candidates')
    pick_world(client,[-.15,.85,.2])
    client.until('urdfWorkbench.surfaceMeasurement.anchors.length===1')
    check(client,'!urdfWorkbench.linkMeasurement.enabled&&urdfWorkbench.linkMeasurement.links.length===0','Shift point measurement switches cleanly out of link measurement')
    check(client,'urdfWorkbench.surfaceMeasurement.anchors[0].kind==="rectangleVertex"&&urdfWorkbench.surfaceMeasurement.anchors[0].world.every((n,i)=>Math.abs(n-[-.15,.85,.2][i])<1e-8)','a box corner snaps to its exact 3D location')
    hover(client,'drum')
    check(client,'document.querySelectorAll("#measure-snap-overlay .snap-point").length<=5&&document.querySelectorAll("#measure-snap-overlay .snap-point[data-kind=circleCenter]").length>0','cylinder recommendations retain face centers within the five-point limit')
    pick_world(client,[.35,.81,0])
    client.until('urdfWorkbench.surfaceMeasurement.anchors.length===2')
    check(client,'urdfWorkbench.surfaceMeasurement.anchors[1].kind==="circleCenter"&&urdfWorkbench.surfaceMeasurement.delta.every((n,i)=>Math.abs(n-[.5,-.04,-.2][i])<1e-8)','circular face centers produce correct signed XYZ distances')
    expected=math.sqrt(.5**2+.04**2+.2**2)
    check(client,'Math.abs(urdfWorkbench.surfaceMeasurement.distance-'+str(expected)+')<1e-8&&document.getElementById("surface-measure-distance").textContent==='+json.dumps(f'{expected*1000:.1f} mm'),'surface distances default to millimeters')
    check(client,'JSON.stringify([...document.querySelectorAll("#link-measure-overlay text")].map(el=>el.textContent))===JSON.stringify(["ΔX 500.0mm","ΔY -40.0mm","ΔZ -200.0mm","'+f'{expected*1000:.1f}'+'mm"])','surface measurement shares the colored XYZ and dashed total annotations')
    pointer(client,'mouseMoved',{'x':10,'y':65},button='none')
    client.screenshot(str(output/'surface-measurement.png'))
    client.js('document.getElementById("surface-measure-unit").value="m";document.getElementById("surface-measure-unit").dispatchEvent(new Event("change"))')
    check(client,'document.getElementById("surface-measure-distance").textContent==='+json.dumps(f'{expected:.4f} m'),'surface measurements can switch to meters')
    client.js('urdfWorkbench.viewer.setJointValue("drum_joint",.5)')
    rendered_frame(client)
    delta=[.5+.16*math.sin(.5),.16*math.cos(.5)-.2,-.2]
    check(client,'urdfWorkbench.surfaceMeasurement.delta.every((n,i)=>Math.abs(n-'+json.dumps(delta)+'[i])<1e-8)','snapped points follow the link when its joint moves')
    client.js('document.getElementById("up-select").value="+Y";document.getElementById("up-select").dispatchEvent(new Event("change"))')
    rendered_frame(client)
    check(client,'urdfWorkbench.surfaceMeasurement.anchors.length===2&&urdfWorkbench.surfaceMeasurement.delta.every((n,i)=>Math.abs(n-'+json.dumps([delta[0],.2,delta[1]])+'[i])<1e-8)','changing the up axis preserves points and recomputes scene components')
    client.js('urdfWorkbench.viewer.focus();window.dispatchEvent(new KeyboardEvent("keydown",{key:"Escape",bubbles:true}));document.getElementById("up-select").value="+Z";document.getElementById("up-select").dispatchEvent(new Event("change"));urdfWorkbench.viewer.setJointValue("drum_joint",0)')
    rendered_frame(client)
    check(client,'urdfWorkbench.surfaceMeasurement.anchors.length===0&&document.getElementById("measure-result").hidden&&document.getElementById("measure-snap-overlay").hasAttribute("hidden")','Escape removes endpoints and snap previews')
    hover(client,'base')
    pick_world(client,[-.15,.85,.2],modifiers=9)
    client.until('urdfWorkbench.surfaceMeasurement.anchors.length===1')
    check(client,'urdfWorkbench.surfaceMeasurement.anchors[0].kind==="surfacePoint"','Alt bypasses snapping and retains arbitrary surface picks')
    choose_model(client,'base')
    check(client,'urdfWorkbench.surfaceMeasurement.anchors.length===0&&urdfWorkbench.linkMeasurement.links[0]==="base"&&document.getElementById("measure-result").hidden','Ctrl link selection clears an unfinished Shift point measurement')
    client.js('document.querySelector("#measure-result button").click()')
    # Known tessellated geometry tests have numeric oracles, independent of DOM styling.
    result=client.js('''(async()=>{
      const T=parcelRequire('../node_modules/three/build/three.module.js'),extractor=new URDFMeasurementFeatures(T);
      const cylinder=new T.CylinderGeometry(.2,.2,.5,32).toNonIndexed(),open=new T.CylinderGeometry(.2,.2,.5,32,1,true).toNonIndexed();
      const [closedFeatures,openFeatures,rect]=await Promise.all([extractor.get(cylinder),extractor.get(open),extractor.get(new T.BoxGeometry(.6,.4,.2).toNonIndexed())]);
      const mesh=new T.Mesh(cylinder);mesh.rotation.set(.4,.6,.3);mesh.position.set(.1,.3,.4);mesh.updateMatrixWorld(true);
      const candidates=extractor.worldCandidates(mesh,closedFeatures),center=new T.Vector3(0,.25,0).applyMatrix4(mesh.matrixWorld);
      let n=new T.Vector3(0,1,0).transformDirection(mesh.matrixWorld);
      const dominant=[0,1,2].sort((a,b)=>Math.abs(n.getComponent(b))-Math.abs(n.getComponent(a)))[0];if(n.getComponent(dominant)<0)n.negate();
      const u=new T.Vector3(1,0,0).addScaledVector(n,-n.x).normalize(),v=n.clone().cross(u);
      const upper=candidates.filter(c=>c.kind==='quadrant'&&Math.abs(c.local.y-.25)<1e-5);
      const directions=upper.every(c=>{const d=c.world.clone().sub(center);return Math.abs(d.length()-.2)<1e-6&&Math.abs(d.dot(n))<1e-7&&d.clone().normalize().distanceTo((c.angle%180===0?u:v).clone().multiplyScalar(c.angle<180?1:-1))<1e-6});
      const bad=new T.CylinderGeometry(.2,.2,.5,12).toNonIndexed(),p=bad.getAttribute('position');for(let i=0;i<p.count;i++)p.setX(i,p.getX(i)*1.6);const elliptical=await extractor.get(bad);
      return {closed:closedFeatures.circles.length,open:openFeatures.circles.length,fitted:closedFeatures.circles.every(c=>c.fitted),rectangle:rect.points.filter(p=>p.kind==='rectangleCenter').length,upper:upper.length,directions,ellipse:elliptical.circles.length,cached:(await extractor.get(cylinder))===closedFeatures};
    })()''')
    assert result=={'closed':3,'open':3,'fitted':True,'rectangle':6,'upper':4,'directions':True,'ellipse':0,'cached':True},result
    print('PASS triangulated closed/open cylinders and rectangle faces are recognized; rotated quadrant points follow world axes; ellipses are rejected; feature extraction is cached',flush=True)
    # Rendered feature preview, useful when inspecting marker placement and the tooltip.
    hover(client,'drum')
    point=client.js('(()=>{const c=[...document.querySelectorAll("#measure-snap-overlay [data-kind=quadrant]")].find(c=>Number(c.getAttribute("cx"))>innerWidth*.3);return {x:Number(c.getAttribute("cx")),y:Number(c.getAttribute("cy"))}})()')
    pointer(client,'mouseMoved',point,button='none',modifiers=8)
    rendered_frame(client)
    client.screenshot(str(output/'surface-snap-preview.png'))
    client.js('document.getElementById("reload").click()')
    client.until('!urdfWorkbench.loading')
    check(client,'urdfWorkbench.surfaceMeasurement.anchors.length===0&&document.getElementById("measure-snap-overlay").children.length===0','reload clears stale surface anchors and snap features')
