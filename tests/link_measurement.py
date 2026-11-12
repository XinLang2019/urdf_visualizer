"""Measure known link-origin displacements through real structure-diagram input."""
import json
import math
import time
from model_interaction import pointer, rendered_frame

FIXTURE = '''<robot name="measure_fixture">
<link name="base"><visual><origin xyz="0 0 .85"/><geometry><box size=".7 .25 .25"/></geometry></visual></link>
<link name="hip"><visual><geometry><sphere radius=".06"/></geometry></visual></link>
<link name="knee"><visual><origin xyz="0 0 -.15"/><geometry><box size=".07 .07 .3"/></geometry></visual></link>
<link name="toe"><visual><origin xyz=".07 .01 -.01"/><geometry><box size=".2 .12 .08"/></geometry></visual></link>
<link name="toe_clone"/>
<joint name="hip_fixed" type="fixed"><parent link="base"/><child link="hip"/><origin xyz="0 0 .8"/></joint>
<joint name="knee_joint" type="revolute"><parent link="hip"/><child link="knee"/><origin xyz="0 0 -.3"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="10" velocity="2"/></joint>
<joint name="toe_fixed" type="fixed"><parent link="knee"/><child link="toe"/><origin xyz=".117 .264 -.4"/></joint>
<joint name="clone_fixed" type="fixed"><parent link="toe"/><child link="toe_clone"/></joint>
</robot>'''


def choose(client, name, modifiers=0):
    point=client.js('(()=>{const r=document.querySelector('+json.dumps(f'.structure-node[data-kind="link"][data-name="{name}"] rect')+').getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()')
    pointer(client, 'mousePressed', point, clickCount=1, modifiers=modifiers)
    pointer(client, 'mouseReleased', point, clickCount=1, modifiers=modifiers)
    rendered_frame(client)


def click(client, name):
    client.js('document.getElementById('+json.dumps(name)+').click()')
    rendered_frame(client)


def expect_delta(client, expected):
    return client.js('(()=>{const m=urdfWorkbench.linkMeasurement,e='+json.dumps(expected)+';return m.links.length===2&&m.delta.every((v,i)=>Math.abs(v-e[i])<1e-8)&&Math.abs(m.distance-Math.hypot(...e))<1e-8})()')


def check_link_measurement(client, output, check):
    client.js('document.getElementById("source").value='+json.dumps(FIXTURE)+';document.getElementById("source").dispatchEvent(new Event("input"));document.getElementById("reload").click()')
    client.until('urdfWorkbench.viewer.robot?.robotName==="measure_fixture"&&!urdfWorkbench.loading')
    click(client,'tab-structure')
    if not client.js('document.getElementById("graph-fixed").getAttribute("aria-pressed")==="true"'):
        click(client, 'graph-fixed')
    click(client, 'graph-measure')
    check(client, 'urdfWorkbench.linkMeasurement.enabled&&!document.getElementById("link-measure-panel").hidden', 'structure measurement mode provides endpoint and unit controls')
    client.js('document.querySelector(`.structure-node[data-name="knee_joint"]`).dispatchEvent(new MouseEvent("click",{bubbles:true}))')
    check(client, 'urdfWorkbench.linkMeasurement.links.length===0', 'joints cannot be used as link measurement endpoints')
    choose(client, 'hip')
    choose(client, 'hip')
    check(client, 'JSON.stringify(urdfWorkbench.linkMeasurement.links)===JSON.stringify(["hip"])', 'measurement requires two different links')
    choose(client, 'toe')
    assert expect_delta(client, [.117, -.7, -.264])
    expected=math.sqrt(.117**2+.7**2+.264**2)
    check(client, 'document.getElementById("link-measure-distance").textContent==='+json.dumps(f'{expected*1000:.1f} mm'), 'known 3D origin distance is displayed in millimeters')
    check(client, 'document.querySelectorAll(".structure-node.measured-link").length===2&&!document.getElementById("link-measure-overlay").hasAttribute("hidden")&&document.querySelectorAll("#link-measure-overlay line").length===4', 'both links are highlighted and the 3D projection has a dashed diagonal plus XYZ segments')
    check(client, 'JSON.stringify([...document.querySelectorAll("#link-measure-overlay text")].slice(0,3).map(t=>t.textContent))===JSON.stringify(["ΔX 117.0mm","ΔY -700.0mm","ΔZ -264.0mm"])', 'XYZ labels preserve signs in scene coordinates')
    projection_check='''(()=>{const T=parcelRequire('../node_modules/three/build/three.module.js'),v=urdfWorkbench.viewer;v.world.updateWorldMatrix(true,true);return ['hip','toe'].every((name,i)=>{const p=v.robot.links[name].getWorldPosition(new T.Vector3()).project(v.camera),circle=document.querySelectorAll('#link-measure-overlay circle')[i];return Math.abs(Number(circle.getAttribute('cx'))-(p.x+1)*innerWidth/2)<.1&&Math.abs(Number(circle.getAttribute('cy'))-(1-p.y)*innerHeight/2)<.1})})()'''
    check(client, projection_check, 'annotation endpoints align with link origins rather than mesh centers')
    client.screenshot(str(output/'link-measurement.png'))
    before=client.js('document.querySelector("#link-measure-overlay circle").getAttribute("cx")')
    client.js('(()=>{const v=urdfWorkbench.viewer,d=v.camera.position.clone().sub(v.controls.target);d.applyAxisAngle(new (parcelRequire("../node_modules/three/build/three.module.js").Vector3)(0,1,0),.4);v.camera.position.copy(v.controls.target).add(d);v.controls.update();v.redraw()})()')
    rendered_frame(client)
    assert expect_delta(client, [.117, -.7, -.264])
    check(client, projection_check+'&&document.querySelector("#link-measure-overlay circle").getAttribute("cx")!=='+json.dumps(before), 'orbiting updates screen annotations without changing 3D distances')
    client.js('document.getElementById("link-measure-unit").value="m";document.getElementById("link-measure-unit").dispatchEvent(new Event("change"))')
    check(client, 'document.getElementById("link-measure-distance").textContent==='+json.dumps(f'{expected:.4f} m'), 'measurement units switch to meters')
    client.js('const measureSlider=document.querySelector(`#joint-list li[data-name="knee_joint"] input[type=range]`);measureSlider.value=.5;measureSlider.dispatchEvent(new Event("input"))')
    rendered_frame(client)
    assert expect_delta(client, [.117*math.cos(.5)-.4*math.sin(.5), -.3-.117*math.sin(.5)-.4*math.cos(.5), -.264])
    check(client, 'urdfWorkbench.linkMeasurement.links.length===2', 'joint slider edits retain and update link measurements')
    client.js('urdfWorkbench.viewer.setJointValue("knee_joint",0)')
    rendered_frame(client)
    click(client, 'animate')
    click(client, 'motion-record')
    client.js('urdfWorkbench.viewer.setJointValue("knee_joint",.5)')
    click(client, 'motion-record')
    click(client, 'motion-rewind')
    click(client, 'motion-play')
    client.until(f'Math.abs(urdfWorkbench.linkMeasurement.distance-({expected}))>.000001')
    click(client, 'motion-play')
    check(client, 'urdfWorkbench.linkMeasurement.links.length===2', 'keyframe playback keeps live endpoint measurements')
    client.js('(()=>{const p=urdfWorkbench.motion.project;p.keyframes=[];const d=new DataTransfer();d.items.add(new File([JSON.stringify(p)],"empty.motion.json"));const i=document.getElementById("motion-import");i.files=d.files;i.dispatchEvent(new Event("change"));})()')
    rendered_frame(client)
    click(client, 'motion-close')
    click(client, 'tab-structure')
    client.js('urdfWorkbench.viewer.setJointValue("knee_joint",0);document.getElementById("up-select").value="+Y";document.getElementById("up-select").dispatchEvent(new Event("change"))')
    rendered_frame(client)
    assert expect_delta(client, [.117, .264, -.7])
    check(client, 'urdfWorkbench.linkMeasurement.links.length===2', 'changing the up axis recomputes signed components and preserves distance')
    client.js('document.getElementById("up-select").value="+Z";document.getElementById("up-select").dispatchEvent(new Event("change"))')
    rendered_frame(client)
    click(client, 'graph-fixed')
    assert expect_delta(client, [.117, -.7, -.264])
    click(client, 'graph-fixed')
    check(client, 'document.querySelectorAll(".structure-node.measured-link").length===2', 'measurement endpoint highlights survive topology rebuilds')
    click(client, 'link-measure-clear')
    check(client, 'urdfWorkbench.linkMeasurement.enabled&&urdfWorkbench.linkMeasurement.links.length===0&&document.getElementById("link-measure-overlay").hasAttribute("hidden")', 'clear removes the pair and leaves measurement mode ready')
    choose(client, 'toe')
    choose(client, 'hip')
    assert expect_delta(client, [-.117, .7, .264])
    check(client, 'urdfWorkbench.linkMeasurement.distance>0', 'reversing endpoints reverses components and keeps total distance')
    choose(client, 'toe')
    choose(client, 'toe_clone')
    assert expect_delta(client, [0, 0, 0])
    check(client, '![...document.querySelectorAll("#link-measure-overlay line")].some(line=>["x1","x2","y1","y2"].some(key=>!Number.isFinite(Number(line.getAttribute(key)))))', 'coincident link origins produce a finite zero-distance measurement')
    client.js('urdfWorkbench.viewer.focus();window.dispatchEvent(new KeyboardEvent("keydown",{key:"Escape",bubbles:true}))')
    check(client, '!urdfWorkbench.linkMeasurement.enabled&&urdfWorkbench.linkMeasurement.links.length===0&&document.querySelectorAll(".measured-link").length===0', 'Escape exits link measurement mode and removes all annotations')
    choose(client, 'hip', modifiers=2)
    choose(client, 'toe', modifiers=2)
    check(client, 'urdfWorkbench.linkMeasurement.enabled&&urdfWorkbench.linkMeasurement.links.length===2', 'Ctrl + click structure nodes starts a two-link measurement')
    point=client.js('pickTestPoint("base")')
    for event_type in ['mousePressed','mouseReleased']:
        pointer(client,event_type,point,modifiers=8,clickCount=1)
    check(client, '!urdfWorkbench.linkMeasurement.enabled&&document.getElementById("status-text").textContent.includes("第二个点")', 'surface-point measurement switches cleanly out of link measurement')
    client.js('document.querySelector("#measure-result button").click()')
    click(client, 'graph-measure')
    choose(client, 'hip')
    choose(client, 'toe')
    client.js('window.measurementOldRobot=urdfWorkbench.viewer.robot;document.getElementById("reload").click()')
    client.until('urdfWorkbench.viewer.robot!==measurementOldRobot&&!urdfWorkbench.loading')
    check(client, '!urdfWorkbench.linkMeasurement.enabled&&urdfWorkbench.linkMeasurement.links.length===0&&document.getElementById("link-measure-overlay").children.length===0', 'model reload clears stale measurement references and graphics')
