"""Axis-position readout, delayed dismissal and constrained direct editing."""
import json
import time
from model_interaction import FIXTURE as PICKING_FIXTURE, POINT_ORACLE, pointer, rendered_frame

FIXTURE=PICKING_FIXTURE.replace('picking_fixture','angle_editor_fixture').replace('</robot>','''
<link name="spin"><visual><geometry><sphere radius=".1"/></geometry></visual></link>
<joint name="spin_joint" type="continuous"><parent link="base"/><child link="spin"/><origin xyz="0 .35 .7"/><axis xyz="0 0 1"/></joint></robot>''')


def drag(client,name):
    point=client.js('pickTestPoint('+json.dumps(name)+')')
    pointer(client,'mouseMoved',point,button='none')
    pointer(client,'mousePressed',point,clickCount=1)
    moved={'x':point['x']+35,'y':point['y']+20}
    pointer(client,'mouseMoved',moved,buttons=1)
    rendered_frame(client)
    return moved


def release(client,point):
    pointer(client,'mouseReleased',point)
    pointer(client,'mouseMoved',{'x':10,'y':70},button='none')
    rendered_frame(client)


def click_readout(client):
    point=client.js('(()=>{const r=document.getElementById("joint-angle-readout").getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()')
    for event in ['mouseMoved','mousePressed','mouseReleased']:
        pointer(client,event,point,clickCount=1)
    rendered_frame(client)


def edit(client,value):
    client.js('document.getElementById("joint-angle-readout").click();document.getElementById("joint-angle-input").value='+json.dumps(str(value))+';document.getElementById("joint-angle-input").dispatchEvent(new Event("input"));document.querySelector("#joint-angle-form button[type=submit]").click()')
    rendered_frame(client)


def check_joint_angle_editor(client,output,check):
    client.js('document.getElementById("source").value='+json.dumps(FIXTURE)+';document.getElementById("source").dispatchEvent(new Event("input"));document.getElementById("reload").click()')
    client.until('urdfWorkbench.viewer.robot?.robotName==="angle_editor_fixture"&&!urdfWorkbench.loading')
    client.js(POINT_ORACLE)
    client.js('document.getElementById("unit-rad").click();if(urdfWorkbench.display.jointAxes)document.querySelector("[data-display=jointAxes]").click()')
    moved=drag(client,'left_leg')
    check(client,'(()=>{const root=document.getElementById("joint-angle-tooltip"),b=document.getElementById("joint-angle-readout");return !root.hidden&&root.dataset.joint==="left_joint"&&b.disabled&&Math.abs(parseFloat(b.textContent)-urdfWorkbench.viewer.robot.joints.left_joint.angle)<.00051})()','joint-axis readout shows the live angle while dragging')
    release(client,moved)
    check(client,'!document.getElementById("joint-angle-tooltip").hidden&&!document.getElementById("joint-angle-readout").disabled','release keeps the angle clickable')
    client.screenshot(str(output/'joint-axis-angle-readout.png'))
    time.sleep(.7)
    click_readout(client)
    check(client,'!document.getElementById("joint-angle-form").hidden&&document.activeElement.id==="joint-angle-input"','clicking the angle opens its inline editor through real pointer input')
    pointer(client,'mouseMoved',{'x':10,'y':70},button='none')
    client.screenshot(str(output/'joint-axis-angle-edit.png'))
    time.sleep(3.2)
    check(client,'!document.getElementById("joint-angle-tooltip").hidden&&!document.getElementById("joint-angle-form").hidden','the three-second hide timer pauses throughout editing')
    client.js('document.getElementById("joint-angle-input").value=".4";document.getElementById("joint-angle-input").dispatchEvent(new Event("input"))')
    for kind in ['keyDown','keyUp']:
        client.call('Input.dispatchKeyEvent',{'type':kind,'key':'Enter','code':'Enter','windowsVirtualKeyCode':13})
    rendered_frame(client)
    check(client,'Math.abs(urdfWorkbench.viewer.robot.joints.left_joint.angle-.4)<1e-8&&Math.abs(Number(document.querySelector(`#joint-list li[data-name="left_joint"] input[type=number]`).value)-.4)<1e-8&&document.getElementById("joint-angle-form").hidden','Enter applies the angle and synchronizes the joint card')
    # An explicit request for constrained direct editing takes precedence over a disabled toolbar limit toggle.
    client.js('if(document.getElementById("limits").getAttribute("aria-pressed")==="true")document.getElementById("limits").click()')
    edit(client,100)
    check(client,'urdfWorkbench.viewer.robot.joints.left_joint.angle===1&&document.getElementById("limits").getAttribute("aria-pressed")==="false"&&document.getElementById("joint-angle-feedback").textContent.includes("限位")','direct angle input clamps to the upper URDF limit even with toolbar limits off')
    edit(client,-100)
    check(client,'urdfWorkbench.viewer.robot.joints.left_joint.angle===-1','direct angle input clamps to the lower URDF limit')
    client.js('document.getElementById("unit-deg").click()')
    rendered_frame(client)
    check(client,'document.getElementById("joint-angle-readout").textContent.includes("°")','axis labels follow the degree/radian setting')
    edit(client,30)
    check(client,'Math.abs(urdfWorkbench.viewer.robot.joints.left_joint.angle-Math.PI/6)<1e-8','degree input converts accurately into the joint radian value')
    client.js('document.getElementById("joint-angle-readout").click();document.getElementById("joint-angle-input").value="";document.querySelector("#joint-angle-form button[type=submit]").click()')
    check(client,'Math.abs(urdfWorkbench.viewer.robot.joints.left_joint.angle-Math.PI/6)<1e-8&&!document.getElementById("joint-angle-form").hidden&&document.getElementById("joint-angle-feedback").textContent.length>0','blank input is rejected without moving the joint')
    client.js('document.getElementById("joint-angle-input").value="-20"')
    for kind in ['keyDown','keyUp']:
        client.call('Input.dispatchKeyEvent',{'type':kind,'key':'Escape','code':'Escape','windowsVirtualKeyCode':27})
    check(client,'Math.abs(urdfWorkbench.viewer.robot.joints.left_joint.angle-Math.PI/6)<1e-8&&document.getElementById("joint-angle-form").hidden','Escape cancels an edit without changing its angle')
    pointer(client,'mouseMoved',{'x':10,'y':70},button='none')
    started=time.monotonic()
    client.until('document.getElementById("joint-angle-tooltip").hidden',timeout=5)
    assert time.monotonic()-started>2.5
    check(client,'(()=>{let shown=false;urdfWorkbench.viewer.robot.traverse(o=>{if(o.userData.axisKind==="jointAxes"&&o.visible)shown=true});return !shown})()','the label and temporary axis disappear three seconds after interaction ends')
    moved=drag(client,'left_leg');release(client,moved)
    same=client.js('pickTestPoint("left_leg")')
    for event in ['mousePressed','mouseReleased']:
        pointer(client,event,same,clickCount=1)
    check(client,'!document.getElementById("joint-angle-tooltip").hidden&&document.getElementById("joint-angle-tooltip").dataset.joint==="left_joint"','clicking the current joint link preserves its axis readout')
    blank={'x':client.js('innerWidth/2'),'y':80}
    for event in ['mousePressed','mouseReleased']:
        pointer(client,event,blank,clickCount=1)
    check(client,'document.getElementById("joint-angle-tooltip").hidden&&!urdfWorkbench.viewer.robot.joints.left_joint.children.some(o=>o.userData.axisKind==="jointAxes"&&o.visible)','clicking empty 3D space immediately hides the readout and temporary axis')
    moved=drag(client,'left_leg');release(client,moved)
    other=client.js('pickTestPoint("base")')
    other['x']+=90  # Avoid the floating angle editor over the body's center.
    for event in ['mousePressed','mouseReleased']:
        pointer(client,event,other,clickCount=1)
    check(client,'document.getElementById("joint-angle-tooltip").hidden&&urdfWorkbench.selectedObject?.name==="base"','clicking another link immediately dismisses the previous joint readout')
    moved=drag(client,'left_leg');release(client,moved)
    click_readout(client)
    before=client.js('urdfWorkbench.viewer.robot.joints.left_joint.angle')
    client.js('document.getElementById("joint-angle-input").value=".2"')
    panel=client.js('(()=>{const r=document.getElementById("tab-files").getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()')
    for event in ['mousePressed','mouseReleased']:
        pointer(client,event,panel,clickCount=1)
    check(client,'document.getElementById("joint-angle-tooltip").hidden&&urdfWorkbench.viewer.robot.joints.left_joint.angle==='+str(before),'clicking another panel dismisses the editor without applying an unfinished value')
    moved=drag(client,'right_leg');release(client,moved)
    check(client,'document.getElementById("joint-angle-readout").textContent.includes("m")','prismatic joints display position in meters')
    edit(client,.8)
    check(client,'urdfWorkbench.viewer.robot.joints.right_joint.angle===.3','prismatic direct input respects its travel limit')
    moved=drag(client,'spin');release(client,moved)
    edit(client,720)
    check(client,'Math.abs(urdfWorkbench.viewer.robot.joints.spin_joint.angle-4*Math.PI)<1e-8','continuous joints accept multiple turns without artificial limits')
    client.js('document.querySelector("[data-display=jointAxes]").click()')
    pointer(client,'mouseMoved',{'x':10,'y':70},button='none')
    client.until('document.getElementById("joint-angle-tooltip").hidden',timeout=5)
    check(client,'(()=>{const axes=[];urdfWorkbench.viewer.robot.traverse(o=>{if(o.userData.axisKind==="jointAxes")axes.push(o)});return axes.every(o=>o.visible)})()','label expiration preserves the explicitly enabled global joint axes')
    moved=drag(client,'left_leg');release(client,moved)
    client.js('document.getElementById("reload").click()')
    client.until('!urdfWorkbench.loading')
    check(client,'document.getElementById("joint-angle-tooltip").hidden&&!document.getElementById("joint-angle-tooltip").hasAttribute("data-joint")','reload clears stale joint editors and delayed hide callbacks')
    client.js('document.querySelector("[data-display=jointAxes]").click();document.getElementById("limits").click();document.getElementById("unit-rad").click()')
