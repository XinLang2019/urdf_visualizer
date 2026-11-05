"""Native double-click navigation and temporary manipulation-axis behavior."""
import json
from model_interaction import FIXTURE as PICKING_FIXTURE, POINT_ORACLE, pointer, rendered_frame

PADDING='\n'.join('<!-- editor padding '+str(i)+' -->' for i in range(45))
DUMMIES=''.join(f'<link name="dummy_{i}"/><joint name="dummy_joint_{i}" type="continuous"><parent link="base"/><child link="dummy_{i}"/><axis xyz="0 0 1"/></joint>' for i in range(20))
FIXTURE=PICKING_FIXTURE.replace('picking_fixture','navigation_fixture').replace('<link name="left_leg">',PADDING+'\n<!-- <link name="left_leg">fake block</link> -->\n<link name="left_leg">').replace('<joint name="left_joint"',DUMMIES+'<joint name="left_joint"').replace('><','>\n<')


def double_click(client,name,modifiers=0):
    point=client.js('pickTestPoint('+json.dumps(name)+')')
    pointer(client,'mouseMoved',point,button='none',modifiers=modifiers)
    for count in [1,2]:
        pointer(client,'mousePressed',point,clickCount=count,modifiers=modifiers)
        pointer(client,'mouseReleased',point,clickCount=count,modifiers=modifiers)
    rendered_frame(client)


def check_link_navigation(client,output,check):
    client.js('document.getElementById("source").value='+json.dumps(FIXTURE)+';document.getElementById("source").dispatchEvent(new Event("input"));document.getElementById("reload").click()')
    client.until('urdfWorkbench.viewer.robot?.robotName==="navigation_fixture"&&!urdfWorkbench.loading')
    client.js(POINT_ORACLE)
    client.js('document.getElementById("joint-search").value="no-match";document.getElementById("joint-search").dispatchEvent(new Event("input"));document.querySelector("[data-close=joints]").click();document.querySelector("[data-collapse=editor]").click()')
    original=client.js('document.getElementById("source").value')
    double_click(client,'left_leg')
    client.until('document.getElementById("source-locate-highlight").dataset.link==="left_leg"')
    check(client,'!document.getElementById("joints-panel").hidden&&!document.getElementById("editor-panel").classList.contains("collapsed")&&!document.querySelector(`#joint-list li[data-name="left_joint"]`).hidden','double-click opens panels and reveals the exact joint through active filters')
    check(client,'(()=>{const row=document.querySelector(`#joint-list li[data-name="left_joint"]`),body=document.querySelector("#joints-panel .panel-body"),r=row.getBoundingClientRect(),b=body.getBoundingClientRect();return r.top>=b.top&&r.bottom<=b.bottom&&body.scrollTop>0&&row.getAnimations().some(a=>a.effect.getTiming().iterations===2)})()','joint card scrolls into view and flashes exactly twice')
    check(client,'(()=>{const s=document.getElementById("source"),block=s.value.slice(s.selectionStart,s.selectionEnd),flash=document.getElementById("source-locate-highlight");return block.startsWith(`<link name="left_leg">`)&&block.endsWith("</link>")&&block.includes("<visual>")&&!block.includes("fake block")&&s.scrollTop>0&&!flash.hidden&&flash.getAnimations().some(a=>a.effect.getTiming().iterations===2)})()','editor selects the real link block, skips commented duplicates and flashes twice')
    check(client,'document.getElementById("source").value==='+json.dumps(original)+'&&document.getElementById("save-state").textContent==="已应用 · 未保存"','navigation preserves applied editor text and its unsaved state')
    client.screenshot(str(output/'double-click-link-navigation.png'))
    # Locate a fixed attachment, rather than incorrectly using its movable ancestor.
    client.js('document.getElementById("hide-fixed").checked=true;document.getElementById("hide-fixed").dispatchEvent(new Event("change"))')
    double_click(client,'badge')
    check(client,'document.getElementById("source-locate-highlight").dataset.link==="badge"&&!document.querySelector(`#joint-list li[data-name="badge_fixed"]`).hidden&&document.querySelector(`#joint-list li[data-name="badge_fixed"]`).getAnimations().length>0','fixed attachments reveal their own parent joint')
    # The live editor may have unapplied edits: locate against the current text.
    client.js('document.getElementById("source").value='+json.dumps('<!-- unapplied prefix -->\n')+'+document.getElementById("source").value;document.getElementById("source").dispatchEvent(new Event("input"))')
    double_click(client,'right_leg')
    check(client,'document.getElementById("source").value.slice(document.getElementById("source").selectionStart,document.getElementById("source").selectionEnd).startsWith(`<link name="right_leg">`)&&document.getElementById("save-state").textContent==="未应用"','unapplied source edits retain correct link block offsets')
    double_click(client,'base')
    check(client,'document.getElementById("source-locate-highlight").dataset.link==="base"&&document.getElementById("status-text").textContent.includes("根 link")','root links locate their source without inventing a parent joint')
    # Only the manipulated joint gets a temporary glyph when the global toggle is off.
    client.js('if(urdfWorkbench.display.jointAxes)document.querySelector("[data-display=jointAxes]").click()')
    point=client.js('pickTestPoint("left_leg")')
    pointer(client,'mousePressed',point,clickCount=1)
    check(client,'(()=>{const axes=[];urdfWorkbench.viewer.robot.traverse(o=>{if(o.userData.axisKind==="jointAxes"&&o.visible)axes.push(o)});return !urdfWorkbench.display.jointAxes&&axes.length===1&&axes[0].parent.name==="left_joint"&&axes[0].userData.dragActive})()','dragging shows only the actual joint axis without changing the global toggle')
    moved={'x':point['x']+35,'y':point['y']+20}
    pointer(client,'mouseMoved',moved,buttons=1)
    rendered_frame(client)
    client.screenshot(str(output/'temporary-drag-joint-axis.png'))
    pointer(client,'mouseReleased',moved)
    check(client,'(()=>{const axes=[];urdfWorkbench.viewer.robot.traverse(o=>{if(o.userData.axisKind==="jointAxes"&&o.visible)axes.push(o)});return axes.length===1&&axes[0].parent.name==="left_joint"&&!axes[0].userData.dragActive&&!document.getElementById("joint-angle-tooltip").hidden&&Math.abs(urdfWorkbench.viewer.robot.joints.left_joint.angle)>.001})()','release retains the axis and editable angle for three seconds')
    point=client.js('pickTestPoint("badge")')
    pointer(client,'mousePressed',point,clickCount=1)
    check(client,'urdfWorkbench.viewer.robot.joints.left_joint.children.some(o=>o.userData.axisKind==="jointAxes"&&o.visible&&o.userData.dragActive)','dragging a fixed attachment shows the ancestor joint actually being manipulated')
    client.js('window.dispatchEvent(new Event("blur"))')
    pointer(client,'mouseReleased',point)
    check(client,'(()=>{let active=false;urdfWorkbench.viewer.robot.traverse(o=>{if(o.userData.axisKind==="jointAxes"&&o.visible)active=true});return active&&!document.getElementById("joint-angle-tooltip").hidden})()','focus loss retains the recent joint axis and angle briefly')
    client.js('document.querySelector("[data-display=jointAxes]").click()')
    point=client.js('pickTestPoint("right_leg")')
    pointer(client,'mousePressed',point,clickCount=1)
    pointer(client,'mouseReleased',point)
    check(client,'(()=>{const axes=[];urdfWorkbench.viewer.robot.traverse(o=>{if(o.userData.axisKind==="jointAxes")axes.push(o)});return urdfWorkbench.display.jointAxes&&axes.every(o=>o.visible)})()','release preserves explicitly enabled global joint axes')
    client.js('document.querySelector("[data-display=jointAxes]").click()')
    client.call('Emulation.setDeviceMetricsOverride',{'width':390,'height':844,'deviceScaleFactor':1,'mobile':False})
    rendered_frame(client)
    client.js('document.getElementById("fit-view").click()')
    rendered_frame(client)
    double_click(client,'left_leg')
    check(client,'document.getElementById("joints-panel").classList.contains("mobile-active")&&document.querySelector(`#joint-list li[data-name="left_joint"]`).getAnimations().length>0','mobile navigation first shows the joint card flashing')
    client.until('document.getElementById("editor-panel").classList.contains("mobile-active")&&document.getElementById("source-locate-highlight").dataset.link==="left_leg"&&!document.getElementById("source-locate-highlight").hidden')
    check(client,'document.getElementById("source-locate-highlight").getAnimations().some(a=>a.effect.getTiming().iterations===2)','mobile navigation then shows the editor block flashing twice')
    client.call('Emulation.setDeviceMetricsOverride',{'width':1866,'height':961,'deviceScaleFactor':1,'mobile':False})
    rendered_frame(client)
