"""All-joint two-line labels, hip arrow tips and reversible moment-arm inspection."""
import json
from model_interaction import pointer, rendered_frame
from wheel_contacts import FIXTURE as WHEEL_FIXTURE

LABELS_OK=r'''(()=>{const T=parcelRequire('../node_modules/three/build/three.module.js'),v=urdfWorkbench.viewer;v.world.updateWorldMatrix(true,true);v.camera.updateMatrixWorld(true);return [...document.querySelectorAll('.statics-axis-label:not([hidden])')].every(label=>{const j=v.robot.joints[label.dataset.joint],axis=j.children.find(o=>o.userData.axisKind==='jointAxes'),p=axis.localToWorld(new T.Vector3(...(/hip/i.test(j.name)?axis.userData.tipAnchor:axis.userData.labelAnchor))).project(v.camera),r=label.getBoundingClientRect(),name=label.querySelector('.statics-label-name'),value=label.querySelector('.statics-label-value');return Math.abs(r.x+r.width/2-(p.x+1)*innerWidth/2)<.2&&Math.abs(r.y+r.height/2-(1-p.y)*innerHeight/2)<.2&&r.height<=35&&r.width<=162&&name.textContent===j.name&&name.getBoundingClientRect().bottom<=value.getBoundingClientRect().top+.1&&/^-?[\d.]+ N(?:·m)?$/.test(value.textContent)})})()'''


def reload_text(client,text,name):
    client.js('document.getElementById("source").value='+json.dumps(text)+';document.getElementById("source").dispatchEvent(new Event("input"));document.getElementById("reload").click()')
    client.until('urdfWorkbench.viewer.robot?.robotName==='+json.dumps(name)+'&&!urdfWorkbench.loading')
    rendered_frame(client)


def check_statics_labels(client,output,check):
    analytic=client.js('''(()=>{const E=URDFStatics,j={name:'hip',type:'revolute',position:[-.2,1,0],axis:[0,0,-1],descendants:['leg'],limit:20},input={masses:[{link:'body',mass:10,position:[0,1,0]},{link:'leg',mass:1,position:[-.5,0,0]}],joints:[j],contacts:[{link:'leg',position:[-.5,0,-.1]},{link:'leg',position:[-.5,0,.1]},{link:'body',position:[.5,0,0]}]},r=E.analyze(input),b=E.leverBreakdown(r,'hip'),near=(x,y)=>Math.abs(x-y)<1e-6;
    const out={sum:r.valid&&near(b.components.reduce((s,c)=>s+c.effort,0),r.joints[0].rawEffort),gravity:near(b.components[0].arm.distance,.3)&&near(b.components[0].effort,2.943),grouped:b.components.length===2&&near(b.components[1].magnitude,58.86)&&near(b.components[1].arm.distance,.3)};
    input.joints=[{...j,axis:[0,.6,-.8]}];const tilted=E.leverBreakdown(E.analyze(input),'hip');out.tilted=tilted.components.every(c=>c.arm&&near(c.arm.delta.reduce((s,x,i)=>s+x*tilted.axis[i],0),0)&&near(c.arm.delta[1],0)&&near(Math.abs(c.effectiveArm),c.arm.distance*.8));
    input.joints=[{...j,axis:[0,1,0]}];const yaw=E.leverBreakdown(E.analyze(input),'hip');out.parallel=yaw.components.every(c=>!c.arm&&c.parallel&&near(c.effort,0));
    input.joints=[{...j,type:'prismatic',axis:[0,1,0]}];const slide=E.leverBreakdown(E.analyze(input),'hip');out.slide=slide.unit==='N'&&slide.components.every(c=>!c.arm)&&near(slide.components.reduce((s,c)=>s+c.effort,0),slide.rawEffort);return out;})()''')
    assert all(analytic.values()),analytic
    print('PASS lever dimensions, force grouping, torque contributions, inclined/parallel axes and prismatic forces match analytical geometry',flush=True)
    reload_text(client,WHEEL_FIXTURE,'wheel_contact_fixture')
    client.js('if(!urdfWorkbench.statics.enabled)document.getElementById("statics-toggle").click();(()=>{const v=urdfWorkbench.viewer;v.controls.target.set(0,.2,0);v.camera.position.set(0,.2,3);v.controls.update();v.redraw()})()');rendered_frame(client)
    check(client,'urdfWorkbench.statics.valid&&urdfWorkbench.statics.visibleLabels.length===4','all four wheel labels display even when two joints are behind other wheels')
    check(client,LABELS_OK,'compact two-line labels show joint names and numeric loads at axis midpoints')
    client.js('(()=>{const v2=urdfWorkbench.viewer;v2.camera.position.set(0,.2,-3);v2.controls.update();v2.redraw()})()');rendered_frame(client)
    check(client,'urdfWorkbench.statics.visibleLabels.length===4','orbiting to the rear preserves all four labels without overlap filtering')
    client.screenshot(str(output/'statics-all-wheel-labels.png'))
    client.js('document.getElementById("example-select").value="urdf/go2/urdf/go2.urdf";document.getElementById("load-example").click()')
    client.until('urdfWorkbench.viewer.robot?.robotName==="go2_description"&&!urdfWorkbench.loading',55);rendered_frame(client)
    check(client,'urdfWorkbench.statics.valid&&urdfWorkbench.statics.visibleLabels.length===12&&urdfWorkbench.statics.visibleLabels.filter(n=>n.includes("hip")).length===4','Go2 overview shows all twelve joints including all four hip joints')
    check(client,LABELS_OK,'hip labels follow the exact arrow tips; other two-line labels stay at the axis midpoints')
    client.screenshot(str(output/'statics-all-joint-labels.png'))
    # Camera-only projection must not raycast meshes or measure every label rectangle.
    pointer(client,'mouseMoved',{'x':24,'y':200})
    rendered_frame(client)
    client.js('''(()=>{const T=parcelRequire('../node_modules/three/build/three.module.js');window.projectionProbe={rays:0,rects:0,raycast:T.Mesh.prototype.raycast,rect:Element.prototype.getBoundingClientRect};T.Mesh.prototype.raycast=function(...args){projectionProbe.rays++;return projectionProbe.raycast.apply(this,args)};Element.prototype.getBoundingClientRect=function(...args){if(this.classList.contains('statics-axis-label'))projectionProbe.rects++;return projectionProbe.rect.apply(this,args)};const v=urdfWorkbench.viewer;v.camera.position.sub(v.controls.target).applyAxisAngle(new T.Vector3(0,1,0),.1).add(v.controls.target);v.controls.update();v.redraw()})()''')
    rendered_frame(client)
    probe=client.js('''(()=>{const T=parcelRequire('../node_modules/three/build/three.module.js'),p=projectionProbe;T.Mesh.prototype.raycast=p.raycast;Element.prototype.getBoundingClientRect=p.rect;delete window.projectionProbe;return {rays:p.rays,rects:p.rects}})()''')
    assert probe=={'rays':0,'rects':0},probe
    print('PASS camera-only label projection performs no mesh raycasts or per-label layout reads',flush=True)
    name='FR_hip_joint'
    point=client.js('(()=>{const r=[...document.querySelectorAll(".statics-axis-label")].find(l=>l.dataset.joint==='+json.dumps(name)+').getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()')
    for kind in ['mousePressed','mouseReleased']:pointer(client,kind,point,clickCount=1)
    rendered_frame(client)
    check(client,'urdfWorkbench.statics.inspection==='+json.dumps(name)+'&&urdfWorkbench.statics.visibleLabels.length===1&&urdfWorkbench.statics.visibleLabels[0]==='+json.dumps(name)+'&&document.querySelector(".statics-arm-diagram")?.dataset.joint==='+json.dumps(name),'clicking a visible label isolates exactly that joint and reveals its lever diagram')
    check(client,'(()=>{const s=urdfWorkbench.statics,b=s.leverBreakdown,j=s.joints.find(j=>j.name===s.inspection);return b.components.length>=1&&Math.abs(b.components.reduce((sum,c)=>sum+c.effort,0)-j.rawEffort)<1e-7&&document.querySelectorAll(".statics-moment-arm").length===b.components.filter(c=>c.arm).length&&document.querySelectorAll(".statics-force-arrow").length===b.components.length&&document.querySelectorAll(".statics-dimension-label").length>=b.components.length})()','force arrows and dimensioned perpendicular arms exactly reproduce the selected raw torque')
    check(client,'[...document.querySelectorAll(".statics-force-arrow,.statics-moment-arm")].every(n=>parseFloat(getComputedStyle(n).strokeWidth)>=3)&&getComputedStyle(document.querySelector(".statics-force-component.gravity .statics-force-arrow")).stroke!==getComputedStyle(document.querySelector(".statics-force-component.support .statics-force-arrow")).stroke','gravity and support forces retain distinct colors and measurement-style line thickness')
    check(client,'[...document.querySelectorAll(".statics-moment-arm")].every(n=>Number.isFinite(Number(n.dataset.length))&&Number(n.dataset.length)>=0)&&document.getElementById("statics-joint-detail").textContent.includes("d ")','dimension values remain finite and are also explained in the parameter panel')
    client.screenshot(str(output/'statics-selected-arms.png'))
    client.js('urdfWorkbench.viewer.camera.position.sub(urdfWorkbench.viewer.controls.target).applyAxisAngle(new (parcelRequire("../node_modules/three/build/three.module.js").Vector3)(0,1,0),.25).add(urdfWorkbench.viewer.controls.target);urdfWorkbench.viewer.controls.update();urdfWorkbench.viewer.redraw()');rendered_frame(client)
    check(client,'urdfWorkbench.statics.inspection==='+json.dumps(name)+'&&document.querySelector(".statics-arm-diagram")!==null','camera orbit preserves single-joint inspection')
    check(client,LABELS_OK,'the isolated hip label continues following its arrow tip while orbiting')
    client.js('[...document.querySelectorAll(".statics-axis-label")].find(l=>l.dataset.joint==='+json.dumps(name)+').click()');rendered_frame(client)
    check(client,'urdfWorkbench.statics.inspection===null&&urdfWorkbench.statics.visibleLabels.length===12&&!document.querySelector(".statics-arm-diagram")&&document.querySelectorAll("#statics-overlay .statics-contact").length===4','clicking the same label again removes arm annotations and restores all twelve labels and contacts')
    for width,height in [(1280,800),(390,844)]:
        client.call('Emulation.setDeviceMetricsOverride',{'width':width,'height':height,'deviceScaleFactor':1,'mobile':False});rendered_frame(client)
        client.js('document.getElementById("right-statics-tab").click()');rendered_frame(client)
        check(client,LABELS_OK,f'two-line labels retain hip-tip and other midpoint anchors at {width}px')
        client.screenshot(str(output/f'statics-midpoints-{width}.png'))
    client.call('Emulation.setDeviceMetricsOverride',{'width':1866,'height':961,'deviceScaleFactor':1,'mobile':False});rendered_frame(client)
    client.js('document.getElementById("statics-toggle").click();document.getElementById("right-code-tab").click()')
