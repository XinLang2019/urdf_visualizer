"""Incline statics: analytic axle torques, friction cones and reversible 3D placement."""
import json
from model_interaction import rendered_frame
from wheel_contacts import FIXTURE as WHEEL_FIXTURE


def change(client,id,value):
    client.js('document.getElementById('+json.dumps(id)+').value='+json.dumps(str(value))+';document.getElementById('+json.dumps(id)+').dispatchEvent(new Event("input"))')
    rendered_frame(client)


def check_slope_statics(client,output,check):
    numeric=client.js('''(()=>{
      const E=URDFStatics,near=(a,b,t=1e-6)=>Math.abs(a-b)<t,angle=20*Math.PI/180,c=Math.cos(angle),s=Math.sin(angle),W=98.1,r=.2,h=.6;
      const rotate=([x,y,z])=>[x*c-y*s,x*s+y*c,z],points=[[-1,0,-.5],[-1,0,.5],[1,0,-.5],[1,0,.5]];
      const input={masses:[{link:'body',mass:10,position:rotate([0,h,0])}],contacts:points.map((p,i)=>({link:'wheel'+i,wheelJoint:'axle'+i,position:rotate(p)})),joints:points.map((p,i)=>({name:'axle'+i,type:'continuous',position:rotate([p[0],r,p[2]]),axis:[0,0,1],descendants:['wheel'+i],limit:100})),groundNormal:[-s,c,0],friction:.9};
      const result=E.analyze(input),out={};
      out.balance=result.valid&&result.forceResidual<1e-6&&result.momentResidual<1e-6&&near(result.totalNormal,W*c)&&near(result.totalFriction,W*s)&&near(result.requiredFriction,Math.tan(angle));
      out.analytic=result.valid&&result.contacts.every((f,i)=>near(f.frictionForce,W*s/4)&&near(f.normalForce,(W*c-(i<2?-1:1)*h*W*s)/4))&&result.joints.every(j=>near(j.effort,-r*W*s/4)&&near(j.normalEffort,0)&&near(j.frictionEffort,j.effort));
      const limited=E.analyze({...input,friction:.4});
      out.cones=limited.valid&&limited.contacts.every(f=>f.normalForce>=0&&f.frictionForce<=.4*f.normalForce+1e-6)&&near(limited.totalFriction,W*s)&&limited.forceResidual<1e-5&&limited.momentResidual<1e-5&&limited.joints.some((j,i)=>Math.abs(j.effort-result.joints[i].effort)>.01);
      const slipping=E.analyze({...input,friction:.2}),boundary=E.analyze({...input,friction:Math.tan(angle)});
      out.sliding=!slipping.valid&&slipping.reason==='sliding'&&slipping.joints.length===0;
      out.boundary=boundary.valid&&boundary.contacts.every(f=>near(f.frictionForce,Math.tan(angle)*f.normalForce))&&boundary.forceResidual<1e-6&&boundary.momentResidual<1e-6;
      const offset=[13,7,-4],move=p=>p.map((v,i)=>v+offset[i]);
      const translated=E.analyze({...input,masses:input.masses.map(m=>({...m,position:move(m.position)})),contacts:input.contacts.map(f=>({...f,position:move(f.position)})),joints:input.joints.map(j=>({...j,position:move(j.position)}))});
      out.translation=translated.valid&&translated.joints.every((j,i)=>near(j.effort,result.joints[i].effort));
      const yaw=([x,y,z])=>[z,y,-x],turned=E.analyze({...input,groundNormal:yaw(input.groundNormal),masses:input.masses.map(m=>({...m,position:yaw(m.position)})),contacts:input.contacts.map(f=>({...f,position:yaw(f.position)})),joints:input.joints.map(j=>({...j,position:yaw(j.position),axis:yaw(j.axis)}))});
      out.heading=turned.valid&&turned.joints.every((j,i)=>near(j.effort,result.joints[i].effort));
      const high=E.analyze({...input,masses:[{link:'body',mass:10,position:rotate([0,4,0])}]});out.tipping=!high.valid&&high.reason==='outsideSupport';
      out.arms=result.joints.every(j=>{const b=E.leverBreakdown(result,j.name);return near(b.components.reduce((sum,f)=>sum+f.effort,0),j.rawEffort)&&b.components.some(f=>f.kind==='friction')&&b.components.filter(f=>f.arm).every(f=>near(f.arm.delta.reduce((v,x,i)=>v+x*j.axis[i],0),0)&&near(f.arm.delta.reduce((v,x,i)=>v+x*f.force[i],0),0))});
      return out;
    })()''')
    for name,passed in numeric.items():assert passed,('slope analytical statics',name,numeric)
    print('PASS incline force balance, closed-form axle torques, active Coulomb cones, slip threshold, tipping, translations, headings and 3D moment arms',flush=True)
    client.js('document.getElementById("source").value='+json.dumps(WHEEL_FIXTURE)+';document.getElementById("source").dispatchEvent(new Event("input"));document.getElementById("reload").click()')
    client.until('urdfWorkbench.viewer.robot?.robotName==="wheel_contact_fixture"&&!urdfWorkbench.loading')
    client.js('if(!urdfWorkbench.statics.enabled)document.getElementById("statics-toggle").click();urdfWorkbench.viewer.setJointValue("wheel_0_joint",.4);window.flatAngles=Object.fromEntries(Object.entries(urdfWorkbench.viewer.robot.joints).map(([n,j])=>[n,j.angle]));urdfWorkbench.viewer.robot.updateMatrix();window.flatRoot=urdfWorkbench.viewer.robot.matrix.toArray();window.slopeSource=document.getElementById("source").value')
    change(client,'statics-slope-angle',20);change(client,'statics-slope-heading',0);change(client,'statics-friction',.9)
    client.js('document.getElementById("statics-place-slope").click()');rendered_frame(client)
    check(client,'(()=>{const s=urdfWorkbench.statics;return s.valid&&s.terrain.active&&s.terrain.angle===20&&s.joints.length===4&&s.joints.every(j=>j.isWheel&&Math.abs(Math.abs(j.effort)-14*9.81*Math.sin(20*Math.PI/180)*.2/4)<1e-5)&&s.forceResidual<1e-6&&s.momentResidual<1e-6&&document.querySelectorAll(".statics-wheel-row").length===4})()','one-click slope placement shows four wheel holding torques matching m g r sin(angle) / 4')
    check(client,'Object.entries(urdfWorkbench.viewer.robot.joints).every(([n,j])=>j.angle===flatAngles[n])&&document.getElementById("source").value===slopeSource','placement preserves every configured joint angle and leaves source XML untouched')
    check(client,'(()=>{const T=parcelRequire("../node_modules/three/build/three.module.js"),s=urdfWorkbench.statics,r=urdfWorkbench.viewer.robot,n=new T.Vector3(...s.terrain.normal);return s.contacts.length===4&&s.contacts.every(f=>{const link=r.links[f.link],local=link.worldToLocal(new T.Vector3(...f.position));return f.kind==="wheelMidplane"&&Math.abs(n.dot(new T.Vector3(...f.position))-s.ground)<1e-7&&Math.abs(local.y-(f.link==="wheel_0"||f.link==="wheel_3"?.03:-.03))<1e-7})})()','inclined contacts stay on the common plane and at each mirrored wheel tread midplane')
    client.js('urdfWorkbench.viewer.robot.updateMatrix();window.placedRoot=urdfWorkbench.viewer.robot.matrix.toArray();document.getElementById("statics-place-slope").click()');rendered_frame(client)
    check(client,'urdfWorkbench.viewer.robot.matrix.toArray().every((x,i)=>Math.abs(x-placedRoot[i])<1e-8)','repeated placement is idempotent and never accumulates pitch')
    client.screenshot(str(output/'slope-wheel-overview.png'))
    client.js('document.querySelector(".statics-wheel-row").click()');rendered_frame(client)
    check(client,'urdfWorkbench.statics.inspection==="wheel_0_joint"&&document.querySelector(".statics-force-component.normal")&&document.querySelector(".statics-force-component.friction")&&Math.abs(urdfWorkbench.statics.leverBreakdown.components.reduce((sum,f)=>sum+f.effort,0)-urdfWorkbench.statics.joints[0].rawEffort)<1e-6&&document.getElementById("statics-joint-detail").textContent.includes("摩擦力矩项")','clicking a wheel row reveals separate normal/friction forces and their exact torque contributions')
    client.screenshot(str(output/'slope-wheel-arms.png'))
    change(client,'statics-friction',.1)
    check(client,'!urdfWorkbench.statics.valid&&urdfWorkbench.statics.reason==="sliding"&&document.querySelectorAll(".statics-axis-label:not([hidden])").length===0&&document.getElementById("statics-status").textContent.includes("摩擦不足")','insufficient grip invalidates standing torques instead of showing a fabricated stable result')
    change(client,'statics-friction',.9)
    check(client,'urdfWorkbench.statics.valid','raising friction restores a valid equilibrium without re-placing the robot')
    change(client,'statics-slope-angle',-20)
    check(client,'urdfWorkbench.statics.terrain.angle===20&&document.getElementById("statics-slope-state").textContent.includes("再次放置")','editing slope settings is staged until the placement button is clicked')
    client.js('document.getElementById("statics-place-slope").click()');rendered_frame(client)
    check(client,'urdfWorkbench.statics.valid&&urdfWorkbench.statics.joints[0].effort<0&&urdfWorkbench.statics.terrain.angle===-20','reversing slope reverses the axle holding torque')
    change(client,'statics-slope-angle',10);change(client,'statics-slope-heading',90)
    client.js('document.getElementById("statics-place-slope").click()');rendered_frame(client)
    check(client,'urdfWorkbench.statics.valid&&Math.abs(urdfWorkbench.statics.terrain.normal[0])<1e-7&&urdfWorkbench.statics.terrain.normal[2]<0&&urdfWorkbench.statics.joints.every(j=>Math.abs(j.effort)<1e-6)','a cross-slope applies lateral friction without inventing wheel drive torque about the axle')
    change(client,'statics-friction','')
    check(client,'!urdfWorkbench.statics.valid&&urdfWorkbench.statics.reason==="invalidSettings"','blank friction input invalidates stale load labels')
    change(client,'statics-friction',.9)
    for width,height in [(1280,800),(390,844)]:
        client.call('Emulation.setDeviceMetricsOverride',{'width':width,'height':height,'deviceScaleFactor':1,'mobile':False});rendered_frame(client)
        check(client,'document.body.scrollWidth===innerWidth&&document.querySelectorAll(".statics-wheel-row").length===4',f'slope controls and wheel rows remain usable at {width}px')
        client.screenshot(str(output/f'slope-{width}.png'))
    client.call('Emulation.setDeviceMetricsOverride',{'width':1866,'height':961,'deviceScaleFactor':1,'mobile':False});rendered_frame(client)
    client.js('document.getElementById("statics-flat").click()');rendered_frame(client)
    check(client,'urdfWorkbench.statics.valid&&!urdfWorkbench.statics.terrain.active&&urdfWorkbench.viewer.robot.matrix.toArray().every((x,i)=>Math.abs(x-flatRoot[i])<1e-8)&&urdfWorkbench.statics.joints.every(j=>Math.abs(j.effort)<1e-6)&&Object.entries(urdfWorkbench.viewer.robot.joints).every(([n,j])=>j.angle===flatAngles[n])','restore-flat reverses only rigid placement and retains the user stance')
    client.js('document.getElementById("statics-toggle").click();document.getElementById("right-code-tab").click();document.getElementById("statics-friction").value=.7')


def check_terrain_lifecycle(client,output,check):
    client.js('document.getElementById("example-select").value="urdf/go2/urdf/go2.urdf";document.getElementById("load-example").click()')
    client.until('urdfWorkbench.viewer.robot?.robotName==="go2_description"&&!urdfWorkbench.loading',55)
    client.js('if(!urdfWorkbench.statics.enabled)document.getElementById("statics-toggle").click();urdfWorkbench.viewer.robot.updateMatrix();window.go2FlatMatrix=urdfWorkbench.viewer.robot.matrix.toArray();window.go2SlopeXML=document.getElementById("source").value')
    change(client,'statics-slope-angle',5);change(client,'statics-slope-heading',0);change(client,'statics-friction',.7)
    client.js('document.getElementById("statics-place-slope").click()');rendered_frame(client)
    check(client,'urdfWorkbench.statics.valid&&urdfWorkbench.statics.terrain.active&&urdfWorkbench.statics.contacts.length===4&&urdfWorkbench.statics.contacts.every(c=>c.kind!=="wheelMidplane"&&c.frictionForce<=.7*c.normalForce+1e-6)&&document.querySelectorAll(".statics-wheel-row").length===0&&document.getElementById("source").value===go2SlopeXML','the actual Go2 point-foot model stands on the incline without being mislabeled as a wheeled robot')
    client.screenshot(str(output/'slope-go2-point-feet.png'))
    client.js('urdfWorkbench.viewer.camera.position.x+=.1;urdfWorkbench.viewer.controls.update();urdfWorkbench.viewer._updateEnvironment();urdfWorkbench.viewer.redraw()');rendered_frame(client)
    check(client,'(()=>{const T=parcelRequire("../node_modules/three/build/three.module.js"),v=urdfWorkbench.viewer,n=new T.Vector3(...urdfWorkbench.statics.terrain.normal),pn=new T.Vector3(0,0,1).applyQuaternion(v.plane.quaternion);return pn.dot(n)>1-1e-8&&Math.abs(v.plane.position.dot(n)-urdfWorkbench.statics.ground+.001)<1e-7})()','camera and environment updates preserve the inclined shadow plane orientation and offset')
    client.js('window.previousSlopeRobot=urdfWorkbench.viewer.robot;document.getElementById("load-example").click()')
    client.until('urdfWorkbench.viewer.robot!==previousSlopeRobot&&!urdfWorkbench.loading',55);rendered_frame(client)
    check(client,'urdfWorkbench.statics.valid&&!urdfWorkbench.statics.terrain.active&&urdfWorkbench.viewer.robot.matrix.toArray().every((x,i)=>Math.abs(x-go2FlatMatrix[i])<1e-8)&&urdfWorkbench.viewer.scene.children.filter(o=>o.userData.staticsTerrain).every(o=>!o.visible)','reloading a model clears the previous robot placement and slope surface')
    client.js('document.getElementById("statics-place-slope").click()');rendered_frame(client)
    client.js('document.getElementById("up-select").value="+Y";document.getElementById("up-select").dispatchEvent(new Event("change"))');rendered_frame(client)
    check(client,'!urdfWorkbench.statics.terrain.active&&urdfWorkbench.viewer.robot.matrix.toArray().every((x,i)=>Math.abs(x-go2FlatMatrix[i])<1e-8)','changing the model up axis exits placement and restores its original local root transform')
    client.js('document.getElementById("up-select").value="+Z";document.getElementById("up-select").dispatchEvent(new Event("change"))');rendered_frame(client)
    check(client,'urdfWorkbench.statics.valid&&!urdfWorkbench.statics.terrain.active','restoring the default up axis returns Go2 to valid flat-ground analysis')
    client.js('document.getElementById("statics-toggle").click();document.getElementById("right-code-tab").click()')
