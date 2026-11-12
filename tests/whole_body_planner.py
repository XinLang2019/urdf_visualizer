"""Real HTTP + IPOPT + WebGL integration. Requires the configured Go2 solver env."""
import json
import os
import sys
from pathlib import Path
from model_interaction import rendered_frame
from motion_body import GROUND,change
import browser_smoke as smoke


def run(client,url,output):
    client.call('Runtime.enable');client.call('Page.enable');client.call('Emulation.setDeviceMetricsOverride',{'width':1897,'height':942,'deviceScaleFactor':1,'mobile':False});client.call('Page.navigate',{'url':url});client.until('!!window.urdfWorkbench')
    check=smoke.check
    client.js('document.getElementById("animate").click()');client.until('!document.getElementById("planner-load-model").disabled')
    check(client,'document.getElementById("planner-start").disabled','whole-body planning is the default workspace and requires its matching model')
    client.js('document.getElementById("planner-load-model").click()');client.until('urdfWorkbench.motion?.planner?.matching&&!urdfWorkbench.loading',55)
    check(client,'!document.querySelector("#status.error")&&!!urdfWorkbench.viewer.robot.links.FL_hip_rotor&&urdfWorkbench.motion.project.model.joints.length===12','loads the solver Go2 with rotor masses and the correct independent joints')
    client.js('document.getElementById("planner-motion").value="stand";document.getElementById("planner-motion").dispatchEvent(new Event("change"))')
    change(client,'planner-dt',.1);change(client,'planner-maxIterations',80);change(client,'planner-rounds',1);change(client,'planner-wallTime',30)
    client.js('window.planningGround='+GROUND+';document.getElementById("planner-start").click()')
    client.until('!!urdfWorkbench.motion.planner.job',20)
    client.until('urdfWorkbench.motion.planner.job?.iteration>0',40)
    check(client,'!document.getElementById("planner-progress").hidden&&document.getElementById("planner-progress-detail").textContent.includes("迭代")','planning progress comes from real IPOPT callbacks while the browser remains responsive')
    client.until('["succeeded","failed"].includes(urdfWorkbench.motion.planner.job?.state)',70)
    check(client,'urdfWorkbench.motion.planner.job.state==="succeeded"&&!!urdfWorkbench.motion.project.plan','a real constrained solve produces a feasible plan and installs it directly for playback')
    check(client,'urdfWorkbench.motion.project.plan.report.maximumConstraintViolation<=urdfWorkbench.motion.project.plan.report.tolerance&&urdfWorkbench.motion.project.plan.report.discreteDynamicsFeasible','only converged plans passing independent full-NLP residual checks become playable')
    check(client,'urdfWorkbench.motion.project.plan.frames.every(f=>f.joint_torques.length===12&&Object.values(f.contact_forces).length===4)&&urdfWorkbench.motion.project.plan.frames.length===4','planner output retains nodal torques, contact forces and explicit joint order')
    client.until('!urdfWorkbench.motion.playing',10)
    client.js('document.getElementById("motion-rewind").click();document.getElementById("motion-play").click()');rendered_frame(client)
    check(client,'urdfWorkbench.motion.playing||urdfWorkbench.motion.time>0','the existing timeline plays the solver trajectory without authored keyframes')
    client.js('if(urdfWorkbench.motion.playing)document.getElementById("motion-play").click();window.completedGround='+GROUND+';document.getElementById("motion-cursor").value=".15";document.getElementById("motion-cursor").dispatchEvent(new Event("input"));urdfWorkbench.viewer._updateEnvironment()')
    check(client,'''(()=>{const T=parcelRequire('../node_modules/three/build/three.module.js'),p=urdfWorkbench.motion.project,C=createTrajectoryCore(T),s=C.evaluate(p,.15),position=new T.Vector3(...s.base_position).applyQuaternion(new T.Quaternion(...p.reference.quaternion)).add(new T.Vector3(...p.reference.origin));return position.distanceTo(urdfWorkbench.viewer.robot.position)<1e-7&&p.model.joints.every((j,i)=>Math.abs(urdfWorkbench.viewer.robot.joints[j.name].angle-s.joint_positions[i])<1e-5)})()'''+ '&&JSON.stringify('+GROUND+')===JSON.stringify(completedGround)','solver-node scrubbing keeps its reference ground fixed')
    client.js('window.planningDownloads=[];window.realPlanningDownload=HTMLAnchorElement.prototype.click;HTMLAnchorElement.prototype.click=function(){if(this.download){const name=this.download;fetch(this.href).then(r=>r.text()).then(text=>planningDownloads.push({name,text}));}else realPlanningDownload.call(this)};document.getElementById("planner-export").click()')
    client.until('planningDownloads.length===1')
    check(client,'(()=>{const d=JSON.parse(planningDownloads[0].text);return d.frames.length===4&&d.frames.every(f=>f.joint_torques.length===12)&&d.metadata.dynamics.discreteDynamicsFeasible&&!d.metadata.kinematic_only&&d.frames.at(-1).time===urdfWorkbench.motion.project.plan.frames.at(-1).time})()','reference export preserves exact solver nodes, torques and forces with discrete-only dynamics metadata')
    client.js('document.getElementById("planner-save-project").click()');client.until('planningDownloads.length===2')
    client.js('''(()=>{const d=new DataTransfer();d.items.add(new File([planningDownloads[1].text],'whole-body.motion.json'));const i=document.getElementById('motion-import');i.files=d.files;i.dispatchEvent(new Event('change'))})()''');rendered_frame(client)
    check(client,'!!urdfWorkbench.motion.project.plan&&!document.getElementById("motion-status").classList.contains("error")','whole-body planning projects save and import with the reference ground intact')
    check(client,'''(()=>{const T=parcelRequire('../node_modules/three/build/three.module.js'),C=createTrajectoryCore(T),p=urdfWorkbench.motion.project,b=JSON.parse(JSON.stringify(p));b.plan.modelHash='wrong';try{C.validate(b,p.model);return false}catch(e){}b.plan=p.plan;b.plan.frames[0].base_quaternion=[0,0,0,0];try{C.validate(b,p.model);return false}catch(e){}return true})()''','invalid model mappings and malformed plan quaternions are rejected')
    # A failed one-iteration flight solve must leave the previous feasible motion intact.
    client.js('document.getElementById("planner-motion").value="jump";document.getElementById("planner-motion").dispatchEvent(new Event("change"))');change(client,'planner-dt',.1);change(client,'planner-maxIterations',1);change(client,'planner-rounds',1)
    client.js('window.feasiblePlanBeforeFailure=JSON.stringify(urdfWorkbench.motion.project.plan);document.getElementById("planner-start").click()');client.until('urdfWorkbench.motion.planner.job?.state==="failed"',70)
    check(client,'JSON.stringify(urdfWorkbench.motion.project.plan)===feasiblePlanBeforeFailure&&!urdfWorkbench.motion.playing','an unconverged whole-body solve does not replace or auto-play a previous feasible plan')
    change(client,'planner-maxIterations',300)
    client.js('document.getElementById("planner-start").click()');client.until('urdfWorkbench.motion.planner.job?.state==="running"',20);client.js('document.getElementById("planner-cancel").click()');client.until('urdfWorkbench.motion.planner.job?.state==="cancelled"',20)
    check(client,'JSON.stringify(urdfWorkbench.motion.project.plan)===feasiblePlanBeforeFailure&&!document.getElementById("planner-start").disabled','stop planning terminates the isolated process and keeps the previous result')
    client.js('document.getElementById("fit-view").click()');client.screenshot(str(output/'whole-body-planner.png'))
    for width,height in [(390,844),(1897,942)]:
        client.call('Emulation.setDeviceMetricsOverride',{'width':width,'height':height,'deviceScaleFactor':1,'mobile':False});rendered_frame(client)
        check(client,'document.documentElement.scrollWidth<=innerWidth&&document.getElementById("motion-panel").scrollWidth<=document.getElementById("motion-panel").clientWidth',f'whole-body planning controls fit the {width}px viewport')
    client.js('HTMLAnchorElement.prototype.click=realPlanningDownload')
    assert not [e for e in client.events if e.get('method')=='Runtime.exceptionThrown']
    print('PASS real whole-body planning, progress, playback, exact nodal export, failure handling, cancellation, model validation and responsive controls',flush=True)

if __name__=='__main__':
    os.environ['URDF_TEST_PLANNER']='1';smoke.run=run;smoke.main()
