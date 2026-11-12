"""Constrained minimum-jerk math and real URDF support-phase optimization."""
from motion_body import GROUND, change
from model_interaction import rendered_frame


def check_trajectory_optimization(client, output, check):
    result=client.js('''(async()=>{
      const T=parcelRequire('../node_modules/three/build/three.module.js'),C=createTrajectoryCore(T),O=createTrajectoryOptimizer(T,C),near=(a,b,e=1e-5)=>Math.abs(a-b)<e;
      const model={name:'optimizer_math',root:'base',up:'+Z',joints:[{name:'hinge',type:'revolute',lower:-2,upper:2,velocity:1}]};
      const p={schema:'urdf-viewer.motion.v1',name:'optimization',model,rotation:'rpy',fps:30,loop:false,keyframes:[0,.2,.5].map((time,i)=>({id:String(i),time,label:'pose',interpolation:'quintic',base:{position:[i/3,i/4,1],rpy:[0,i*Math.PI,0]},joints:[i/2]}))};
      const before=JSON.stringify(p),solved=await O.optimize(p,{jointAcceleration:4}),r=solved.optimization.report,t=solved.keyframes[1].time,mid=C.evaluate(solved,t),a=C.evaluate(solved,t-1e-7),b=C.evaluate(solved,t+1e-7),out={};
      out.isolated=before===JSON.stringify(p);
      out.retimed=r.timeScale>1&&r.peakVelocityBounds[6]<=1.00001&&r.peakAccelerationBounds[6]<=4.00001&&r.baseSpeedBound<=1.50001&&r.angularSpeedBound<=6.00001&&r.angularAccelerationBound<=20.00001;
      out.continuous=mid.joint_velocities[0]>.1&&near(a.joint_positions[0],b.joint_positions[0])&&near(a.joint_velocities[0],b.joint_velocities[0])&&near(a.joint_accelerations[0],b.joint_accelerations[0]);
      out.turns=near(C.evaluate(solved,C.duration(solved)).base_rpy[1],2*Math.PI)&&near(mid.base_rpy[1],Math.PI);
      out.rest=near(C.evaluate(solved,0).joint_velocities[0],0)&&near(C.evaluate(solved,0).joint_accelerations[0],0)&&near(C.evaluate(solved,C.duration(solved)).joint_velocities[0],0);
      const baseline=solved.keyframes.map(f=>({time:f.time,position:O.values(f),velocity:O.values(f).map(()=>0),acceleration:O.values(f).map(()=>0)}));out.jerk=r.jerkCost<O.bounds(baseline,model,solved.optimization.settings).cost*.95;
      const dt=1e-5,s=C.evaluate(solved,t*.7),sa=C.evaluate(solved,t*.7-dt),sb=C.evaluate(solved,t*.7+dt);out.derivatives=near((sb.joint_positions[0]-sa.joint_positions[0])/(2*dt),s.joint_velocities[0])&&near((sb.joint_velocities[0]-sa.joint_velocities[0])/(2*dt),s.joint_accelerations[0]);
      const data=await C.sample(solved,17);out.export=data.metadata.optimization.kinematicOnly&&data.metadata.optimization.dynamicsValidated===false&&data.frames.at(-1).time===C.duration(solved)&&!!C.validate(solved,model).optimization;
      const bounded=C.clone(p);bounded.keyframes=[0,1,2,3].map((time,i)=>({id:String(i),time,label:'bound',interpolation:'quintic',base:{position:[0,0,1],rpy:[0,0,0]},joints:[[0,1,1,0][i]]}));bounded.model.joints[0].lower=0;bounded.model.joints[0].upper=1;
      const limited=await O.optimize(bounded);out.bounds=true;for(let t=0;t<=C.duration(limited);t+=.002){const v=C.evaluate(limited,t).joint_positions[0];out.bounds&&=v>=-1e-7&&v<=1+1e-7;}
      let failures=0;for(const job of [()=>O.optimize(p,{autoRetime:false}),()=>O.optimize(p,{jointSpeed:0}),()=>O.optimize({...p,rotation:'slerp'}),()=>O.optimize({...p,keyframes:p.keyframes.map((f,i)=>({...f,time:i*.001}))})])try{await job();}catch(e){failures++;}out.failures=failures===4;
      const stale=C.clone(solved);stale.keyframes[1].joints[0]+=.1;try{C.validate(stale,model);out.stale=false;}catch(e){out.stale=true;}
      const corrupt=C.clone(solved);corrupt.optimization.knots[0].velocity[0]=1e9;try{C.validate(corrupt,model);out.corrupt=false;}catch(e){out.corrupt=true;}
      return out;
    })()''')
    assert all(result.values()),result
    print('PASS global jerk reduction, C2 continuity, bounds, vector/spatial rate limits, full turns, analytical derivatives, retiming and invalid/stale solution checks',flush=True)
    client.js('document.getElementById("animate").click();window.optGround='+GROUND+';window.optVisiblePose={p:urdfWorkbench.viewer.robot.position.toArray(),j:Object.fromEntries(Object.entries(urdfWorkbench.viewer.robot.joints).map(([n,j])=>[n,j.angle]))};document.getElementById("motion-record").click()')
    change(client,'motion-root-2',.30)
    client.js('document.getElementById("motion-record").click();document.querySelector("#motion-keyframes button").click();document.getElementById("motion-key-phase").value="support";document.getElementById("motion-key-phase").dispatchEvent(new Event("change"));window.optSource=urdfWorkbench.motion.project;document.getElementById("motion-optimize").click()')
    client.until('!document.getElementById("motion-optimize").disabled',60)
    check(client,'!!urdfWorkbench.motion.project.optimization&&urdfWorkbench.motion.project.optimization.report.contactConstraints===4&&urdfWorkbench.motion.project.optimization.report.contactError<.002&&urdfWorkbench.motion.project.optimization.report.maximumJointCorrection>.1','Go2 support-phase optimization adjusts the legs for a crouching base while locking all four feet')
    check(client,'JSON.stringify('+GROUND+')===JSON.stringify(optGround)&&JSON.stringify(urdfWorkbench.motion.project.keyframes.map(f=>f.joints))===JSON.stringify(optSource.keyframes.map(f=>f.joints))','optimization keeps the reference ground and authored joint targets intact')
    check(client,'''(()=>{const T=parcelRequire('../node_modules/three/build/three.module.js'),C=createTrajectoryCore(T),p=urdfWorkbench.motion.project,K=createTrajectoryKinematics(T,urdfWorkbench.viewer.robot,p),targets=p.reference.contacts;let max=0;for(let i=0;i<=50;i++){const pose=C.evaluate(p,C.duration(p)*i/50);K.set([...pose.base_position,...pose.base_rpy,...pose.joint_positions]);for(const t of targets)max=Math.max(max,Math.hypot(...K.point(t.link).map((v,d)=>v-t.position[d])));}return max<.002})()''','independent forward-kinematics checks confirm foot lock between collocation knots')
    change(client,'motion-opt-jointSpeed',.2)
    check(client,'!urdfWorkbench.motion.project.optimization','changing optimizer limits invalidates the previous result')
    client.js('document.getElementById("motion-optimize").click()');client.until('!document.getElementById("motion-optimize").disabled',60)
    check(client,'urdfWorkbench.motion.project.optimization.report.timeScale>1&&urdfWorkbench.motion.project.optimization.report.peakVelocityBounds.slice(6).every(v=>v<=.20001)','automatic retiming honors user joint-speed limits on the actual contact-corrected path')
    client.js('window.optSolved=urdfWorkbench.motion.project;document.getElementById("motion-undo").click()')
    check(client,'!urdfWorkbench.motion.project.optimization','undo restores the pre-optimization timeline')
    client.js('document.getElementById("motion-redo").click()')
    check(client,'!!urdfWorkbench.motion.project.optimization&&urdfWorkbench.motion.project.optimization.report.dynamicsValidated===false','redo restores the optimized curve and its explicit kinematic status')
    client.js('''(()=>{const d=new DataTransfer();d.items.add(new File([JSON.stringify(optSolved)],'optimized.motion.json'));const i=document.getElementById('motion-import');i.files=d.files;i.dispatchEvent(new Event('change'))})()''');rendered_frame(client)
    check(client,'!!urdfWorkbench.motion.project.optimization&&!document.getElementById("motion-status").classList.contains("error")','optimized projects round-trip through the file import path with their solved curve')
    client.js('document.querySelectorAll("#motion-keyframes button")[1].click();document.getElementById("motion-key-label").value="edited";document.getElementById("motion-key-label").dispatchEvent(new Event("change"))')
    check(client,'!urdfWorkbench.motion.project.optimization','editing a source key invalidates the compiled trajectory')
    # An unreachable support must leave the source motion and visible draft alone.
    change(client,'motion-root-2',1.5)
    client.js('document.getElementById("motion-update").click();window.optFailureProject=JSON.stringify(urdfWorkbench.motion.project);window.optFailurePose={p:urdfWorkbench.viewer.robot.position.toArray(),q:urdfWorkbench.viewer.robot.quaternion.toArray(),j:Object.fromEntries(Object.entries(urdfWorkbench.viewer.robot.joints).map(([n,j])=>[n,j.angle]))};document.getElementById("motion-optimize").click()')
    client.until('!document.getElementById("motion-optimize").disabled',60)
    check(client,'document.getElementById("motion-status").classList.contains("error")&&JSON.stringify(urdfWorkbench.motion.project)===optFailureProject&&JSON.stringify({p:urdfWorkbench.viewer.robot.position.toArray(),q:urdfWorkbench.viewer.robot.quaternion.toArray(),j:Object.fromEntries(Object.entries(urdfWorkbench.viewer.robot.joints).map(([n,j])=>[n,j.angle]))})===JSON.stringify(optFailurePose)','unreachable contact optimization reports failure without changing the project or visible robot')
    client.js('document.getElementById("motion-optimize").click();document.getElementById("motion-opt-cancel").click()');client.until('!document.getElementById("motion-optimize").disabled',60)
    check(client,'!urdfWorkbench.motion.project.optimization&&JSON.stringify(urdfWorkbench.motion.project)===optFailureProject','cancelled optimization never installs a partial result')
    # Return to a valid support motion for screenshot and responsive verification.
    client.js('''(()=>{const d=new DataTransfer();d.items.add(new File([JSON.stringify(optSolved)],'optimized.motion.json'));const i=document.getElementById('motion-import');i.files=d.files;i.dispatchEvent(new Event('change'))})()''');rendered_frame(client)
    client.js('document.getElementById("motion-cursor").value=String(urdfWorkbench.motion.project.keyframes.at(-1).time);document.getElementById("motion-cursor").dispatchEvent(new Event("input"));document.getElementById("fit-view").click();document.getElementById("motion-optimize").scrollIntoView({block:"center"})')
    client.screenshot(str(output/'motion-optimizer.png'))
    for width,height in [(390,844),(1897,942)]:
        client.call('Emulation.setDeviceMetricsOverride',{'width':width,'height':height,'deviceScaleFactor':1,'mobile':False});rendered_frame(client)
        check(client,'(()=>{const p=document.getElementById("motion-panel");return p.scrollWidth<=p.clientWidth&&document.documentElement.scrollWidth<=innerWidth})()',f'optimizer controls do not overflow the {width}px viewport')
    client.js('document.getElementById("source").value+="\\n<!-- changed description invalidates the compiled trajectory -->";document.getElementById("source").dispatchEvent(new Event("input"));document.getElementById("reload").click()')
    client.until('!urdfWorkbench.loading')
    check(client,'!urdfWorkbench.motion.project.optimization&&urdfWorkbench.motion.project.keyframes.length===2&&urdfWorkbench.motion.dirty','reloading a changed URDF invalidates solved geometry constraints while preserving the editable keys')
    client.js('window.optConfirm=window.confirm;window.confirm=()=>true;document.getElementById("motion-new").click();window.confirm=optConfirm;document.getElementById("motion-default-stand").click();document.getElementById("motion-close").click()')
    print('PASS real support IK, constraint retiming, undo/redo, solution invalidation/import, cancellation, failure atomicity and responsive optimization controls',flush=True)
