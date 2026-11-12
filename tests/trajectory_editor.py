"""Reference motion math, winding, editor interaction, and export round trips."""
import json
from model_interaction import rendered_frame, pointer
from motion_body import check_motion_body


def check_trajectory_editor(client,output,check):
    math_result=client.js('''(async()=>{
      const T=parcelRequire('../node_modules/three/build/three.module.js'),C=createTrajectoryCore(T),near=(a,b,e=1e-6)=>Math.abs(a-b)<e;
      const model={name:'math',root:'base',up:'+Z',joints:[{name:'hinge',type:'revolute',lower:-2,upper:2,velocity:1}]};
      const project={schema:'urdf-viewer.motion.v1',name:'math',model,rotation:'rpy',fps:10,loop:false,keyframes:[
        {id:'a',time:0,label:'a',interpolation:'quintic',base:{position:[0,0,0],rpy:[0,0,0]},joints:[-1]},
        {id:'b',time:2,label:'b',interpolation:'quintic',base:{position:[1,2,3],rpy:[0,2*Math.PI,0]},joints:[1]}]};
      const mid=C.evaluate(project,1),first=C.evaluate(project,0),last=C.evaluate(project,2),out={};
      out.flip=near(mid.base_rpy[1],Math.PI)&&Math.abs(mid.base_quaternion[1])>.999&&near(last.base_rpy[1],2*Math.PI)&&near(mid.base_angular_velocity[1],2*Math.PI*1.875/2);
      out.endpoints=first.joint_positions[0]===-1&&last.joint_positions[0]===1&&first.joint_velocities[0]===0&&last.joint_velocities[0]===0&&first.joint_accelerations[0]===0&&last.joint_accelerations[0]===0;
      out.derivatives=true;
      for(const mode of C.modes){project.keyframes[0].interpolation=mode;const t=.63,h=1e-5,a=C.evaluate(project,t-h),b=C.evaluate(project,t),c=C.evaluate(project,t+h);out.derivatives&&=near((c.joint_positions[0]-a.joint_positions[0])/(2*h),b.joint_velocities[0],1e-5)&&near((c.joint_velocities[0]-a.joint_velocities[0])/(2*h),b.joint_accelerations[0],1e-4);}
      project.keyframes[0].interpolation='quintic';project.rotation='slerp';out.slerp=Math.abs(C.evaluate(project,1).base_quaternion[3])>.999999;
      project.keyframes[1].base.rpy=[.7,1.3,2];
      for(const rotation of ['rpy','slerp']){project.rotation=rotation;const t=.7,h=1e-5,a=C.evaluate(project,t-h),b=C.evaluate(project,t),c=C.evaluate(project,t+h),dq=new T.Quaternion(...c.base_quaternion).multiply(new T.Quaternion(...a.base_quaternion).invert());const omega=[dq.x/h,dq.y/h,dq.z/h];out.derivatives&&=omega.every((v,i)=>near(v,b.base_angular_velocity[i],1e-5))&&b.base_angular_acceleration.every((v,i)=>near(v,(c.base_angular_velocity[i]-a.base_angular_velocity[i])/(2*h),1e-4));}
      project.rotation='rpy';project.keyframes[0].interpolation='cubic';project.keyframes[1].interpolation='cubic';project.keyframes[1].time=.7;project.keyframes.push({...C.clone(project.keyframes[1]),id:'c',time:2.4,joints:[.2]});
      out.shape=true;for(let t=0;t<=2.4;t+=.003){const v=C.evaluate(project,t).joint_positions[0];out.shape&&=v>=-1-1e-9&&v<=1+1e-9;}
      out.c1=near(C.evaluate(project,.7-1e-7).joint_velocities[0],C.evaluate(project,.7+1e-7).joint_velocities[0],1e-5);
      const data=await C.sample(project,7);out.samples=near(data.frames.at(-1).time,2.4)&&data.frames.length===18&&data.frames.every((f,i)=>!i||f.time>data.frames[i-1].time)&&data.metadata.quaternion_order==='xyzw'&&data.frames.every(f=>f.base_quaternion.every(Number.isFinite));
      out.csv=C.csv(data).split('\\n')[0].includes('joint_velocities.hinge');
      let invalid=0;for(const alter of [p=>p.keyframes[1].time=0,p=>p.keyframes[0].joints[0]=10,p=>p.keyframes[0].base.position[0]=NaN,p=>p.model.joints[0].name='bad',p=>p.fps=0]){const p=C.clone(project);alter(p);try{C.validate(p,model);}catch(e){invalid++;}}out.validation=invalid===5;
      return out;
    })()''')
    assert all(math_result.values()),math_result
    print('PASS analytic interpolation derivatives, nonuniform cubic continuity/limits, full-turn RPY vs SLERP, endpoint sampling, CSV and validation',flush=True)
    client.js('''window.motionInitial={position:urdfWorkbench.viewer.robot.position.toArray(),quaternion:urdfWorkbench.viewer.robot.quaternion.toArray(),joints:Object.fromEntries(Object.entries(urdfWorkbench.viewer.robot.joints).map(([n,j])=>[n,j.angle]))};document.getElementById('animate').click()''')
    check(client,'!document.getElementById("motion-panel").hidden&&!document.getElementById("motion-timeline").hidden&&!urdfWorkbench.motion.playing','Motion opens an empty keyframe editor instead of sinusoidal animation')
    client.js('document.getElementById("motion-record").click()')
    check(client,'urdfWorkbench.motion.project.keyframes.length===1&&urdfWorkbench.motion.project.keyframes[0].time===0','record captures the base and all independent joints at t=0')
    client.js('document.getElementById("motion-root-4").value="360";document.getElementById("motion-root-4").dispatchEvent(new Event("change"));document.getElementById("motion-root-2").value="1.2";document.getElementById("motion-root-2").dispatchEvent(new Event("change"));document.getElementById("motion-record").click()')
    check(client,'urdfWorkbench.motion.project.keyframes.length===2&&Math.abs(urdfWorkbench.motion.project.keyframes[1].base.rpy[1]-2*Math.PI)<1e-9','editing base Pitch to 360 degrees preserves winding when recording')
    client.js('document.getElementById("motion-cursor").value=".25";document.getElementById("motion-cursor").dispatchEvent(new Event("input"))');rendered_frame(client)
    check(client,'Math.abs(urdfWorkbench.viewer.robot.quaternion.y)>.999&&Math.abs(urdfWorkbench.viewer.robot.position.z-.6)<1e-6','scrubbing applies halfway full-flip orientation and base translation in 3D')
    client.js('document.getElementById("statics-toggle").click();document.getElementById("motion-play").click()')
    rendered_frame(client)
    check(client,'(urdfWorkbench.motion.playing||urdfWorkbench.motion.time===.5)&&!urdfWorkbench.statics.enabled','playback suspends quasi-static torque labels')
    client.js('if(urdfWorkbench.motion.playing)document.getElementById("motion-play").click();document.getElementById("motion-edit-joints").click()')
    client.js('''const slider=document.querySelector('#joint-list li[data-name="FL_hip_joint"] input[type=range]');slider.value=.2;slider.dispatchEvent(new Event('input'));document.getElementById('tab-motion').click();document.getElementById('motion-update').click()''')
    check(client,'Math.abs(urdfWorkbench.motion.project.keyframes[1].joints[urdfWorkbench.motion.project.model.joints.findIndex(j=>j.name==="FL_hip_joint")]-.2)<1e-6&&!urdfWorkbench.motion.playing','joint editing pauses playback and updates the selected key explicitly')
    client.js('document.getElementById("motion-key-time").value="1.3";document.getElementById("motion-key-time").dispatchEvent(new Event("change"))')
    check(client,'urdfWorkbench.motion.project.keyframes[1].time===1.3','keyframe retiming changes the actual trajectory duration')
    client.js('document.getElementById("motion-undo").click()')
    check(client,'urdfWorkbench.motion.project.keyframes[1].time===.5','undo restores the previous keyframe timing')
    client.js('document.getElementById("motion-redo").click()')
    check(client,'urdfWorkbench.motion.project.keyframes[1].time===1.3','redo restores the retimed trajectory')
    client.js('document.getElementById("motion-key-mode").value="cubic";document.getElementById("motion-key-mode").dispatchEvent(new Event("change"));document.getElementById("motion-channel").value="angle:1";document.getElementById("motion-channel").dispatchEvent(new Event("change"))')
    client.js('document.getElementById("right-code-tab").click();document.getElementById("motion-fit").click();document.getElementById("motion-cursor").value=".65";document.getElementById("motion-cursor").dispatchEvent(new Event("input"))');rendered_frame(client)
    client.screenshot(str(output/'motion-editor.png'))
    client.js('''window.motionDownloads=[];window.originalDownloadClick=HTMLAnchorElement.prototype.click;HTMLAnchorElement.prototype.click=function(){if(this.download){const name=this.download;fetch(this.href).then(r=>r.text()).then(text=>motionDownloads.push({name,text}));}else originalDownloadClick.call(this);};document.getElementById('motion-save-project').click()''')
    client.until('motionDownloads.length===1')
    check(client,'JSON.parse(motionDownloads[0].text).schema==="urdf-viewer.motion.v1"&&!urdfWorkbench.motion.dirty','editable project download preserves keyframes and clears its dirty flag')
    project=client.js('JSON.parse(motionDownloads[0].text)')
    client.js('window.actualSourceSave=document.getElementById("save").onclick;window.sourceSaveInvoked=0;document.getElementById("save").onclick=()=>sourceSaveInvoked++;document.getElementById("source").focus()')
    client.call('Input.dispatchKeyEvent',{'type':'keyDown','key':'s','code':'KeyS','windowsVirtualKeyCode':83,'modifiers':2})
    client.call('Input.dispatchKeyEvent',{'type':'keyUp','key':'s','code':'KeyS','windowsVirtualKeyCode':83,'modifiers':2})
    check(client,'sourceSaveInvoked===1&&motionDownloads.length===1','Ctrl+S in the source editor saves source even while the motion timeline is open')
    client.js('document.getElementById("save").onclick=actualSourceSave')
    client.js('document.getElementById("motion-export-json").click()');client.until('motionDownloads.length===2',30)
    check(client,'(()=>{const d=JSON.parse(motionDownloads[1].text);return d.schema==="urdf-viewer.reference.v1"&&d.frames.at(-1).time===1.3&&d.frames[0].joint_positions.length===12&&d.frames[0].base_angular_velocity.length===3&&d.metadata.kinematic_only})()','JSON reference export includes named joint order, base pose, derivatives and exact final sample')
    client.js('document.getElementById("motion-export-csv").click()');client.until('motionDownloads.length===4',30)
    check(client,'motionDownloads.some(d=>d.name.endsWith(".csv")&&d.text.includes("base_quaternion.w"))&&motionDownloads.some(d=>d.name.endsWith(".metadata.json")&&JSON.parse(d.text).metadata.quaternion_order==="xyzw")','CSV export includes a metadata sidecar with coordinate/unit conventions')
    transfer='''(()=>{const transfer=new DataTransfer();transfer.items.add(new File([JSON.stringify(PROJECT)],'motion.json',{type:'application/json'}));const input=document.getElementById('motion-import');input.files=transfer.files;input.dispatchEvent(new Event('change'));})()'''
    client.js(transfer.replace('PROJECT',json.dumps(project)));rendered_frame(client)
    check(client,'urdfWorkbench.motion.project.keyframes.length===2&&urdfWorkbench.motion.time===0','project import round-trips the editable motion')
    bad=json.loads(json.dumps(project));bad['model']['joints'][0]['name']='wrong_joint'
    client.js(transfer.replace('PROJECT',json.dumps(bad)));rendered_frame(client)
    check(client,'urdfWorkbench.motion.project.keyframes.length===2&&document.getElementById("motion-status").classList.contains("error")','incompatible imports retain the existing project and display an error')
    # Real timeline marker dragging and a viewport move ensure it remains usable.
    client.js('document.getElementById("motion-root-4").value="720";document.getElementById("motion-root-4").dispatchEvent(new Event("change"));document.getElementById("motion-record").click()')
    check(client,'urdfWorkbench.motion.project.keyframes.length===3','recording after import can extend the timeline')
    point=client.js('(()=>{const r=document.querySelectorAll(".motion-markers button")[1].getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()')
    before=client.js('urdfWorkbench.motion.project.keyframes[1].time')
    pointer(client,'mousePressed',point,clickCount=1)
    pointer(client,'mouseMoved',{'x':point['x']-30,'y':point['y']},buttons=1)
    pointer(client,'mouseReleased',{'x':point['x']-30,'y':point['y']},clickCount=1)
    rendered_frame(client)
    check(client,f'urdfWorkbench.motion.project.keyframes[1].time<{before}&&urdfWorkbench.motion.project.keyframes[0].time===0','real pointer dragging retimes a keyframe without moving the first key')
    client.js('document.getElementById("language").click();document.getElementById("theme").click()')
    check(client,'document.getElementById("motion-record").textContent==="Record pose"&&document.body.classList.contains("light")','motion controls follow English and light theme')
    for width,height in [(1280,800),(390,844)]:
        client.call('Emulation.setDeviceMetricsOverride',{'width':width,'height':height,'deviceScaleFactor':1,'mobile':False});rendered_frame(client)
        check(client,'(()=>{const p=document.getElementById("motion-timeline"),r=p.getBoundingClientRect();return r.left>=0&&r.right<=innerWidth&&r.top>=0&&r.bottom<=innerHeight&&p.scrollWidth<=p.clientWidth})()',f'motion timeline stays inside the {width}px viewport')
        client.screenshot(str(output/f'motion-{width}.png'))
    client.call('Emulation.setDeviceMetricsOverride',{'width':1897,'height':942,'deviceScaleFactor':1,'mobile':False})
    client.js('document.getElementById("language").click();document.getElementById("theme").click();HTMLAnchorElement.prototype.click=originalDownloadClick')
    project['keyframes']=[]
    client.js(transfer.replace('PROJECT',json.dumps(project)));rendered_frame(client)
    client.js('document.getElementById("motion-close").click();(()=>{const v=urdfWorkbench.viewer;v.robot.position.fromArray(motionInitial.position);v.robot.quaternion.fromArray(motionInitial.quaternion);for(const [n,value]of Object.entries(motionInitial.joints))v.setJointValue(n,value);v.redraw()})()')
    check_motion_models(client,check)
    check_motion_body(client,output,check)
    print('PASS keyframe editor, scrubbing, winding, joints/base poses, retiming, undo/redo, export/import, statics coordination and responsive controls',flush=True)


def check_motion_models(client,check):
    """The same editor handles a humanoid and independent continuous/prismatic DOFs."""
    transfer='''(()=>{const transfer=new DataTransfer();transfer.items.add(new File([JSON.stringify(PROJECT)],'motion.json',{type:'application/json'}));const input=document.getElementById('motion-import');input.files=transfer.files;input.dispatchEvent(new Event('change'));})()'''
    client.js('document.getElementById("example-select").value="urdf/g1/urdf/g1.urdf";document.getElementById("load-example").click()')
    client.until('urdfWorkbench.viewer.robot?.robotName!=="go2_description"&&!urdfWorkbench.loading',55)
    client.js('document.getElementById("animate").click();document.getElementById("motion-record").click();document.getElementById("motion-root-3").value="20";document.getElementById("motion-root-3").dispatchEvent(new Event("change"));document.getElementById("motion-record").click()')
    check(client,'urdfWorkbench.motion.project.model.joints.length>12&&urdfWorkbench.motion.project.keyframes.every(f=>f.joints.length===urdfWorkbench.motion.project.model.joints.length)','G1 humanoid captures every independent joint together with the base pose')
    empty=client.js('(()=>{const p=urdfWorkbench.motion.project;p.keyframes=[];return p})()')
    client.js(transfer.replace('PROJECT',json.dumps(empty)));rendered_frame(client)
    fixture='''<robot name="motion_dofs"><link name="base"><visual><geometry><box size=".2 .2 .2"/></geometry></visual></link><link name="slide"/><joint name="slider" type="prismatic"><parent link="base"/><child link="slide"/><axis xyz="0 0 1"/><limit lower="0" upper="1" effort="10" velocity="1"/></joint><link name="spin"/><joint name="spinner" type="continuous"><parent link="slide"/><child link="spin"/><axis xyz="0 1 0"/><limit effort="10" velocity="1"/></joint><link name="mimic"/><joint name="follower" type="continuous"><parent link="spin"/><child link="mimic"/><axis xyz="0 1 0"/><mimic joint="spinner" multiplier="2" offset=".1"/></joint></robot>'''
    client.js('document.getElementById("source").value='+json.dumps(fixture)+';document.getElementById("source").dispatchEvent(new Event("input"));document.getElementById("reload").click()')
    client.until('urdfWorkbench.viewer.robot?.robotName==="motion_dofs"&&!urdfWorkbench.loading')
    client.js('document.getElementById("motion-record").click();urdfWorkbench.viewer.setJointValue("slider",.4);urdfWorkbench.viewer.setJointValue("spinner",4*Math.PI);document.getElementById("motion-record").click();document.getElementById("motion-cursor").value=".25";document.getElementById("motion-cursor").dispatchEvent(new Event("input"))')
    check(client,'(()=>{const p=urdfWorkbench.motion.project,j=urdfWorkbench.viewer.robot.joints;return p.model.joints.length===2&&p.model.joints[0].type==="prismatic"&&p.model.joints[1].type==="continuous"&&Math.abs(j.slider.angle-.2)<1e-6&&Math.abs(j.spinner.angle-2*Math.PI)<1e-6&&Math.abs(j.follower.angle-(4*Math.PI+.1))<1e-6})()','prismatic meters, continuous-joint winding and mimic propagation remain correct during playback')
    check(client,'urdfWorkbench.motion.project.model.joints.every(j=>j.velocity===1)','motion metadata reads actual URDF velocity limits')
    # Front/back naming follows URDF +X forward: positive pitch points the nose down.
    client.js('window.realConfirm=window.confirm;window.confirm=()=>true;document.getElementById("motion-template").value="front";document.getElementById("motion-create-template").click();window.confirm=realConfirm')
    check(client,'urdfWorkbench.motion.project.keyframes.length===5&&urdfWorkbench.motion.project.keyframes.at(-1).base.rpy[1]>6.28','front-flip sketch retains the full forward turn in URDF coordinates')
    empty=client.js('(()=>{const p=urdfWorkbench.motion.project;p.keyframes=[];return p})()')
    client.js(transfer.replace('PROJECT',json.dumps(empty)));rendered_frame(client)
    client.js('document.getElementById("motion-close").click();document.getElementById("example-select").value="urdf/go2/urdf/go2.urdf";document.getElementById("load-example").click()')
    client.until('urdfWorkbench.viewer.robot?.robotName==="go2_description"&&!urdfWorkbench.loading',55)
    check(client,'urdfWorkbench.motion.project.keyframes.length===0&&urdfWorkbench.motion.project.model.joints.length===12','switching back restores the Go2 motion project without leaking the previous model channels')
