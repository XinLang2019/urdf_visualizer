'use strict';
/* Keyframe authoring adapter. Editing poses never edits the URDF source. */
window.createTrajectoryWorkbench=function({THREE:T,viewer,getLanguage,showPanel,refreshLayout,beforePose,onPose,getGround,getJointInfo,fitBounds}){
  const $=id=>document.getElementById(id),core=createTrajectoryCore(T),clone=core.clone;
  const text=(zh,en)=>getLanguage()==='zh'?zh:en,rad=Math.PI/180;
  const caches=new Map(),undo=[],redo=[];let project=null,model=null,cacheKey='',selected=null,open=false,playing=false,applying=false,time=0,clock=0,startTime=0,speed=1,recordTime=0,dirty=false,serial=1,busy=false,rpy=[0,0,0],channel='base:2';
  const path=new T.Line(new T.BufferGeometry(),new T.LineBasicMaterial({color:0x6ba3c8,transparent:true,opacity:.7,depthWrite:false}));path.raycast=()=>{};path.visible=false;viewer.world.add(path);
  const modes=()=>[['linear',text('线性 · 匀速','Linear · constant speed')],['smoothstep',text('三次缓入缓出 · 帧间停顿','Cubic easing · stop at keys')],['quintic',text('五次缓入缓出 · 零端点速度/加速度','Quintic easing · zero end v/a')],['cubic',text('保形三次 · 连续通过','Shape-preserving cubic · flow through')]];
  function el(tag,attrs={},content){const n=document.createElement(tag);for(const [k,v]of Object.entries(attrs)){if(k==='class')n.className=v;else n.setAttribute(k,v);}if(content!==undefined)n.textContent=content;return n;}
  function button(label,action,id){const n=el('button',{type:'button',...(id?{id}:{})},label);n.onclick=action;return n;}
  function message(value,error=false){$('motion-status').textContent=value;$('motion-status').classList.toggle('error',error);}
  function guarded(fn){return (...args)=>{try{return fn(...args);}catch(e){message(e.message,true);}};}
  function field(label,node){const host=el('label',{'class':'motion-field'});host.append(el('span',{},label),node);return host;}
  function number(id,value,step='any'){const n=el('input',{id,type:'number',step,value});return n;}
  function select(id,options,value){const n=el('select',{id});for(const [v,label]of options)n.append(el('option',{value:v},label));n.value=value;return n;}
  const frame=()=>project?.keyframes.find(f=>f.id===selected);
  const snapshot=()=>({project:clone(project),selected,time,recordTime,dirty});
  function checkpoint(){undo.push(snapshot());if(undo.length>60)undo.shift();redo.length=0;}
  function restore(s){pause();({project,selected,time,recordTime,dirty}=clone(s));render();drawPath();if(project.keyframes.length)seek(time);}
  function history(from,to){if(!from.length)return;to.push(snapshot());restore(from.pop());}
  function newProject(){return {schema:'urdf-viewer.motion.v1',name:(model?.name||'robot')+'-motion',model:clone(model),rotation:'rpy',fps:60,loop:false,keyframes:[]};}
  function captured(){
    if(!viewer.robot)throw Error(text('请先加载模型','Load a model first'));
    const q=core.quaternion(rpy);if(Math.abs(q.dot(viewer.robot.quaternion))<1-1e-8){const e=new T.Euler().setFromQuaternion(viewer.robot.quaternion,'ZYX');rpy=[e.x,e.y,e.z];}
    return {base:{position:viewer.robot.position.toArray(),rpy:rpy.slice()},joints:model.joints.map(j=>{const value=viewer.robot.joints[j.name].angle||0;return Math.max(j.lower??-Infinity,Math.min(j.upper??Infinity,value));})};
  }
  function apply(pose){
    if(!model||!viewer.robot)return;beforePose();applying=true;
    try{viewer.robot.position.fromArray(pose.base_position);viewer.robot.quaternion.fromArray(pose.base_quaternion);rpy=pose.base_rpy.slice();
      model.joints.forEach((j,i)=>viewer.setJointValue(j.name,Math.max(j.lower??-Infinity,Math.min(j.upper??Infinity,pose.joint_positions[i]))));
      viewer.world.updateMatrixWorld(true);onPose();syncRoot();
    }finally{applying=false;}
  }
  function seek(value){pause();if(!project)return;time=Math.max(0,Math.min(core.duration(project),Number(value)||0));if(project.keyframes.length)apply(core.evaluate(project,time));syncTime();}
  function pause(){playing=false;syncTime();}
  function play(){
    if(!project||project.keyframes.length<2)throw Error(text('请先记录至少两个关键帧','Record at least two keyframes'));
    if(playing)return pause();if(time>=core.duration(project))time=0;
    beforePose();playing=true;clock=performance.now();startTime=time;syncTime();
  }
  function tick(now){
    if(!playing||!project)return;const end=core.duration(project);let next=startTime+(now-clock)*speed/1000;
    if(next>=end){if(project.loop){next%=end;}else{next=end;playing=false;}}
    time=next;apply(core.evaluate(project,time));syncTime();
  }
  function mark(){dirty=true;render();drawPath();}
  function record(){
    if(!model)throw Error(text('请先加载模型','Load a model first'));
    if(model.unsupported.length)throw Error('Unsupported internal multi-DOF joints: '+model.unsupported.join(', '));
    if(project.keyframes.length>=2000)throw Error('Maximum 2000 keyframes');
    const t=recordTime;if(!Number.isFinite(t)||t<0||t>3600)throw Error(text('记录时间应为 0–3600 秒','Record time must be 0–3600 s'));
    if(!project.keyframes.length&&t!==0)throw Error(text('首帧时间必须为 0 秒','First frame must be at 0 s'));
    if(project.keyframes.some(f=>Math.abs(f.time-t)<1e-4))throw Error(text('该时间已有帧，请选择后更新','A frame already exists here; select it and update'));
    const pose=captured(),clipped=model.joints.some((j,i)=>Math.abs(pose.joints[i]-(viewer.robot.joints[j.name].angle||0))>1e-9);checkpoint();pause();const f={id:'k'+serial++,time:t,label:text('姿态 ','Pose ')+(project.keyframes.length+1),interpolation:$('motion-default-mode').value,...pose};
    project.keyframes.push(f);project.keyframes.sort((a,b)=>a.time-b.time);selected=f.id;time=t;recordTime=Math.min(3600,t+.5);mark();seek(t);message(clipped?text('已按 URDF 限位调整并记录姿态','Pose clamped to URDF limits and recorded'):text('已记录当前机身与全部独立关节','Recorded base pose and all independent joints'));
  }
  function update(){const f=frame();if(!f)return;const pose=captured();checkpoint();pause();Object.assign(f,pose);mark();seek(f.time);message(text('已更新所选帧','Selected frame updated'));}
  function remove(){const f=frame();if(!f)return;checkpoint();pause();project.keyframes=project.keyframes.filter(x=>x!==f);if(project.keyframes.length){const offset=project.keyframes[0].time;project.keyframes.forEach(x=>x.time-=offset);}selected=project.keyframes[0]?.id||null;time=0;recordTime=core.duration(project)+.5;mark();if(project.keyframes.length)seek(0);}
  function choose(id){selected=id;const f=frame();if(f){recordTime=f.time+.5;render();seek(f.time);}}
  function changeTime(value){const f=frame();if(!f)return;const t=Number(value);if(!Number.isFinite(t)||t<0||t>3600||(f===project.keyframes[0]&&t!==0)||project.keyframes.some(x=>x!==f&&Math.abs(x.time-t)<1e-4))throw Error(text('时间无效：首帧需为 0，其他帧不可重合','Invalid time: first frame is 0; frames cannot overlap'));
    checkpoint();f.time=t;project.keyframes.sort((a,b)=>a.time-b.time);time=t;recordTime=Math.max(recordTime,core.duration(project)+.5);mark();}
  function syncRoot(){
    if(!viewer.robot)return;const pose=captured();
    for(let i=0;i<6;i++){const n=$('motion-root-'+i);if(n&&document.activeElement!==n)n.value=String(Number((i<3?pose.base.position[i]:pose.base.rpy[i-3]/rad).toFixed(5)));}
  }
  function editRoot(){
    const values=Array.from({length:6},(_,i)=>Number($('motion-root-'+i).value));if(values.some((v,i)=>!Number.isFinite(v)||$('motion-root-'+i).value===''))throw Error(text('机身姿态需为有效数值','Enter finite base pose values'));
    pause();beforePose();viewer.robot.position.fromArray(values.slice(0,3));rpy=values.slice(3).map(v=>v*rad);viewer.robot.quaternion.copy(core.quaternion(rpy));viewer.world.updateMatrixWorld(true);onPose();message(text('当前姿态已更改，点击记录或更新帧保存','Pose changed; record or update a keyframe to keep it'));
  }
  function syncTime(){
    if(!$('motion-play'))return;$('motion-play').textContent=playing?'Ⅱ': '▶';$('motion-play').setAttribute('aria-label',playing?text('暂停','Pause'):text('播放','Play'));$('motion-play').setAttribute('aria-pressed',String(playing));
    $('motion-cursor').value=String(time);$('motion-time').textContent=time.toFixed(3)+' / '+core.duration(project||{keyframes:[]}).toFixed(3)+' s';
    $('motion-playhead').style.left=(core.duration(project)?time/core.duration(project)*100:0)+'%';
  }
  function drawPath(){
    const points=[];if(open&&project?.keyframes.length>1){const n=Math.min(240,Math.max(2,Math.ceil(core.duration(project)*30)));for(let i=0;i<=n;i++)points.push(new T.Vector3(...core.evaluate(project,core.duration(project)*i/n).base_position));}
    path.geometry.dispose();path.geometry=new T.BufferGeometry().setFromPoints(points);path.visible=points.length>1&&open;viewer.redraw();drawCurve();
  }
  function drawCurve(){
    const svg=$('motion-curve');if(!svg)return;svg.replaceChildren();if(!project?.keyframes.length)return;
    const samples=[],end=core.duration(project),[kind,indexText]=channel.split(':'),index=Number(indexText);
    for(let i=0;i<=120;i++){const p=core.evaluate(project,end*i/120);samples.push(kind==='joint'?p.joint_positions[index]||0:kind==='angle'?p.base_rpy[index]/rad:p.base_position[index]);}
    const min=Math.min(...samples),max=Math.max(...samples),range=Math.max(max-min,1e-6);
    const p=document.createElementNS('http://www.w3.org/2000/svg','path');p.setAttribute('d',samples.map((v,i)=>(i?'L':'M')+(i/120*1000).toFixed(2)+','+(38-(v-min)/range*32).toFixed(2)).join(' '));p.setAttribute('vector-effect','non-scaling-stroke');svg.append(p);
    $('motion-curve-range').textContent=min.toFixed(2)+' → '+max.toFixed(2)+(kind==='angle'?' °':kind==='joint'?(model.joints[index]?.type==='prismatic'?' m':' rad'):' m');
  }
  function fitTrajectory(){
    if(!viewer.robot)return;
    const robot=viewer.robot,position=robot.position.clone(),rotation=robot.quaternion.clone(),values=model.joints.map(j=>robot.joints[j.name].angle),box=new T.Box3();
    try{const samples=project.keyframes.length?Array.from({length:33},(_,i)=>core.evaluate(project,core.duration(project)*i/32)):[];
      if(!samples.length){viewer.world.updateMatrixWorld(true);robot.traverse(o=>{if(o.isURDFVisual)box.expandByObject(o);});}
      for(const p of samples){robot.position.fromArray(p.base_position);robot.quaternion.fromArray(p.base_quaternion);model.joints.forEach((j,i)=>robot.setJointValue(j.name,p.joint_positions[i]));viewer.world.updateMatrixWorld(true);robot.traverse(o=>{if(o.isURDFVisual)box.expandByObject(o);});}
    }finally{robot.position.copy(position);robot.quaternion.copy(rotation);model.joints.forEach((j,i)=>robot.setJointValue(j.name,values[i]));viewer.world.updateMatrixWorld(true);viewer.redraw();}
    if(!box.isEmpty())fitBounds(box);
  }
  function toggle(value=!open){open=value;if(!open)pause();document.body.classList.toggle('motion-open',open);$('motion-timeline').hidden=!open;$('animate').classList.toggle('active',open);$('animate').setAttribute('aria-pressed',String(open));if(open)showPanel('motion');else if(!$('motion-panel').hidden)showPanel('joints');refreshLayout();drawPath();if(open)fitTrajectory();}
  function download(name,contents,type){const url=URL.createObjectURL(new Blob([contents],{type})),a=el('a',{href:url,download:name});a.click();setTimeout(()=>URL.revokeObjectURL(url),30000);}
  function filename(){return (project.name||'motion').replace(/[^\p{L}\p{N}_.-]/gu,'_');}
  function saveProject(){if(!project)return;download(filename()+'.motion.json',JSON.stringify(project,null,2),'application/json');dirty=false;render();message(text('工程已导出，可再次导入编辑','Project exported; import it to continue editing'));}
  async function exportReference(format){
    if(!project||busy)return;busy=true;pause();render();const p=clone(project),baseName=(p.name||'motion').replace(/[^\p{L}\p{N}_.-]/gu,'_');message(text('正在采样轨迹…','Sampling trajectory…'));
    try{const data=await core.sample(p);if(format==='csv'){download(baseName+'.reference.csv',core.csv(data),'text/csv');download(baseName+'.metadata.json',JSON.stringify({...data,frames:undefined},null,2),'application/json');}else download(baseName+'.reference.json',JSON.stringify(data),'application/json');
      message(text('已导出 ','Exported ')+data.frames.length+text(' 帧',' samples')+(data.diagnostics.velocity_limit_exceeded.length?text('；存在 URDF 速度超限，见 diagnostics','; URDF velocity limits exceeded; see diagnostics'):''));
    }catch(e){message(e.message,true);}finally{busy=false;render();}
  }
  function importProject(value){if(!model)throw Error(text('请先加载对应模型','Load the matching model first'));const p=core.validate(value,model);p.keyframes.forEach(f=>f.id='k'+serial++);checkpoint();pause();project=p;selected=p.keyframes[0]?.id||null;time=0;recordTime=core.duration(p)+.5;dirty=false;render();drawPath();if(p.keyframes.length)seek(0);message(text('已导入工程','Project imported'));}
  function template(){
    if(!model||model.up!=='+Z')throw Error(text('翻转示意需使用 +Z 向上坐标','Flip sketches require +Z up'));
    if(project.keyframes.length&&!confirm(text('用翻转示意替换当前关键帧？可撤销。','Replace keyframes with a flip sketch? You can undo.')))return;
    const pose=captured(),direction=$('motion-template').value==='front'?1:-1;
    checkpoint();pause();project.keyframes=[0,.25,.5,.75,1].map((t,i)=>({id:'k'+serial++,time:t,label:['start','takeoff','apex','landing','end'][i],interpolation:'cubic',base:{position:pose.base.position.map((v,j)=>v+(j===2?[0,.2,.8,.2,0][i]:0)),rpy:pose.base.rpy.map((v,j)=>v+(j===1?direction*[0,.15,Math.PI,2*Math.PI-.15,2*Math.PI][i]:0))},joints:pose.joints.slice()}));project.rotation='rpy';selected=project.keyframes[0].id;recordTime=1.5;time=0;mark();seek(0);message(text('已生成运动学翻转示意，请继续编排起跳、收腿与落地关节姿态','Kinematic flip sketch created. Author takeoff, tuck, and landing joint poses next'));
  }
  function render(){
    const host=$('motion-content'),f=frame();host.replaceChildren();
    if(!model){host.append(el('p',{'class':'empty-state'},text('加载模型后开始编排轨迹','Load a model to author motion')));renderTimeline();return;}
    const name=el('input',{id:'motion-name',value:project.name,maxlength:160});name.onchange=()=>{checkpoint();project.name=name.value||'motion';dirty=true;render();};host.append(field(text('动作名称','Motion name')+(dirty?' *':''),name));
    const row=el('div',{'class':'motion-actions'});row.append(button(text('记录姿态','Record pose'),guarded(record),'motion-record'),button(text('更新帧','Update key'),guarded(update),'motion-update'),button(text('删除','Delete'),guarded(remove),'motion-delete'));host.append(row);$('motion-update').disabled=$('motion-delete').disabled=!f;
    const recordInput=number('motion-record-time',recordTime,.05);recordInput.min='0';recordInput.max='3600';recordInput.onchange=()=>recordTime=Number(recordInput.value);host.append(field(text('下次记录时间 · s','Next record time · s'),recordInput));
    const def=select('motion-default-mode',modes(),f?.interpolation||'quintic');host.append(field(text('新帧默认插值','Default interpolation'),def));
    const list=el('div',{id:'motion-keyframes','class':'motion-keyframes',role:'list', 'aria-label':text('关键帧列表','Keyframes')});
    for(const key of project.keyframes){const b=button(key.time.toFixed(3)+' s · '+key.label,()=>choose(key.id));b.dataset.frame=key.id;b.classList.toggle('selected',key.id===selected);b.setAttribute('aria-current',String(key.id===selected));list.append(b);}host.append(list);
    if(f){const section=el('section',{'class':'motion-group'});section.append(el('h3',{},text('所选关键帧','Selected keyframe')));
      const timestamp=number('motion-key-time',f.time,.05);timestamp.min='0';timestamp.max='3600';timestamp.disabled=f===project.keyframes[0];timestamp.onchange=guarded(()=>{try{changeTime(timestamp.value);}finally{timestamp.value=String(f.time);}});section.append(field(text('时间 · s','Time · s'),timestamp));
      const label=el('input',{id:'motion-key-label',value:f.label,maxlength:160});label.onchange=()=>{checkpoint();f.label=label.value;mark();};section.append(field(text('名称','Label'),label));
      const method=select('motion-key-mode',modes(),f.interpolation);method.onchange=()=>{checkpoint();f.interpolation=method.value;mark();};section.append(field(text('到下一帧的插值','Interpolation to next key'),method));host.append(section);
    }
    const base=el('section',{'class':'motion-group'});base.append(el('h3',{},text('当前机身姿态','Current base pose')),el('p',{'class':'muted'},text('URDF 模型坐标 · XYZ 米 / RPY 度','URDF model frame · XYZ m / RPY deg')));
    const fields=el('div',{'class':'motion-pose-grid'});['X','Y','Z','Roll','Pitch','Yaw'].forEach((label,i)=>{const n=number('motion-root-'+i,0,i<3?.01:5);n.onchange=guarded(editRoot);fields.append(field(label,n));});base.append(fields,el('p',{'class':'muted'},text('Pitch 可输入 ±360° / ±720°；修改后需记录或更新帧。','Enter ±360° / ±720° in Pitch; record or update the frame afterward.')));
    const rotation=select('motion-rotation',[['rpy',text('连续 RPY · 保留旋转圈数','Unwrapped RPY · retain turns')],['slerp',text('四元数 SLERP · 最短路径','Quaternion SLERP · shortest path')]],project.rotation);rotation.onchange=()=>{checkpoint();project.rotation=rotation.value;mark();};base.append(rotation);host.append(base);
    const edit=button(text('切换关节调节 →','Adjust joints →'),()=>showPanel('joints'),'motion-edit-joints');host.append(edit);
    const preset=el('div',{'class':'motion-actions'});preset.append(select('motion-template',[['front',text('前空翻示意','Front flip sketch')],['back',text('后空翻示意','Back flip sketch')]],'front'),button(text('生成','Create'),guarded(template),'motion-create-template'));host.append(preset);
    const io=el('section',{'class':'motion-group'});io.append(el('h3',{},text('工程与参考轨迹','Project & reference data')));
    const fps=number('motion-fps',project.fps,1);fps.min='1';fps.max='1000';fps.onchange=guarded(()=>{const value=Number(fps.value);if(!fps.value||!Number.isFinite(value)||value<1||value>1000)throw Error('Sample rate must be 1–1000 Hz');checkpoint();project.fps=value;dirty=true;});io.append(field(text('采样频率 · Hz','Sample rate · Hz'),fps));
    const actions=el('div',{'class':'motion-actions'});actions.append(button(text('保存工程','Save project'),saveProject,'motion-save-project'),button(text('导入工程','Import project'),()=>$('motion-import').click(),'motion-load-project'));io.append(actions);
    const exports=el('div',{'class':'motion-actions'});for(const format of ['json','csv']){const b=button(text('导出参考 ','Export reference ')+format.toUpperCase(),()=>exportReference(format),'motion-export-'+format);b.disabled=busy||project.keyframes.length<2;exports.append(b);}io.append(exports,el('p',{'class':'muted'},text('导出含时间、机身位置/四元数、关节位置、速度与加速度；按导出的独立关节名称表对齐。','Exports time, base pose/quaternion, joint positions, velocities and accelerations, with an explicit joint-name order.')));host.append(io,el('p',{'class':'motion-note'},text('运动学参考，不验证动力学、碰撞或落地可行性。播放会暂停静力学分析。','Kinematic reference; dynamics, collisions and landing feasibility are not validated. Playback pauses statics.')));
    renderTimeline();syncRoot();
  }
  function renderTimeline(){
    const host=$('motion-timeline');host.replaceChildren();
    const header=el('div',{'class':'motion-transport'});header.append(button('▶',guarded(play),'motion-play'),button('↤',()=>seek(0),'motion-rewind'),button('‹',()=>seek(time-1/(project?.fps||60)),'motion-prev'),button('›',()=>seek(time+1/(project?.fps||60)),'motion-next'),el('output',{id:'motion-time'},'0 / 0 s'));
    const speedSelect=select('motion-speed',[[.25,'0.25×'],[.5,'0.5×'],[1,'1×'],[2,'2×']],speed);speedSelect.setAttribute('aria-label',text('播放速度','Playback speed'));speedSelect.onchange=()=>{startTime=time;clock=performance.now();speed=Number(speedSelect.value);};header.append(speedSelect);
    const loop=el('input',{type:'checkbox',id:'motion-loop'});loop.checked=!!project?.loop;loop.onchange=()=>{if(project){checkpoint();project.loop=loop.checked;dirty=true;}};header.append(field(text('循环','Loop'),loop));
    const undoButton=button('↶',()=>history(undo,redo),'motion-undo'),redoButton=button('↷',()=>history(redo,undo),'motion-redo');undoButton.title=text('撤销','Undo');redoButton.title=text('重做','Redo');undoButton.disabled=!undo.length;redoButton.disabled=!redo.length;const fit=button('⌖',guarded(fitTrajectory),'motion-fit');fit.title=text('适应整段轨迹','Fit entire trajectory');header.append(undoButton,redoButton,fit,button('×',()=>toggle(false),'motion-close'));host.append(header);
    const graph=el('div',{'class':'motion-graph'}),svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.id='motion-curve';svg.setAttribute('viewBox','0 0 1000 44');svg.setAttribute('preserveAspectRatio','none');svg.setAttribute('aria-hidden','true');graph.append(svg,el('span',{id:'motion-playhead','aria-hidden':'true'}));
    const markers=el('div',{'class':'motion-markers'});
    for(const f of project?.keyframes||[]){const b=button('◆',()=>choose(f.id));b.classList.toggle('selected',f.id===selected);b.style.left=(core.duration(project)?f.time/core.duration(project)*100:0)+'%';b.title=f.time.toFixed(3)+' s · '+f.label;b.setAttribute('aria-label',b.title);
      b.onkeydown=guarded(e=>{if(['ArrowLeft','ArrowRight'].includes(e.key)){e.preventDefault();selected=f.id;changeTime(f.time+(e.key==='ArrowRight'?.05:-.05));}});
      b.onpointerdown=e=>{if(f===project.keyframes[0])return;pause();b.setPointerCapture(e.pointerId);const rect=markers.getBoundingClientRect(),end=core.duration(project),index=project.keyframes.indexOf(f);let moved=false;
        b.onpointermove=ev=>{const lower=project.keyframes[index-1].time+.001,upper=project.keyframes[index+1]?.time-.001||Math.max(end+1,f.time);const next=Math.max(lower,Math.min(upper,Math.round((ev.clientX-rect.left)/rect.width*end*1000)/1000));if(Math.abs(next-f.time)>.0001){if(!moved)checkpoint();f.time=next;selected=f.id;time=next;dirty=true;moved=true;b.style.left=(next/end*100)+'%';}};
        b.onpointerup=b.onpointercancel=()=>{b.onpointermove=null;b.onpointerup=b.onpointercancel=null;if(moved){recordTime=Math.max(recordTime,core.duration(project)+.5);render();drawPath();seek(time);}};
      };markers.append(b);
    }
    graph.append(markers);host.append(graph);
    const cursor=el('input',{id:'motion-cursor',type:'range',min:0,max:core.duration(project||{keyframes:[]})||1,step:'any',value:time,'aria-label':text('轨迹时间','Trajectory time')});cursor.oninput=()=>seek(Number(cursor.value));host.append(cursor);
    const footer=el('div',{'class':'motion-curve-tools'}),options=[['base:0','Base X · m'],['base:1','Base Y · m'],['base:2','Base Z · m'],['angle:0','Roll · °'],['angle:1','Pitch · °'],['angle:2','Yaw · °'],...(model?.joints||[]).map((j,i)=>['joint:'+i,j.name+(j.type==='prismatic'?' · m':' · rad')])];const channelSelect=select('motion-channel',options,channel);channelSelect.setAttribute('aria-label',text('曲线通道','Curve channel'));channelSelect.onchange=()=>{channel=channelSelect.value;drawCurve();};footer.append(channelSelect,el('span',{id:'motion-curve-range'}),el('span',{'class':'muted'},text('◆ 选择 / 拖动关键帧','◆ Select / drag keyframes')));host.append(footer);for(const [id,label]of [['motion-rewind',text('返回起点','Rewind')],['motion-prev',text('上一采样帧','Previous sample')],['motion-next',text('下一采样帧','Next sample')],['motion-undo',text('撤销','Undo')],['motion-redo',text('重做','Redo')],['motion-fit',text('适应整段轨迹','Fit entire trajectory')],['motion-close',text('关闭时间轴','Close timeline')]]){$(id).setAttribute('aria-label',label);$(id).title=label;}drawCurve();syncTime();
  }
  $('motion-import').onchange=async event=>{const file=event.target.files[0];event.target.value='';if(!file)return;try{if(file.size>20*1024*1024)throw Error('Project file exceeds 20 MB');importProject(JSON.parse(await file.text()));}catch(e){message(e.message,true);}};
  viewer.addEventListener('angle-change',()=>{if(!applying&&open){pause();syncRoot();}});
  window.addEventListener('beforeunload',event=>{if(dirty||[...caches.values()].some(c=>c.dirty)){event.preventDefault();event.returnValue='';}});
  return {toggle,pause,tick,seek,record:guarded(record),importProject,exportReference,saveProject,
    get isOpen(){return open;},
    get snapshot(){return project?{project:clone(project),time,playing,open,selected,dirty}:null;},
    modelChanged(){pause();if(project)caches.set(cacheKey,{project:clone(project),dirty});
      const robot=viewer.robot;if(!robot)return;const info=new Map(getJointInfo().map(j=>[j.name,j]));const joints=Object.values(robot.joints).filter(j=>['revolute','continuous','prismatic'].includes(j.jointType)&&!j.mimicJoint);
      model={name:robot.robotName||robot.name,root:robot.name,up:viewer.up,groundPlane:getGround(),unsupported:Object.values(robot.joints).filter(j=>['floating','planar'].includes(j.jointType)).map(j=>j.name),joints:joints.map(j=>({name:j.name,type:j.jointType,lower:j.jointType!=='continuous'&&Number.isFinite(j.limit?.lower)?j.limit.lower:null,upper:j.jointType!=='continuous'&&Number.isFinite(j.limit?.upper)?j.limit.upper:null,velocity:Number.isFinite(info.get(j.name)?.velocity)&&info.get(j.name).velocity>0?info.get(j.name).velocity:null}))};
      cacheKey=JSON.stringify([model.name,model.root,model.joints]);if(channel.startsWith('joint:')&&Number(channel.split(':')[1])>=joints.length)channel='base:2';const cached=caches.get(cacheKey);caches.delete(cacheKey);project=cached?{...cached.project,model:clone(model)}:newProject();dirty=!!cached?.dirty;undo.length=redo.length=0;selected=project.keyframes[0]?.id||null;time=0;recordTime=project.keyframes.length?core.duration(project)+.5:0;rpy=[0,0,0];captured();render();drawPath();
    },languageChanged(){message('');render();},
    orientationChanged(){if(model){model.up=viewer.up;model.groundPlane=getGround();project.model=clone(model);}},
    handleShortcut(event){if(!open||$('help-dialog').open)return false;const editing=/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName),inside=$('motion-panel').contains(document.activeElement)||$('motion-timeline').contains(document.activeElement);if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='s'&&inside){event.preventDefault();saveProject();return true;}if(!editing&&(event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='z'){event.preventDefault();history(event.shiftKey?redo:undo,event.shiftKey?undo:redo);return true;}if(!editing&&event.code==='Space'){event.preventDefault();guarded(play)();return true;}return false;}
  };
};
