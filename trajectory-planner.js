/* Same-origin client for the isolated Python whole-body planner. */
window.createWholeBodyPlanner=function({getLanguage,getState,loadModel,acceptPlan,pause,refresh,saveProject,loadProject,exportReference}){
  const text=(zh,en)=>getLanguage()==='zh'?zh:en,$=id=>document.getElementById(id);
  let capabilities=null,settings=null,job=null,result=null,timer=null,connecting=false,active=false,note='',error=false,used=false,cancelRequested=false,lastPlan=null;
  const stages={queued:['等待启动','Queued'],building:['构建动力学与碰撞约束','Building dynamics and collisions'],A:['阶段 A · 全身规划','Stage A · whole-body solve'],B:['阶段 B · 下蹲时序约束','Stage B · crouch constraints'],validating:['检查约束残差','Validating constraints'],complete:['规划完成','Complete'],retrying:['原生库崩溃，隔离重试','Native crash; retrying'],failed:['规划未通过','Planning failed']};
  function el(tag,attrs={},content){const n=document.createElement(tag);for(const [k,v]of Object.entries(attrs))n.setAttribute(k,v);if(content!==undefined)n.textContent=content;return n;}
  function button(id,label,action){const b=el('button',{id,type:'button'},label);b.onclick=()=>Promise.resolve().then(action).catch(e=>{note=e.message;error=true;updateStatus();});return b;}
  function field(label,input){const n=el('label',{'class':'motion-field'});n.append(el('span',{},label),input);return n;}
  async function request(path,body){
    const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),12000);
    try{const r=await fetch('/api/planner/'+path,{cache:'no-store',signal:controller.signal,...(body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{})});let data;try{data=await r.json();}catch(e){throw Error(text('请用 python3 planner_server.py 启动本地规划服务','Start the app using python3 planner_server.py'));}if(!r.ok)throw Error(data.error||'HTTP '+r.status);return data;}finally{clearTimeout(timeout);}
  }
  const matching=()=>capabilities?.model?.descriptionHash===getState().model?.descriptionHash&&getState().model?.up==='+Z';
  function redraw(){refresh();updateStatus();}
  async function connect(){
    if(connecting)return;connecting=true;note=text('正在连接本地求解器…','Connecting to the local solver…');error=false;updateStatus();
    try{capabilities=await request('capabilities');settings||={lockLandingJoints:true,...structuredClone(capabilities.defaults)};note=capabilities.available?text('Pinocchio / IPOPT 已就绪','Pinocchio / IPOPT ready'):capabilities.problem;error=!capabilities.available;
      const saved=sessionStorage.getItem('urdf.planner.job');if(saved&&!job){try{job=await request('jobs/'+saved);active=['queued','running','cancelling'].includes(job.state);poll();}catch(e){sessionStorage.removeItem('urdf.planner.job');}}
    }catch(e){capabilities=null;note=e.message;error=true;}
    finally{connecting=false;redraw();}
  }
  async function poll(){
    clearTimeout(timer);if(!job)return;
    try{
      job=await request('jobs/'+job.id);active=['queued','running','cancelling'].includes(job.state);note=job.message;error=job.state==='failed';
      if(job.state==='succeeded'&&!result){result=await request('jobs/'+job.id+'/result');used=false;if(matching()&&getState().open&&getState().mode==='planner'){await acceptPlan(result,true);used=true;}}
      updateStatus();
      if(active)timer=setTimeout(poll,1000);else redraw();
    }catch(e){note=e.message;error=true;active=false;redraw();}
  }
  async function start(){
    if(active||!matching())return;pause();note=text('提交全身轨迹规划…','Submitting whole-body plan…');error=false;result=null;job=null;used=false;cancelRequested=false;active=true;redraw();
    try{job=await request('jobs',{settings,modelHash:capabilities.model.descriptionHash});sessionStorage.setItem('urdf.planner.job',job.id);if(cancelRequested)await request('jobs/'+job.id+'/cancel',{});poll();}catch(e){active=false;note=e.message;error=true;redraw();}
  }
  function updateStatus(){
    const host=$('planner-status');if(!host)return;host.textContent=note;host.classList.toggle('error',error);
    const status=$('planner-progress-detail'),progress=$('planner-progress');
    if(job){const stage=stages[job.stage];status.textContent=(stage?text(...stage):job.stage)+text(' · 已用 ',' · elapsed ')+job.elapsed.toFixed(1)+' s'+(job.iteration!==undefined?text(' · 轮次 ',' · round ')+(job.round||1)+'/'+job.maxRounds+text(' · 迭代 ',' · iteration ')+job.iteration+'/'+job.maxIterations:'')+(job.primal!==undefined&&job.primal!==null?text(' · 原始残差 ',' · primal residual ')+job.primal.toExponential(2):'');progress.max=job.maxIterations;if(job.state==='succeeded')progress.value=job.maxIterations;else if(job.iteration!==undefined)progress.value=job.iteration;else progress.removeAttribute('value');}
    else{status.textContent='';progress.removeAttribute('value');}
    progress.hidden=!active&&job?.state!=='succeeded';
    const start=$('planner-start');if(start)start.disabled=active||!capabilities?.available||!matching();
    const cancel=$('planner-cancel');if(cancel)cancel.disabled=!active;
    const use=$('planner-use-result');if(use)use.disabled=!result||!matching();
  }
  function render(host){
    const loadedPlan=getState().plan;if(loadedPlan&&loadedPlan!==lastPlan){settings={lockLandingJoints:true,...structuredClone(loadedPlan.settings)};lastPlan=loadedPlan;}
    const intro=el('p',{'class':'muted'},text('全身动力学 + 接触摩擦 + 碰撞约束。规划由本地 Python / IPOPT 求解，成功后自动播放。','Whole-body dynamics, contact friction and collisions. Local Python / IPOPT solve; feasible results play automatically.'));
    const actions=el('div',{'class':'motion-actions'}),connectButton=button('planner-connect',text('连接求解器','Connect solver'),connect),modelButton=button('planner-load-model',text('加载规划模型 · Go2','Load planner model · Go2'),async()=>{await loadModel();redraw();});connectButton.disabled=connecting;modelButton.disabled=!capabilities?.model;actions.append(connectButton,modelButton);host.append(intro,actions,el('p',{id:'planner-status',role:'status','class':'muted'}));
    if(capabilities?.model&&!matching())host.append(el('p',{'class':'motion-note'},text('请加载规划器对应的 Go2（含转子惯量和电机限值），并使用 +Z 向上。其他机器人暂未接入此规划器。','Load the matching Go2 with rotor inertia and motor limits, using +Z up. This planner currently supports Go2.')));
    const controls=el('fieldset',{id:'planner-settings','class':'planner-settings'});controls.disabled=active;
    if(settings){
      const motion=el('select',{id:'planner-motion'});for(const [v,zh,en]of [['backflip','后空翻','Backflip'],['frontflip','前空翻','Frontflip'],['jump','跳跃','Jump'],['stand','站立验证','Standing test']])motion.append(el('option',{value:v},text(zh,en)));motion.value=settings.motion;
      motion.onchange=()=>{settings.motion=motion.value;settings.targetX=motion.value==='backflip'?-.4:motion.value==='frontflip'?.4:0;settings.waypointPitch=motion.value==='frontflip'?180:-180;if(motion.value==='stand')settings.phases={stand:.2,crouch:0,load:0,flight:0,landing:0,settle:.2};else settings.phases=structuredClone(capabilities.defaults.phases);redraw();};controls.append(field(text('动作模板','Motion template'),motion));
      const number=(key,zh,en,step=.01,container=controls)=>{const input=el('input',{id:'planner-'+key,type:'number',step,value:settings[key]});input.onchange=()=>settings[key]=Number(input.value);container.append(field(text(zh,en),input));};
      const check=(key,zh,en,container=controls)=>{const input=el('input',{id:'planner-'+key,type:'checkbox'});input.checked=settings[key];input.onchange=()=>settings[key]=input.checked;container.append(field(text(zh,en),input));};
      const phaseGroup=el('div',{'class':'motion-pose-grid'});for(const [key,zh,en]of [['stand','初始站稳','Initial stand'],['crouch','下蹲','Crouch'],['load','蓄力','Loading'],['flight','腾空','Flight'],['landing','落地缓冲','Landing'],['settle','结束站稳','Final stand']]){const input=el('input',{id:'planner-phase-'+key,type:'number',min:0,max:2,step:.02,value:settings.phases[key]});input.onchange=()=>settings.phases[key]=Number(input.value);phaseGroup.append(field(text(zh,en)+' · s',input));}controls.append(el('h3',{},text('接触时序','Contact phases')),phaseGroup);
      const basic=el('div',{'class':'motion-pose-grid'});number('dt','节点间隔 · s','Node interval · s',.01,basic);number('friction','摩擦系数 μ','Friction μ',.05,basic);number('jointSpeed','关节速度 · rad/s','Joint speed · rad/s',1,basic);number('torqueScale','电机力矩比例','Motor torque scale',.05,basic);controls.append(el('h3',{},text('基本约束','Constraints')),basic);
      const pose=el('details',{open:''}),poseFields=el('div',{'class':'motion-pose-grid'});pose.append(el('summary',{},text('初始与落地姿态','Initial & landing pose')));
      for(const [key,zh,en,step]of [['initialHip','初始 hip · rad','Initial hip · rad',.1],['initialThigh','初始 thigh · rad','Initial thigh · rad',.1],['initialCalf','初始 calf · rad','Initial calf · rad',.1],['targetX','落点 ΔX · m','Landing ΔX · m',.1],['targetY','落点 ΔY · m','Landing ΔY · m',.1],['targetZ','落点 ΔZ · m','Landing ΔZ · m',.01],['targetRoll','落地 Roll · °','Landing Roll · °',5],['targetPitch','落地 Pitch · °','Landing Pitch · °',5],['targetYaw','落地 Yaw · °','Landing Yaw · °',5]])number(key,zh,en,step,poseFields);pose.append(poseFields);check('lockLandingJoints','落地恢复初始关节姿态（关闭后可调整腿以满足机身约束）','Restore initial joint pose on landing (disable for free landing IK)',pose);check('waypointEnabled','约束腾空中点姿态','Constrain mid-flight attitude',pose);
      const waypoints=el('div',{'class':'motion-pose-grid'});for(const [key,zh,en,step]of [['waypointRoll','中点 Roll · °','Midpoint Roll · °',5],['waypointPitch','中点 Pitch · °','Midpoint Pitch · °',5],['waypointYaw','中点 Yaw · °','Midpoint Yaw · °',5],['waypointTolerance','各轴容差 · °','Per-axis tolerance · °',1],['waypointHeight','中点最低高度 · m','Midpoint min height · m',.05]])number(key,zh,en,step,waypoints);pose.append(waypoints);controls.append(pose);
      const advanced=el('details'),advancedFields=el('div',{'class':'motion-pose-grid'});advanced.append(el('summary',{},text('碰撞、安全距离与求解设置','Collision margins & solver settings')));
      check('meshCollision','网格凸包自碰撞 / 地面约束','Mesh hull self / ground collision',advanced);check('symmetry','左右腿对称','Left / right symmetry',advanced);
      for(const [key,zh,en,step]of [['gravity','重力 · m/s²','Gravity · m/s²',.1],['angularSpeed','机身角速度 · rad/s','Base angular speed · rad/s',1],['bodyClearance','机身最小高度 · m','Min base height · m',.01],['legClearance','腿连杆离地 · m','Leg frame clearance · m',.01],['swingClearance','腾空足端离地 · m','Flight foot clearance · m',.01],['selfMargin','自碰撞距离 · m','Self margin · m',.001],['groundMargin','地面距离 · m','Ground margin · m',.001],['crouchThigh','深蹲 thigh 下界 · rad','Crouch thigh lower · rad',.1],['maxIterations','每轮迭代上限','Max iterations / round',1],['rounds','每阶段续算轮数','Rounds / stage',1],['tolerance','约束容差','Constraint tolerance',.001],['wallTime','每轮时限 · s','Wall time / round · s',10]])number(key,zh,en,step,advancedFields);advanced.append(advancedFields);controls.append(advanced);
    }
    host.append(controls);
    const run=el('div',{'class':'motion-actions'});run.append(button('planner-start',text('开始全身规划','Plan whole-body motion'),start),button('planner-cancel',text('停止规划','Stop planning'),async()=>{cancelRequested=true;if(job){job=await request('jobs/'+job.id+'/cancel',{});poll();}}));host.append(run,el('progress',{id:'planner-progress'}),el('p',{id:'planner-progress-detail','class':'muted'}),el('p',{'class':'motion-note'},text('迭代条表示本轮预算，不代表收敛百分比。完整空翻规划可能需要数分钟；未收敛结果不会自动播放。','The bar shows iteration budget, not convergence percentage. A flip can take minutes; unconverged results never auto-play.')));
    const results=el('div',{'class':'motion-actions'});results.append(button('planner-use-result',text('加载并播放结果','Load & play result'),async()=>{if(result){await acceptPlan(result,true);used=true;redraw();}}),button('planner-save-project',text('保存工程','Save project'),saveProject),button('planner-load-project',text('导入工程','Import project'),loadProject));const exportButton=button('planner-export',text('导出求解轨迹 JSON','Export solver trajectory JSON'),()=>exportReference('json'));exportButton.disabled=!getState().plan;results.append(exportButton);host.append(results);
    const plan=getState().plan;if(plan){const r=plan.report;host.append(el('p',{'class':'planner-result'},text('离散约束通过 · ','Discrete constraints passed · ')+plan.frames.length+text(' 节点 · 残差 ',' nodes · residual ')+r.maximumConstraintViolation.toExponential(2)+text(' · 求解 ',' · solve ')+r.solveSeconds.toFixed(1)+' s'),el('p',{'class':'motion-note'},text('逐节点播放求解结果；节点间显示插值不等同于连续动力学或实机验证。','Playback uses solver nodes; interpolation between nodes is not continuous dynamics or hardware validation.')));}
    updateStatus();if(!capabilities&&!connecting&&!note)connect();
  }
  return {render,connect,get snapshot(){return {available:capabilities?.available||false,matching:matching(),job,resultReady:!!result,used,settings};}};
};
