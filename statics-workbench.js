'use strict';
/* Viewer adapter for static equilibrium: collision contacts, live overlays and details. */
window.createStaticsWorkbench=function({THREE,viewer,grid,refreshScene,getModel,getLanguage,openRight,refreshAxes,project}) {
  const $=id=>document.getElementById(id),V=a=>new THREE.Vector3(...a),txt=(zh,en)=>getLanguage()==='zh'?zh:en;
  const direct=(el,tag)=>[...el.children].find(c=>c.tagName===tag),vec=s=>s?s.trim().split(/\s+/).map(Number):[0,0,0];
  const contactGeometry=new URDFStaticContacts(THREE),terrain=new URDFStaticTerrain(THREE,viewer,grid);
  const settings=['statics-gravity','statics-ground','statics-tolerance','statics-slope-angle','statics-slope-heading','statics-friction'];
  const validSettings=()=>settings.every(id=>$(id).value!==''&&$(id).checkValidity());
  const annotations=new URDFStaticsAnnotations({THREE,viewer,project,svg});
  const engine=window.URDFStatics,svgNS='http://www.w3.org/2000/svg';
  let enabled=false,model=null,result=null,frame=0,projectionFrame=0,rightView='code',selected=null,ground=null,manual=new Set();
  const labels=new Map(),rows=new Map(),axisOpacity=new WeakMap();
  let hovered=null,pinned=null,lastEmphasis;
  const labelNotice=el('div','','statics-inspection-note');
  function focusName(){return enabled&&result?.valid?(pinned||hovered):null;}
  function updateEmphasis() {
    const focus=focusName(),key=JSON.stringify([focus,pinned]);
    if(lastEmphasis===key)return;lastEmphasis=key;
    $('statics-labels').classList.toggle('is-focusing',!!focus);
    for(const [name,label]of labels){label.classList.toggle('focused',name===focus);label.setAttribute('aria-pressed',String(name===pinned));}
    for(const [name,row]of rows)row.classList.toggle('focused',name===focus);
    for(const joint of model?.joints||[]) {
      const glyph=joint.object.children.find(o=>o.userData.axisKind==='jointAxes');if(!glyph)continue;
      glyph.userData.staticsHighlighted=joint.name===focus;
      glyph.traverse(o=>{for(const material of (Array.isArray(o.material)?o.material:[o.material]).filter(Boolean)){
        if(!axisOpacity.has(material))axisOpacity.set(material,material.opacity);
        material.opacity=axisOpacity.get(material)*(focus && joint.name!==focus ? .22 : 1);
      }});
    }
    viewer.redraw();
  }
  function hoverJoint(name){if(hovered===name)return;hovered=name;updateEmphasis();requestProjection();}
  function bindFocus(node,name){node.addEventListener('pointerenter',()=>hoverJoint(name));node.addEventListener('pointerleave',()=>{if(hovered===name)hoverJoint(null);});node.addEventListener('focus',()=>hoverJoint(name));node.addEventListener('blur',()=>{if(hovered===name)hoverJoint(null);});}
  const number=(v,d=2)=>Number.isFinite(v)?(Math.abs(v)<.5*10**-d?0:v).toFixed(d):'—';
  function el(tag,text,className){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(className)node.className=className;return node;}
  function svg(tag,attrs){const node=document.createElementNS(svgNS,tag);for(const [k,v]of Object.entries(attrs))node.setAttribute(k,v);return node;}
  function setRightView(view,open=true) {
    rightView=view;$('code-workspace').hidden=view!=='code';$('statics-workspace').hidden=view!=='statics';
    document.querySelector('.editor-tools').hidden=view!=='code';
    for(const kind of ['code','statics']){const tab=$(`right-${kind}-tab`);tab.classList.toggle('active',view===kind);tab.setAttribute('aria-selected',String(view===kind));tab.tabIndex=view===kind?0:-1;}
    if(open)openRight();
  }
  function buildModel() {
    const {xml,version}=getModel(),robot=viewer.robot;
    if(!robot||!xml){model=null;return;}
    viewer.world.updateWorldMatrix(true,true);
    const masses=[],shapes=[],missing=[],invalid=[],joints=[];
    for(const node of [...xml.documentElement.children].filter(e=>e.tagName==='link')) {
      const name=node.getAttribute('name'),link=robot.links[name],inertial=direct(node,'inertial'),massTag=inertial&&direct(inertial,'mass');
      const mass=massTag?(massTag.hasAttribute('value')?Number(massTag.getAttribute('value')):NaN):null;
      if(mass===null)missing.push(name);else if(!Number.isFinite(mass)||mass<0)invalid.push(name);
      else if(mass>0)masses.push({link:name,object:link,mass,local:V(vec(direct(inertial,'origin')?.getAttribute('xyz')))});
      const collisions=link.children.filter(c=>c.isURDFCollider),visuals=link.children.filter(c=>c.isURDFVisual);
      const meshes=[];
      for(const group of collisions.length?collisions:visuals)group.traverse(mesh=>{if(mesh.isMesh&&mesh.geometry?.attributes.position){if(!mesh.geometry.boundingBox)mesh.geometry.computeBoundingBox();meshes.push(mesh);}});
      if(meshes.length){const shape={link:name,object:link,meshes,fallback:!collisions.length};contactGeometry.prepare(shape);shapes.push(shape);}
    }
    for(const node of [...xml.documentElement.children].filter(e=>e.tagName==='joint')) {
      const object=robot.joints[node.getAttribute('name')],type=node.getAttribute('type');
      if(!['revolute','continuous','prismatic'].includes(type))continue;
      const descendants=[];object.traverse(o=>{if(o.isURDFLink)descendants.push(o.name);});
      const limit=direct(node,'limit'),mimic=direct(node,'mimic');
      joints.push({name:object.name,object,type,descendants,limit:limit?.hasAttribute('effort')?Math.abs(Number(limit.getAttribute('effort'))):null,mimic:mimic?{joint:mimic.getAttribute('joint'),multiplier:Number(mimic.getAttribute('multiplier')??1)}:null});
    }
    model={version,masses,shapes,missing,invalid,joints,unsupported:[...xml.documentElement.children].filter(e=>e.tagName==='joint'&&['floating','planar'].includes(e.getAttribute('type'))).map(e=>e.getAttribute('name'))};
    lastEmphasis=undefined;ground=null;manual.clear();hovered=null;pinned=null;selected=joints[0]?.name||null;
    $('statics-contact-mode').value='auto';$('statics-contact-add').replaceChildren();
    for(const shape of shapes){const option=el('option',shape.link);option.value=shape.link;$('statics-contact-add').append(option);}
    labels.clear();rows.clear();$('statics-labels').replaceChildren(labelNotice);$('statics-joints').replaceChildren();
    for(const [index,joint]of joints.entries()) {
      const id=String(index+1).padStart(2,'0');joint.index=index;joint.labelId=id;
      const label=el('button',undefined,'statics-axis-label');label.type='button';label.dataset.joint=joint.name;
      label.append(el('span',joint.name,'statics-label-name'),el('strong','—','statics-label-value'));
      label.onclick=()=>toggleInspection(joint.name,true);bindFocus(label,joint.name);labels.set(joint.name,label);$('statics-labels').append(label);
      const row=el('button',undefined,'statics-joint-row');row.type='button';row.dataset.joint=joint.name;row.append(el('span',`${id} · ${joint.name}`),el('strong','—'),el('small',''));row.onclick=()=>toggleInspection(joint.name,false);bindFocus(row,joint.name);rows.set(joint.name,row);$('statics-joints').append(row);
    }
  }
  function collect() {
    viewer.world.updateWorldMatrix(true,true);
    const bottoms=model.shapes.map(shape=>contactGeometry.bottom(shape,terrain)).filter(b=>Number.isFinite(b.min));
    if(ground===null){ground=bottoms.length?Math.min(...bottoms.map(b=>b.min)):0;$('statics-ground').value=number(ground,6);}
    const tolerance=Number($('statics-tolerance').value)/1000,manualMode=$('statics-contact-mode').value==='manual';
    const near=bottoms.filter(b=>Math.abs(b.min-ground)<=tolerance),chosen=manualMode?near.filter(b=>manual.has(b.link)):near;
    const penetrating=bottoms.filter(b=>b.min<ground-tolerance),airborne=manualMode?[...manual].filter(name=>!near.some(b=>b.link===name)):[];
    const masses=model.masses.map(m=>({link:m.link,mass:m.mass,position:m.object.localToWorld(m.local.clone()).toArray()}));
    const joints=model.joints.map(j=>({...j,object:undefined,position:j.object.getWorldPosition(new THREE.Vector3()).toArray(),axis:j.object.axis.clone().transformDirection(j.object.matrixWorld).toArray()}));
    const contacts=chosen.flatMap(b=>b.points.map(position=>{
      let axle=viewer.robot.links[b.link]?.parent;
      while(axle&&!(axle.isURDFJoint&&['revolute','continuous'].includes(axle.jointType)))axle=axle.parent;
      return {link:b.link,position:terrain.onPlane(position[0],position[2],ground).toArray(),fallback:b.fallback,kind:b.kind,wheelJoint:b.kind==='wheelMidplane'?axle?.name:null};
    }));
    terrain.sync(ground);
    return {masses,joints,contacts,groundNormal:terrain.normal.toArray(),friction:Number($('statics-friction').value),gravity:Number($('statics-gravity').value),penetrating,airborne,bottoms};
  }
  function solve() {
    frame=0;if(!enabled)return;
    if(!validSettings()){result={valid:false,reason:'invalidSettings',joints:[],contacts:[]};render();renderProjection();return;}
    if(!model||model.version!==getModel().version)buildModel();
    if(!model){result={valid:false,reason:'noModel',joints:[],contacts:[]};render();return;}
    const input=collect();
    result=engine.analyze(input);result.airborne=input.airborne;result.penetrating=input.penetrating.map(b=>b.link);
    if(model.invalid.length){result.valid=false;result.reason='invalidMass';}
    else if(model.unsupported.length){result.valid=false;result.reason='unsupported';}
    else if(input.penetrating.length){result.valid=false;result.reason='penetration';}
    else if(input.airborne.length){result.valid=false;result.reason='airborne';}
    if(terrain.fit&&!terrain.fit.success){result.valid=false;result.reason='placementMismatch';}
    if(terrain.active&&terrain.fit?.success){
      const heights=input.bottoms.filter(b=>terrain.fit.links.includes(b.link));
      terrain.fit.currentMaxGap=Math.max(0,...heights.map(b=>Math.abs(b.min-ground)));
      if(heights.length!==terrain.fit.links.length||terrain.fit.currentMaxGap>Number($('statics-tolerance').value)/1000+1e-8){result.valid=false;result.reason='placementMismatch';}
    }
    if(!result.valid)result.joints=[];
    render();renderProjection();
  }
  function requestSolve() {if(enabled&&!frame)frame=requestAnimationFrame(solve);}
  function toggle(value=!enabled) {
    enabled=value;if(!enabled){pinned=null;hovered=null;}document.body.classList.toggle('statics-enabled',enabled);
    $('statics-toggle').classList.toggle('active',enabled);$('statics-toggle').setAttribute('aria-pressed',String(enabled));
    $('statics-labels').hidden=!enabled;$('statics-overlay').toggleAttribute('hidden',!enabled);
    updateEmphasis();refreshAxes();if(enabled){setRightView('statics');requestSolve();}else{for(const label of labels.values())label.hidden=true;render();}
  }
  const reasons={invalidSettings:['请输入有效的重力、地面、坡度、摩擦与容差参数','Enter valid gravity, ground, slope, friction and tolerance settings'],invalidTerrain:['斜坡或摩擦参数无效','Invalid slope or friction parameters'],sliding:['无法静态站稳 · 静摩擦不足，会沿坡滑动','No static equilibrium · insufficient static friction'],balanced:['可静态平衡','Static equilibrium'],marginal:['临界支撑 · 稳定裕量为零','Marginal support · zero stability margin'],placementMismatch:['无法同时贴合全部支撑端 · 请调整关节姿态后再次贴合','Cannot fit every support · adjust the stance and fit again'],outsideSupport:['无法静态平衡 · 重心超出支撑范围','No static equilibrium · CoM outside support'],noContacts:['无有效地面接触','No valid ground contacts'],noMass:['缺少有效质量数据','No valid mass data'],noModel:['请先加载模型','Load a model first'],solverFailed:['接触力求解未收敛','Contact solver did not converge'],invalidMass:['质量数据无效','Invalid mass data'],unsupported:['暂不支持浮动 / 平面关节内部载荷','Internal floating / planar joints are unsupported'],invalidMimic:['mimic 约束无效','Invalid mimic constraint'],penetration:['碰撞几何穿入地面 · 请调整姿态或地面','Geometry penetrates ground · adjust pose or ground'],airborne:['指定支撑 link 未接地','Specified support link is airborne']};
  function renderTerrain() {
    const active=terrain.active,pending=active&&(Number($('statics-slope-angle').value)!==terrain.angle||Number($('statics-slope-heading').value)!==terrain.heading);
    $('statics-place-slope').disabled=!model;$('statics-flat').disabled=!active&&!terrain.fit;
    $('statics-slope-state').textContent=(active?txt(`已放置：${number(terrain.angle,1)}° · 坡向 ${number(terrain.heading,0)}°`,`Placed: ${number(terrain.angle,1)}° · heading ${number(terrain.heading,0)}°`):txt('平地 · 坡向 0° = 场景 +X，90° = +Z','Flat · uphill 0° = scene +X, 90° = +Z'))+(pending?txt(' · 设置已更改，请再次放置',' · settings changed; place again'):'');
    const fit=terrain.fit,placement=$('statics-placement-detail');placement.replaceChildren();
    if(fit){
      if(!fit.success)placement.append(el('p',txt(fit.reason==='noSupports'?'未识别到支撑端，可在手动接触中选择足端或轮子':fit.reason==='bodyPenetration'?'贴合后其他连杆碰地，已保留原放置':'保持当前关节角时，所有支撑端无法同时贴合坡面；已保留原放置',fit.reason==='noSupports'?'No supports found; select feet or wheels in manual contacts':fit.reason==='bodyPenetration'?'Another link would intersect the plane; previous placement retained':'The fixed joint stance cannot fit all supports; previous placement retained')));
      const dl=el('dl'),signed=x=>`${x>0?'+':''}${number(x,2)}°`;
      const gap=fit.currentMaxGap??fit.maxGap;
      const values=[[txt('目标支撑端','Target supports'),`${fit.links.length} link`],[fit.success?txt('最大接触间隙','Maximum contact gap'):txt('最佳拟合高度差','Best fitting height spread'),Number.isFinite(gap)?`${number(gap*1000,2)} mm`:'—']];
      if(fit.success&&terrain.bodyAttitude){const a=terrain.bodyAttitude;values.push([txt('机身俯仰 / 横滚（水平）','Body pitch / roll · horizontal'),`${signed(a.horizontal.pitch)} / ${signed(a.horizontal.roll)}`],[txt('机身俯仰 / 横滚（坡面）','Body pitch / roll · slope'),`${signed(a.slope.pitch)} / ${signed(a.slope.roll)}`]);}
      for(const [label,value]of values)dl.append(el('dt',label),el('dd',value));placement.append(dl);
      if(fit.success&&fit.currentMaxGap>Number($('statics-tolerance').value)/1000+1e-8)placement.append(el('p',txt('支撑端位置已改变，请再次贴合后计算四端站姿','Support positions changed; fit again to analyze the full stance')));
    }
    const host=$('statics-friction-summary');host.replaceChildren();
    if(enabled&&result?.requiredFriction!==undefined){const dl=el('dl');
      for(const [label,value]of [[txt('最小静摩擦系数','Minimum static μ'),number(result.requiredFriction,3)],[txt('总法向 / 总摩擦','Normal / friction'),result.valid?`${number(result.totalNormal)} / ${number(result.totalFriction)} N`:'—'],[txt('最大抓地利用率','Peak grip use'),result.valid?`${number(result.maxFrictionUtilization*100,1)}%`:'—']])dl.append(el('dt',label),el('dd',value));
      host.append(dl);if(result.distribution==='proportionalFeasible')host.append(el('p',txt('极限接触：使用可行的比例载荷分配','Limiting contact: feasible proportional load sharing')));
    }
    const wheels=$('statics-wheel-loads');wheels.replaceChildren();
    const entries=result?.valid?result.joints.filter(j=>j.isWheel):[];
    if(!entries.length)wheels.append(el('p',txt(result?.valid?'当前支撑未识别到轮轴':'有效站姿下显示轮轴力矩',result?.valid?'No wheel axles detected among supports':'Wheel torques appear for a valid stance'),'muted'));
    for(const j of entries){const row=el('button',undefined,'statics-wheel-row');row.type='button';row.dataset.joint=j.name;row.append(el('span',j.name),el('strong',`${number(j.mimic?j.rawEffort:j.effort,3)} N·m`),el('small',`N ${number(j.wheelNormal)} · T ${number(j.wheelFriction)} N · ${txt('抓地','grip')} ${number(j.wheelUtilization*100,1)}%`));row.classList.toggle('over-limit',j.overLimit);row.onclick=()=>toggleInspection(j.name,false);wheels.append(row);}
  }
  function render() {
    $('statics-enable').textContent=enabled?txt('关闭分析','Disable'):txt('开启分析','Enable');
    $('statics-enable').classList.toggle('active',enabled);
    renderTerrain();
    const incomplete=model?.missing.length>0,status=$('statics-status');
    status.textContent=!enabled?txt('分析未开启','Analysis is off'):txt(...(reasons[result?.reason]||reasons.noModel))+(incomplete?txt(' · 质量不完整，估算',' · incomplete masses, estimate'):'');
    const overloaded=result?.joints?.filter(j=>j.overLimit).length||0;
    if(enabled&&result?.valid&&overloaded)status.textContent+=txt(` · ${overloaded} 个关节超过 effort 上限`,` · ${overloaded} joints exceed effort limits`);
    status.dataset.state=enabled&&(!result?.valid||overloaded)?'invalid':'ready';
    const summary=$('statics-summary');summary.replaceChildren();
    const max=result?.valid?Math.max(0,...result.joints.filter(j=>j.type!=='prismatic'&&!j.mimic).map(j=>Math.abs(j.effort))):null;
    for(const [label,value]of [[txt('总质量','Mass'),`${number(result?.mass)} kg`],[txt('重力','Weight'),`${number(result?.weight)} N`],[txt('支撑裕量','Support margin'),`${number(Number.isFinite(result?.margin)?result.margin*1000:NaN,1)} mm`],[txt('最大驱动力矩','Peak torque'),`${number(max)} N·m`]]){const box=el('div');box.append(el('span',label),el('strong',value));summary.append(box);}
    $('statics-contact-editor').hidden=$('statics-contact-mode').value!=='manual';
    const grouped=new Map();
    for(const c of result?.contacts||[]){if(!grouped.has(c.link))grouped.set(c.link,{count:0,force:0,normal:0,tangent:0});const group=grouped.get(c.link);group.count++;group.force+=c.force||0;group.normal+=c.normalForce||0;group.tangent+=c.frictionForce||0;}
    if($('statics-contact-mode').value==='manual')for(const name of manual)if(!grouped.has(name))grouped.set(name,{count:0,force:0});
    $('statics-contact-count').textContent=`${grouped.size} link · ${result?.contacts?.length||0} ${txt('点','points')}`;
    const contactHost=$('statics-contacts');contactHost.replaceChildren();
    for(const [name,group]of grouped){const row=el('div',undefined,'statics-contact-row'),nameNode=el('span',name);nameNode.title=name;row.append(nameNode,el('output',result?.valid?(terrain.active?`N ${number(group.normal)} · T ${number(group.tangent)} N`:`${number(group.force)} N`):txt('无解','—')));if($('statics-contact-mode').value==='manual'){const remove=el('button','×');remove.setAttribute('aria-label',txt('移除支撑 ','Remove contact ')+name);remove.onclick=()=>{manual.delete(name);requestSolve();};row.append(remove);}contactHost.append(row);}
    const values=new Map((result?.joints||[]).map(j=>[j.name,j]));
    for(const [name,row]of rows){const data=values.get(name),joint=model.joints.find(j=>j.name===name),over=data?.overLimit;
      row.querySelector('strong').textContent=enabled&&result?.valid?`${number(data?.effort)} ${data?.unit}`:'—';
      row.querySelector('small').textContent=joint.mimic?txt('mimic 约束 · 原始关节载荷','Mimic constraint · raw joint load'):over?txt('超过 URDF effort 限值','Exceeds URDF effort limit'):data&&joint.limit!==null?`${txt('URDF 上限','URDF limit')} ${number(joint.limit)} ${data.unit}`:'';
      row.classList.toggle('over-limit',!!over);row.classList.toggle('selected',selected===name);
      const label=labels.get(name);label.querySelector('.statics-label-value').textContent=`${number(joint.mimic?data?.rawEffort:data?.effort)} ${joint.type==='prismatic'?'N':'N·m'}`;label.title=`${name}${joint.mimic?' · mimic':''} · ${pinned===name?txt('再次点击恢复全部标签','Click again to restore all labels'):txt('点击查看力臂','Click to inspect moment arms')}`;label.classList.toggle('selected',selected===name);label.classList.toggle('over-limit',!!over);if(!enabled||!result?.valid)label.hidden=true;
    }
    $('statics-method').textContent=txt('准静态估算：忽略惯性、加速度与外载。平地沿用非负最小平方竖直支撑载荷；斜坡同时求解三维力、力矩平衡与库仑静摩擦锥 |Ft| ≤ μFn，以最小平方接触合力分配。极限状态采用可行的比例分配并注明。μ 是摩擦上限，不直接指定摩擦力。轮式接触取轮宽中面，轮轴假设由电机或制动器保持静止；不计滚阻、传动比与效率。多接触受力分配不是唯一的，结果不等同于实测载荷。重力沿场景 −Y，坡向 0° 为 +X、90° 为 +Z；XYZ 均为场景坐标。力矩符号遵循 URDF 关节轴，移动关节显示 N。','Quasi-static estimate: no inertia, acceleration or external loads. Flat terrain retains minimum squared vertical loads. Slopes solve full force/moment equilibrium with circular Coulomb cones |Ft| ≤ μFn, minimizing squared contact loads; limiting cases may use a reported feasible proportional distribution. μ limits friction; it does not prescribe its value. Wheel contacts lie in tread midplanes and axles are held by motors or brakes. Rolling resistance, gearing and efficiency are excluded. Load sharing is not unique and is not a measured load. Gravity is scene −Y; heading 0° is +X, 90° is +Z. Signs follow URDF axes; prismatic loads are N.');
    $('statics-mass-note').textContent=model?[model.missing.length?`${txt('未给质量，按 0 计','Missing masses, treated as zero')}: ${model.missing.join(', ')}`:txt('全部 link 已提供质量。','All links provide mass.'),result?.forceResidual!==undefined?`${txt('平衡残差','Equilibrium residual')}: ${number(result.forceResidual,6)} N / ${number(result.momentResidual,6)} N·m`:'' ].join(' '):'';
    updateEmphasis();renderDetail();
  }
  function toggleInspection(name,open) {
    if(pinned===name){pinned=null;render();requestProjection();return;}
    selectJoint(name,open);
  }
  function selectJoint(name,open) {selected=name;pinned=name;if(open)setRightView('statics');render();requestProjection();const details=$('statics-joint-details');details.open=true;details.scrollIntoView({block:'nearest'});}
  function renderDetail() {
    const host=$('statics-joint-detail');host.replaceChildren();
    const data=result?.joints?.find(j=>j.name===selected);
    if(!enabled||!result?.valid||!data){host.append(el('p',txt('有效平衡姿态下点击关节或轴旁数值查看。','Select a joint or axis label in a valid equilibrium.')));return;}
    host.append(el('strong',`${data.labelId} · ${data.name}`,'joint-detail-name'));
    const dl=el('dl'),vector=v=>v?v.map(x=>number(x,3)).join(', '):'—';
    const entries=[[txt('驱动载荷','Actuator effort'),`${number(data.mimic?data.rawEffort:data.effort,3)} ${data.unit}`],[txt('下游质量 / 重力','Downstream mass / weight'),`${number(data.mass)} kg / ${number(data.mass*result.gravity)} N`],[txt('关节位置 XYZ','Joint XYZ'),`${vector(data.position)} m`],[txt('轴方向 XYZ','Axis XYZ'),vector(data.axis)],[txt('下游质心 XYZ','Subtree CoM XYZ'),`${vector(data.subtreeCom)} m`],[txt('重力有效力臂','Gravity effective arm'),`${number(data.gravityArm,4)} m`],[txt('重力补偿项','Gravity term'),`${number(data.gravityEffort,3)} ${data.unit}`],[txt('下游支撑反力','Downstream ground load'),`${number(Math.hypot(...data.supportForce))} N`],[txt('支撑有效力臂','Ground effective arm'),`${number(data.supportArm,4)} m`],[txt('支撑补偿项','Contact term'),`${number(data.contactEffort,3)} ${data.unit}`],[txt('法向力矩项','Normal term'),`${number(data.normalEffort,3)} ${data.unit}`],[txt('摩擦力矩项','Friction term'),`${number(data.frictionEffort,3)} ${data.unit}`],[txt('原始关节载荷','Raw joint load'),`${number(data.rawEffort,3)} ${data.unit}`],[txt('截面反力 XYZ','Cut reaction XYZ'),`${vector(data.reactionForce)} N`],[txt('截面力矩 XYZ','Cut moment XYZ'),`${vector(data.reactionMoment)} N·m`]];
    for(const [label,value]of entries){dl.append(el('dt',label),el('dd',value));}host.append(dl);
    if(pinned===data.name) {
      const breakdown=engine.leverBreakdown(result,pinned),terms=el('div',undefined,'statics-arm-terms');
      terms.append(el('p',txt('力臂图：G 为重力；平地 R 为支撑合力，斜坡 N 为法向力、T 为摩擦力。仅合并同一 link 上同向力，保持力矩一致。d 为轴与力作用线的最短垂距。','Diagram: G is gravity, R flat support, N slope normal and T friction. Only parallel forces on each link are merged, preserving torque. d is the shortest distance between the axis and force line.')));
      for(const c of breakdown.components){const line=el('p');line.textContent=`${c.id}${c.link?` · ${c.link}`:''}: ${number(c.magnitude)} N${c.arm?` · d ${number(c.arm.distance*1000,1)} mm · ${txt('有效','effective')} ${number(c.effectiveArm*1000,1)} mm`:''} · ${number(c.effort,3)} ${data.unit}`;terms.append(line);}host.append(terms);
    }
  }
  function updateAnalysisAxes() {
    let changed=false;
    for(const joint of model?.joints||[]) {
      const glyph=joint.object.children.find(o=>o.userData.axisKind==='jointAxes');
      const hidden=!enabled||!result?.valid||!!pinned&&joint.name!==pinned;
      if(glyph&&glyph.userData.staticsInspectionHidden!==hidden){glyph.userData.staticsInspectionHidden=hidden;changed=true;}
    }
    if(changed)refreshAxes();
  }
  function renderProjection() {
    projectionFrame=0;const overlay=$('statics-overlay');overlay.replaceChildren();
    if(!enabled||!result?.valid){
      for(const label of labels.values())label.hidden=true;
      labelNotice.hidden=true;updateAnalysisAxes();return;
    }
    annotations.begin();overlay.setAttribute('viewBox',`0 0 ${innerWidth} ${innerHeight}`);
    const visible=node=>!node.hidden&&getComputedStyle(node).display!=='none';
    const bounds={left:8,right:innerWidth-8,top:Math.max(64,document.querySelector('.toolbar').getBoundingClientRect().bottom+8),bottom:innerHeight-38};
    for(const id of ['workspace-sidebar','editor-panel']) {
      const node=$(id);if(!visible(node))continue;const r=node.getBoundingClientRect();
      if(innerWidth<=600)bounds.bottom=Math.min(bounds.bottom,r.top-8);
      else if(id==='workspace-sidebar')bounds.left=Math.max(bounds.left,r.right+6);
      else bounds.right=Math.min(bounds.right,r.left-6);
    }
    const placed=[];
    for(const joint of model.joints) {
      const label=labels.get(joint.name),anchor=project(annotations.labelPoint(joint.object));
      const inFrame=anchor.visible&&anchor.x>=bounds.left&&anchor.x<=bounds.right&&anchor.y>=bounds.top&&anchor.y<=bounds.bottom&&annotations.inScene(joint.object);
      label.hidden=!inFrame||!!pinned&&joint.name!==pinned;
      label.classList.toggle('inspecting',pinned===joint.name);
      if(label.hidden)continue;
      Object.assign(label.style,{left:`${anchor.x}px`,top:`${anchor.y}px`});
      // Overview labels are never culled by model surfaces or other labels.
      // Only the isolated diagram needs a label rectangle for its dimension captions.
      if(pinned)placed.push(label.getBoundingClientRect());
    }
    updateAnalysisAxes();updateEmphasis();
    bounds.occupied=placed;
    const breakdown=pinned?engine.leverBreakdown(result,pinned):null;
    labelNotice.hidden=!breakdown;
    if(breakdown) {
      const glyph=model.joints.find(j=>j.name===pinned)?.object.children.find(o=>o.userData.axisKind==='jointAxes');
      annotations.draw(overlay,breakdown,glyph,bounds);
      const symbol=breakdown.type==='prismatic'?'F':'τ',terms=breakdown.components.map(c=>`${symbol}_${c.id}(${number(c.effort,3)})`).join(' + ')||'0';
      let note=`${pinned} · ${symbol} = ${terms} = ${number(breakdown.rawEffort,3)} ${breakdown.unit}`;
      if(breakdown.type==='prismatic')note+=`\n${txt('移动关节按轴向力计算，无旋转力臂。','Prismatic joint: axial forces, no rotational moment arm.')}`;
      else if(breakdown.components.some(c=>c.parallel))note+=`\n${txt('轴近似平行该作用力，不绘制其几何力臂。','Axis nearly parallel to this force; its geometric arm is omitted.')}`;
      if(Math.abs(breakdown.rawEffort-breakdown.actuatorEffort)>1e-8)note+=`\n${txt('图示为原始关节载荷；驱动载荷另含 mimic 折算。','Diagram shows raw joint effort; actuator effort also includes mimic coupling.')}`;
      note+=`\n${txt('G：重力 · R：支撑 · N：法向 · T：摩擦 · d：垂直力臂 · 箭头示意方向 · 再点标签返回','G: gravity · R: support · N: normal · T: friction · d: perpendicular arm · arrows show direction · click label again to return')}`;
      labelNotice.textContent=note;Object.assign(labelNotice.style,{left:`${bounds.left+4}px`,bottom:`${innerHeight-bounds.bottom+4}px`,maxWidth:`${bounds.right-bounds.left-8}px`});
    }else{
      const surface=(x,z)=>new THREE.Vector3(x,(ground-terrain.normal.x*x-terrain.normal.z*z)/terrain.normal.y,z);
      const hull=result.hull.map(p=>project(surface(p[0],p[1])));
      if(hull.length>2&&hull.every(p=>p.visible))overlay.append(svg('polygon',{points:hull.map(p=>`${p.x},${p.y}`).join(' ')}));
      for(const c of result.contacts){const p=project(V(c.position));if(p.visible)overlay.append(svg('circle',{cx:p.x,cy:p.y,r:4,'class':'statics-contact'}));}
      const cp=project(surface(result.com[0],result.com[2]));if(cp.visible)overlay.append(svg('circle',{cx:cp.x,cy:cp.y,r:6,'class':'statics-com'}));
    }
  }
  function requestProjection(){if(enabled&&!projectionFrame)projectionFrame=requestAnimationFrame(renderProjection);}
  $('statics-toggle').onclick=()=>toggle();$('statics-enable').onclick=()=>toggle();
  $('right-statics-tab').onclick=()=>{setRightView('statics');if(!enabled)toggle(true);};$('right-code-tab').onclick=()=>setRightView('code');
  document.querySelector('.right-tabs').onkeydown=event=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(event.key)){event.preventDefault();const view=event.key==='Home'?'code':event.key==='End'?'statics':rightView==='code'?'statics':'code';$(`right-${view}-tab`).click();$(`right-${view}-tab`).focus();}};
  for(const id of settings)$(id).oninput=()=>{const input=$(id);if(input.value===''||!input.checkValidity()){requestSolve();return;}if(id==='statics-ground')ground=Number(input.value);renderTerrain();requestSolve();};
  $('statics-place-slope').onclick=()=>{
    if(!validSettings()||!model){requestSolve();return;}
    if(!enabled)toggle(true);
    const offset=terrain.place(model.shapes,contactGeometry,Number($('statics-slope-angle').value),Number($('statics-slope-heading').value),{selected:$('statics-contact-mode').value==='manual'?manual:null,tolerance:Number($('statics-tolerance').value)/1000});
    if(offset===null){requestSolve();return;}
    ground=offset;$('statics-ground').value=number(ground,6);pinned=null;hovered=null;
    refreshScene();requestSolve();
  };
  $('statics-flat').onclick=()=>{terrain.reset();ground=null;pinned=null;hovered=null;refreshScene();requestSolve();renderTerrain();};
  $('statics-ground-reset').onclick=()=>{ground=null;requestSolve();};
  $('statics-contact-mode').onchange=()=>{if($('statics-contact-mode').value==='manual'){manual=new Set((result?.contacts||[]).map(c=>c.link));}requestSolve();render();};
  $('statics-add-contact').onclick=()=>{if($('statics-contact-add').value)manual.add($('statics-contact-add').value);requestSolve();};
  window.addEventListener('keydown',event=>{if(event.key==='Escape'&&pinned&&!/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)){pinned=null;render();requestProjection();}});
  viewer.addEventListener('angle-change',requestSolve);viewer.controls.addEventListener('change',requestProjection);window.addEventListener('resize',requestProjection);
  return {get enabled(){return enabled;},get view(){return rightView;},get snapshot(){return result?JSON.parse(JSON.stringify({...result,enabled,ground,terrain:terrain.snapshot,selected,inspection:pinned,visibleLabels:[...labels].filter(([,label])=>!label.hidden).map(([name])=>name),leverBreakdown:pinned?engine.leverBreakdown(result,pinned):null,missingMasses:model?.missing||[]})):{enabled,valid:false};},requestSolve,requestProjection,setRightView,selectJoint,
    prepareMotion(){if(enabled)toggle(false);if(terrain.active){terrain.reset(false);ground=null;renderTerrain();}},
    modelChanged(){terrain.reset(false);buildModel();result=null;updateEmphasis();refreshAxes();if(enabled)requestSolve();else render();},
    syncTerrainEnvironment(){terrain.sync();},
    orientationChanged(){terrain.reset();ground=null;refreshScene();requestSolve();renderTerrain();},
    languageChanged(){render();$('statics-contact-mode').options[0].textContent=txt('自动识别','Automatic');$('statics-contact-mode').options[1].textContent=txt('手动指定 link','Manual links');}
  };
};
