/* Workbench UI. The existing Parcel bundle supplies Three.js and URDFLoader only. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const $$ = query => [...document.querySelectorAll(query)];
  const strings = {
    visual:['视觉','Visual'],
    collision:['碰撞','Collision'],
    inertia:['惯量','Inertia'],
    com:['质心','CoM'],
    axes:['坐标轴','Frames'],
    jointAxes:['关节轴','Joint axes'],
    shadow:['阴影','Shadow'],
    lighting:['光照','Light'],
    grid:['网格','Grid'],
    statics:['受力分析','Statics'],
    staticsDetails:['静力学','Statics'],
    staticsTitle:['当前姿态 · 静力估算','Current pose · static estimate'],
    staticsStart:['开启分析','Enable analysis'],
    gravity:['重力加速度','Gravity'],
    groundHeight:['地面高度','Ground height'],
    slopeSettings:['斜坡与摩擦','Slope & friction'],
    slopeAngle:['坡度角','Slope angle'],
    slopeHeading:['坡向','Uphill heading'],
    staticFriction:['静摩擦系数 μ','Static friction μ'],
    placeSlope:['一键贴合斜坡','Fit supports to slope'],
    restoreFlat:['恢复平地','Restore flat'],
    wheelTorques:['轮轴保持力矩','Wheel holding torques'],
    wheelHoldingNote:['假设轮轴由电机或制动器保持静止；数值为轴端力矩。','Wheels are held by motors or brakes; values are axle torques.'],
    contactTolerance:['接触容差','Contact tolerance'],
    supportContacts:['支撑接触','Ground contacts'],
    jointLoads:['关节载荷','Joint loads'],
    jointStatics:['选中关节参数','Selected joint'],
    staticsAssumptions:['计算假设与数据','Assumptions & data'],
    workspace:['工作栏','Workspace'],
    files:['文件','Files'],
    joints:['关节','Joints'],
    animate:['轨迹','Motion'],
    motion:['轨迹','Motion'],
    motionTitle:['关键帧编排','Keyframe motion'],
    structure:['结构','Structure'],
    editor:['编辑','Editor'],
    help:['帮助','Help'],
    dropHint:['拖拽机器人模型文件或文件夹到页面任意位置','Drop robot files or folders anywhere'],
    orLoad:['或点击下面的按钮加载','or use the buttons below'],
    loadFile:['加载文件','Load files'],
    loadFolder:['加载文件夹','Load folder'],
    examples:['内置示例','EXAMPLE MODELS'],
    chooseModel:['选择机器人模型…','Choose a robot…'],
    load:['加载','Load'],
    radians:['弧度','Rad'],
    degrees:['角度','Deg'],
    reset:['重置','Reset'],
    limits:['限位','Limits'],
    noModel:['未加载模型','No model loaded'],
    hideFixed:['隐藏固定关节','Hide fixed joints'],
    viewHint:['点击切换视角','Click to change view'],
    structureHint:['拖动: 移动 · 滚轮: 缩放 · 右键: 显隐 · Ctrl + 点击: 测量','Drag: pan · Scroll: zoom · Right click: visibility · Ctrl + click: measure'],
    reload:['重新加载','Reload'],
    save:['保存','Save'],
    emptyFile:['空白文件','Empty'],
    ready:['就绪 · 加载模型以开始','Ready · Load a model to start'],
    release:['松开以加载机器人模型','Release to load robot model'],
    leftDrag:['左键拖动','Left drag'],
    rightDrag:['右键拖动','Right drag'],
    wheel:['滚轮','Scroll'],
    rotate:['旋转视角','Rotate'],
    pan:['平移视角','Pan'],
    zoom:['缩放视图','Zoom'],
    fit:['适应视图','Fit view'],
    measure:['两点测量','Measure'],
    surfaceMeasureBasis:['场景坐标 · B − A · Shift 选点 · Alt 暂停吸附','Scene coordinates · B − A · Shift to pick · Alt bypasses snapping'],
    snapHint:['Shift 点击选取 · Alt 暂停吸附','Shift + click to pick · Alt bypasses snapping'],
    linkMeasureAction:['连杆测量','Link measure'],
    snapLoading:['正在识别几何推荐点… · 可按 Alt 直接选表面','Finding geometric features… · Alt picks the surface directly'],
    snapFree:['表面点 · 靠近推荐点自动吸附','Surface point · Move near a feature to snap'],
    surfacePoint:['表面点','Surface point'],
    circleCenter:['端面圆心','Circular face center'],
    geometryCenter:['几何中心','Geometric center'],
    sphereCenter:['球心','Sphere center'],
    rectangleVertex:['矩形顶点','Rectangle vertex'],
    rectangleCenter:['矩形面中心','Rectangle face center'],
    boundsCenter:['包围盒中心','Bounding box center'],
    meshVertex:['网格顶点','Mesh vertex'],
    quadrant:['象限点','Quadrant point'],
    meshFitted:['网格拟合','Mesh fitted'],
    linkOrigin:['连杆原点','Link origin'],
    locatedLink:['已定位','Located'],
    rootLink:['根 link 无父关节','Root link has no parent joint'],
    linkCodeMissing:['当前编辑器未找到对应 link 代码块','Link block not found in the current editor'],
    editJointAngle:['点击修改关节位置','Click to edit joint position'],
    invalidJointAngle:['请输入有效数值','Enter a finite number'],
    jointAngleClamped:['已调整到限位边界','Clamped to joint limit'],
    linkMeasure:['测量','Measure'],
    linkMeasureBasis:['连杆原点 · 场景坐标 · B − A','Link origins · Scene coordinates · B − A'],
    pickLinkA:['请选择起点 link A · 可点击结构节点或三维连杆','Select start link A in the structure or 3D view'],
    pickLinkB:['请选择终点 link B','Select end link B'],
    linkMeasureAgain:['再次点击 link 开始新测量 · Esc 退出','Click a link for a new pair · Esc to exit'],
    ctrlLinkHint:['按住 Ctrl 选择 link · 松开恢复关节拖动 · Esc 清除','Hold Ctrl to choose links · Release to drag joints · Esc clears'],
    pickLinkOnly:['请点击 link 连杆节点，关节和 Ground 不作为测量端点','Choose a link node; joints and Ground are not endpoints'],
    pickOtherLink:['请选择另一个不同的 link','Choose a different link'],
    helpTitle:['使用指南','User guide'],
    restoreLayout:['恢复面板布局','Restore panel layout'],
    applied:['已应用','Applied'],
    modified:['未应用','Modified'],
    saved:['已保存','Saved'],
    saving:['保存中…','Saving…'],
    appliedUnsaved:['已应用 · 未保存','Applied · Unsaved'],
    savedUnapplied:['已保存 · 未应用','Saved · Not applied'],
    saveFailed:['保存失败','Save failed'],
    saveUnsupported:['当前浏览器不能覆盖本地文件，请使用 Chrome / Edge 并通过 localhost 或 HTTPS 打开','Use Chrome / Edge on localhost or HTTPS to save directly to local files'],
    chooseSource:['选择对应源文件','Select the original source file'],
    wrongSource:['请选择当前模型对应的源文件','Select the source file for the current model'],
    sourceChanged:['源文件内容已在其他地方修改，请重新打开源文件并合并修改后再保存','The source changed outside this editor. Reopen it and merge your edits before saving'],
    writeDenied:['未获得源文件写入权限，编辑内容仍保留','Write permission was not granted. Your edits are retained'],
    loading:['正在加载模型…','Loading model…'],
    chooseFirst:['请先选择示例模型','Choose an example model first'],
    noUrdf:['未找到 URDF / XML 文件，请选择包含模型的文件夹','No URDF / XML found. Choose a folder containing a robot model.'],
    pickSecond:['已选第一个点，请 Shift + 点击第二个点','First point selected. Shift + click a second point.'],
    emptySource:['请先加载模型或在编辑器中输入 URDF','Load a model or enter URDF in the editor first'],
    noMovable:['模型没有可动画的单轴关节','This model has no movable single-axis joints'],
    unsaved:['有未处理的编辑；请先保存或重新加载，再切换模型','Save or apply your edits before switching models'],
    loadFailed:['加载失败','Load failed'],
    meshMissing:['资源加载失败','Missing resources'],
    modelLoaded:['模型已加载','Model loaded'],
    fixed:['固定','Fixed'],
    fixedNodes:['固定节点','Fixed nodes'],
    jointTotal:['个关节','joints'],
    linkTotal:['个连杆','links'],
    visible:['显示','Visible'],
    hidden:['隐藏','Hidden'],
    invalidXml:['XML 格式错误','Invalid XML'],
    invalidRobot:['需要一个 <robot> 根节点及至少一个命名的 <link>','A <robot> root and at least one named <link> are required'],
    invalidJoint:['关节的父子连杆不存在、重复、断开或形成环','Joint links are missing, duplicated, disconnected or cyclic'],
    badNames:['连杆和关节名称不能为空或重复','Link and joint names must be unique and nonempty'],
    unsupportedJoint:['关节类型不受支持','Unsupported joint type'],
    badNumber:['URDF 中存在无效数值','Invalid numeric value in URDF'],
    fileChoice:['文件夹包含多个模型，请选择','Multiple robot files found. Choose one'],
    saveHint:['已保存到源文件','Saved to the source file'],
    newModel:['模型','Model']
  };
  let lang = 'zh';
  const t = key => strings[key]?.[lang === 'zh' ? 0 : 1] || key;
  const state = { display:{visual:true,collision:false,inertia:false,com:false,axes:false,jointAxes:false,shadow:true,lighting:true,grid:true}, radians:true, limits:true, source:'', baseline:'', saving:false, context:null, version:0, selected:null, rows:new Map(), helpers:[], mobile:'files', activeTab:'files', sidebarCollapsed:false, editorOpen:innerWidth>1000 };
  const sourceFiles=new URDFSourceFiles(t);
  // Keep initialization failures visible even when WebGL is unavailable.
  let THREE, viewer, grid;
  let activeDragJoint=null,statics=null,motion=null;
  const jointAngleUI={dragging:false,editing:false,hovered:false,hideTimer:0,frame:0,factor:1,unit:''};
  function status(message, kind='') { $('status-text').removeAttribute('data-i18n'); $('status-text').textContent=message; $('status').className=kind; }
  function fail(error) { $('editor-error').hidden=false; $('editor-error').textContent=error.message || String(error); status(`${t('loadFailed')} · ${error.message || error}`, 'error'); }
  try {
    THREE = parcelRequire('../node_modules/three/build/three.module.js');
    const Manipulator = parcelRequire('../src/urdf-manipulator-element.js').default;
    customElements.define('urdf-viewer', Manipulator);
    viewer = document.querySelector('urdf-viewer');
    if (!viewer.renderer) throw new Error('WebGL 初始化失败，请启用浏览器硬件加速 / WebGL.');
  } catch (error) { fail(error); return; }
  const URDFLoader = parcelRequire('../src/URDFLoader.js').default;
  const loaders = {};
  for (const name of ['STLLoader','ColladaLoader','OBJLoader','GLTFLoader']) {
    loaders[name] = parcelRequire(`../node_modules/three/examples/jsm/loaders/${name}.js`)[name];
  }
  const updateEnvironment=viewer._updateEnvironment.bind(viewer);
  viewer._updateEnvironment=()=>{
    if(!viewer.robot || objectBounds(viewer.robot).isEmpty())return;
    const target=motion?.groundLocked?viewer.controls.target.clone():null;
    updateEnvironment();
    grid.position.y=viewer.plane.position.y+.002;
    statics?.syncTerrainEnvironment();
    if(target){viewer.controls.target.copy(target);motion.restoreGround();}
  };
  viewer.noAutoRecenter=true;
  viewer.highlightColor='#68b7ff';
  viewer.controls.rotateSpeed=1;
  viewer.controls.zoomSpeed=1.2;
  viewer.controls.panSpeed=1;
  viewer.camera.position.set(3,1.8,3.2);
  viewer.camera.near=.01;
  viewer.camera.updateProjectionMatrix();
  viewer.controls.target.set(0,.3,0);
  // Short segments remain stable when a grid line crosses the camera near plane.
  const gridVertices=[];
  for(let i=-20;i<=20;i++)for(let j=-20;j<20;j++) {
    gridVertices.push(i*.5,0,j*.5,i*.5,0,(j+1)*.5,j*.5,0,i*.5,(j+1)*.5,0,i*.5);
  }
  const gridGeometry=new THREE.BufferGeometry();
  gridGeometry.setAttribute('position',new THREE.Float32BufferAttribute(gridVertices,3));
  grid=new THREE.LineSegments(gridGeometry,new THREE.LineBasicMaterial({color:0x6ba3c8,transparent:true,opacity:.32,depthWrite:false}));
  grid.type='GridHelper';
  viewer.scene.add(grid);
  viewer.plane.position.y=-.002;
  viewer.plane.material.depthWrite=false;
  viewer.redraw();

  const vector = (text, fallback=[0,0,0]) => text ? text.trim().split(/\s+/).map(Number) : fallback;
  const direct = (node, tag) => [...node.children].find(child => child.tagName===tag);
  const nodes = (node, tag) => [...node.children].filter(child => child.tagName===tag);
  function validate(text) {
    const xml = new DOMParser().parseFromString(text, 'application/xml');
    const parseError=xml.querySelector('parsererror');
    if (parseError) throw new Error(`${t('invalidXml')}: ${parseError.textContent.trim().slice(0,230)}`);
    const robot=xml.documentElement;
    const links=nodes(robot,'link'), joints=nodes(robot,'joint');
    if (robot.tagName!=='robot' || !links.length || !links.every(l=>l.getAttribute('name'))) throw new Error(t('invalidRobot'));
    for (const list of [links,joints]) {
      const names=list.map(n=>n.getAttribute('name'));
      if (names.some(n=>!n) || new Set(names).size!==names.length) throw new Error(t('badNames'));
    }
    const names=new Set(links.map(l=>l.getAttribute('name'))), parents=new Map();
    for (const j of joints) {
      const name=j.getAttribute('name'), type=j.getAttribute('type');
      if (!['fixed','revolute','continuous','prismatic','floating','planar'].includes(type)) throw new Error(`${t('unsupportedJoint')}: ${name} (${type})`);
      const parent=direct(j,'parent')?.getAttribute('link'), child=direct(j,'child')?.getAttribute('link');
      if (!names.has(parent)||!names.has(child)||parent===child||parents.has(child)) throw new Error(`${t('invalidJoint')}: ${name}`);
      parents.set(child,parent);
      if (['revolute','prismatic'].includes(type)) {
        const limit=direct(j,'limit');
        const lo=Number(limit?.getAttribute('lower')??0), hi=Number(limit?.getAttribute('upper')??0);
        if (!limit || !Number.isFinite(lo) || !Number.isFinite(hi) || lo>hi) throw new Error(`${t('badNumber')}: ${name} <limit>`);
      }
    }
    for (const name of names) {
      const visited=new Set(); let current=name;
      while(parents.has(current)) { if(visited.has(current)) throw new Error(t('invalidJoint')); visited.add(current); current=parents.get(current); }
    }
    if ([...names].filter(n=>!parents.has(n)).length!==1) throw new Error(t('invalidJoint'));
    for (const el of robot.querySelectorAll('*')) {
      for (const key of ['xyz','rpy','scale','ixx','iyy','izz','ixy','ixz','iyz','radius','length','effort','velocity','lower','upper']) {
        if (el.hasAttribute(key)) {
          const values=vector(el.getAttribute(key));
          if(values.some(n=>!Number.isFinite(n)) || (['xyz','rpy','scale'].includes(key)&&values.length!==3)) throw new Error(`${t('badNumber')}: <${el.tagName}> ${key}`);
        }
      }
      if (el.tagName==='mass' && (!el.hasAttribute('value') || !Number.isFinite(Number(el.getAttribute('value'))) || Number(el.getAttribute('value'))<0)) throw new Error(`${t('badNumber')}: mass`);
    }
    return xml;
  }
  function dispose(object) {
    const geometries=new Set(), materials=new Set(), textures=new Set();
    object?.traverse(child=>{if(child.geometry)geometries.add(child.geometry); for(const mat of child.material ? (Array.isArray(child.material)?child.material:[child.material]) : []) {materials.add(mat);for(const value of Object.values(mat))if(value?.isTexture)textures.add(value);}});
    textures.forEach(texture=>texture.dispose()); geometries.forEach(geometry=>geometry.dispose()); materials.forEach(material=>{if(material!==viewer._collisionMaterial&&material!==viewer.highlightMaterial)material.dispose();}); object?.removeFromParent();
  }
  function normalize(path) { const out=[]; for(const part of decodeURIComponent(path).replace(/\\/g,'/').split('/')) {if(part==='..')out.pop();else if(part&&part!=='.')out.push(part);} return out.join('/'); }
  function releaseContext(context) { if(context?.assets)for(const asset of context.assets.values())if(asset.url)URL.revokeObjectURL(asset.url); }
  function resolveAsset(path, context) {
    if (!context?.assets || /^(blob:|data:)/.test(path)) return path;
    const url=new URL(path,location.href), key=normalize(url.pathname.replace(/^\/__local__\//,''));
    let asset=context.assets.get(key);
    if (!asset) {
      const candidates=[...context.assets.entries()].filter(([name])=>key.endsWith('/'+name)||name.endsWith('/'+key));
      if(candidates.length===1)asset=candidates[0][1];
    }
    // A basename is safe only when unique, including mixed package folders.
    if (!asset) {const base=key.split('/').pop(); const candidates=[...context.assets.entries()].filter(([name])=>name.split('/').pop()===base); if(candidates.length===1)asset=candidates[0][1];}
    if (!asset) return path;
    if (!asset.url)asset.url=URL.createObjectURL(asset.file);
    return asset.url;
  }
  function loadMesh(path, manager, done) {
    const extension=path.split(/[?#]/)[0].split('.').pop().toLowerCase();
    const finish=(object,error)=>{if(object)object.traverse(child=>{if(child.isLight)child.visible=false;});done(object,error);};
    const onError=error=>finish(null,error);
    if(extension==='stl')new loaders.STLLoader(manager).load(path,geometry=>finish(new THREE.Mesh(geometry,new THREE.MeshPhongMaterial({color:0xbfc5cb}))),undefined,onError);
    else if(extension==='dae')new loaders.ColladaLoader(manager).load(path,result=>finish(result.scene),undefined,onError);
    else if(extension==='obj')new loaders.OBJLoader(manager).load(path,finish,undefined,onError);
    else if(['gltf','glb'].includes(extension))new loaders.GLTFLoader(manager).load(path,result=>finish(result.scene),undefined,onError);
    else {manager.itemError(path);finish(null,new Error(`Unsupported mesh: ${extension}`));}
  }
  function stopAnimation() { motion?.pause(); }
  function hasEdits() { return state.saving || ($('source').value!==state.baseline && $('source').value!==state.context?.sourceFile?.savedText); }
  async function loadText(text, context, {replaceEditor=true}={}) {
    text=text.replace(/\r\n?/g,'\n');
    const xml=validate(text);
    context.sourceFile ||= {name:context.name,handle:null,savedText:null,written:false};
    const version=++state.version;
    stopAnimation(); status(t('loading'),'loading'); $('editor-error').hidden=true;
    const failures=new Set();
    const manager=new THREE.LoadingManager();
    manager.setURLModifier(path=>resolveAsset(path,context));
    manager.onError=path=>failures.add(path);
    const loader=new URDFLoader(manager);
    loader.parseCollision=true;
    loader.workingPath=context?.base || new URL('.',location.href).href;
    loader.packages=pkg=>context?.assets ? new URL(`/__local__/${pkg}`,location.href).href : new URL(`urdf/${pkg}`,location.href).href;
    loader.loadMeshCb=loadMesh;
    let robot;
    const ready=new Promise(resolve=>manager.onLoad=resolve);
    manager.itemStart('workbench:parse');
    try {robot=loader.parse(text);} catch(error) {manager.onLoad=()=>{};throw error;} finally {manager.itemEnd('workbench:parse');}
    await ready;
    if(version!==state.version){dispose(robot);return false;}
    // Swap only after XML and meshes finish. Invalid edits retain the current model.
    resetModelPointer();clearJointAngleUI();cancelLinkNavigation();clearMeasurement(); clearSelection();
    state.helpers.forEach(helper=>dispose(helper.object)); state.helpers=[];
    motion?.beforeModelChanged();dispose(viewer.robot);
    const oldContext=state.context;
    viewer.robot=robot;viewer.world.visible=true;viewer.world.add(robot);
    robot.traverse(child=>{if(child.isMesh){child.castShadow=true;child.receiveShadow=true;const mats=Array.isArray(child.material)?child.material:[child.material];mats.forEach(mat=>{if(mat?.map)mat.map.colorSpace=THREE.SRGBColorSpace;});}});
    viewer._setIgnoreLimits(!state.limits);
    state.context=context;state.source=text;state.xml=xml;
    if(oldContext?.assets!==context?.assets)releaseContext(oldContext);
    if(replaceEditor){$('source').value=text;$('filename').value=context.name;state.baseline=text;}
    else state.baseline=text;
    document.body.classList.add('model-loaded');
    $('files-panel').classList.add('has-model');$('loaded-file').hidden=false;
    $('model-name').textContent=robot.robotName || xml.documentElement.getAttribute('name') || context.name;
    $('model-name').title=context.name;
    const linkCount=Object.keys(robot.links).length, jointCount=Object.keys(robot.joints).length;
    $('model-meta').textContent=`${linkCount} ${t('linkTotal')} · ${jointCount} ${t('jointTotal')}`;
    $('joints-empty').hidden=true;$('joint-content').hidden=false;$('structure-empty').hidden=true;
    $('structure-count').hidden=false;$('structure-count').textContent=`${linkCount} / ${jointCount}`;
    $('graph-measure').disabled=false;
    buildHierarchy();buildFileTree();buildJoints();buildTree();buildHelpers();applyDisplay();fitView();updateEditor();statics?.modelChanged();motion?.modelChanged();
    if(failures.size){status(`${t('meshMissing')} · ${failures.size}`, 'error');$('editor-error').textContent=`${t('meshMissing')}: ${[...failures].map(p=>decodeURIComponent(p)).join('\n')}`;$('editor-error').hidden=false;}
    else status(`${t('modelLoaded')} · ${context.name}`);
    return true;
  }
  let catalogPromise;
  async function loadExample() {
    if(hasEdits())return status(t('unsaved'),'error');
    const path=$('example-select').value;
    if(!path)return status(t('chooseFirst'),'error');
    // Some exported examples resolve meshes from their package root.
    const base=$('example-select').selectedOptions[0].dataset.base || '.';
    const version=++state.version;
    status(t('loading'),'loading');
    try {
      const url=new URL(path,location.href);
      const response=await fetch(url,{cache:'no-store'});
      if(!response.ok)throw new Error(`HTTP ${response.status}`);
      const text=await response.text();
      if(version!==state.version)return;
      catalogPromise ||= fetch(new URL('urdf/catalog.json',location.href)).then(response=>response.ok?response.json():{}).catch(()=>({}));
      const catalog=await catalogPromise;
      if(version!==state.version)return;
      await loadText(text,{name:path.split('/').pop(),path:path.replace(/^urdf\//,''),files:catalog[path.split('/')[1]],sourceFile:sourceFiles.fromURL(url,text),base:new URL(base,url).href});
    }catch(error){
      if(version===state.version || state.version===version+1)fail(error);
    }
  }
  async function chooseUrdf(files) {
    const candidates=files.filter(file=>/\.(urdf|xml)$/i.test(file.name)).sort((a,b)=>a.name.localeCompare(b.name));
    if(!candidates.length)throw new Error(t('noUrdf'));
    if(candidates.length===1)return candidates[0];
    const dialog=document.createElement('dialog'), heading=document.createElement('h3');heading.textContent=t('fileChoice');dialog.append(heading);
    return new Promise(resolve=>{for(const file of candidates){const button=document.createElement('button');button.textContent=file.webkitRelativePath || file._relativePath || file.name;button.style.cssText='display:block;width:100%;margin:8px 0;white-space:normal;overflow-wrap:anywhere';button.onclick=()=>{dialog.close();dialog.remove();resolve(file);};dialog.append(button);}dialog.addEventListener('cancel',()=>{dialog.remove();resolve(null);});document.body.append(dialog);dialog.showModal();});
  }
  async function importFiles(files) {
    if(hasEdits())return status(t('unsaved'),'error');
    let context;
    try {
      const file=await chooseUrdf(files);if(!file)return;
      const assets=new Map(files.map(file=>[normalize(file.webkitRelativePath || file._relativePath || file.name),sourceFiles.fromFile(file)]));
      const path=normalize(file.webkitRelativePath || file._relativePath || file.name);
      context={name:file.name,path,assets,sourceFile:assets.get(path),base:new URL(`/__local__/${path.slice(0,path.lastIndexOf('/')+1)}`,location.href).href};
      const loaded=await loadText(await sourceFiles.read(context.sourceFile),context);if(!loaded)releaseContext(context);
    }catch(error){if(context!==state.context)releaseContext(context);fail(error);}
  }
  async function readEntry(entry, prefix='') {
    if(entry.isFile)return new Promise((resolve,reject)=>entry.file(file=>{Object.defineProperty(file,'_relativePath',{value:prefix+file.name});resolve([file]);},reject));
    const reader=entry.createReader(), children=[];
    while(true){const batch=await new Promise((resolve,reject)=>reader.readEntries(resolve,reject));if(!batch.length)break;children.push(...batch);}
    return (await Promise.all(children.map(child=>readEntry(child,prefix+entry.name+'/')))).flat();
  }
  $('load-example').onclick=loadExample;
  async function pickLocal(folder) {
    if(hasEdits())return status(t('unsaved'),'error');
    if(!(folder?window.showDirectoryPicker:window.showOpenFilePicker))return $(folder?'folder-input':'file-input').click();
    try {await importFiles(await (folder?sourceFiles.pickDirectory():sourceFiles.pickFiles()));}
    catch(error){if(error.name!=='AbortError')fail(error);}
  }
  $('load-files').onclick=()=>pickLocal(false);$('load-folder').onclick=()=>pickLocal(true);
  for(const id of ['file-input','folder-input'])$(id).onchange=event=>{const files=[...event.target.files];event.target.value='';if(files.length)importFiles(files);};
  let dragDepth=0;
  window.addEventListener('dragenter',event=>{if([...event.dataTransfer.types].includes('Files')){event.preventDefault();dragDepth++;$('drop-overlay').hidden=false;}});
  window.addEventListener('dragover',event=>{if([...event.dataTransfer.types].includes('Files')){event.preventDefault();event.dataTransfer.dropEffect='copy';}});
  window.addEventListener('dragleave',()=>{dragDepth=Math.max(0,dragDepth-1);if(!dragDepth)$('drop-overlay').hidden=true;});
  window.addEventListener('drop',async event=>{
    event.preventDefault();dragDepth=0;$('drop-overlay').hidden=true;
    const items=[...event.dataTransfer.items].filter(item=>item.kind==='file');
    // Capture handles and fallback entries in the drop event before awaiting anything.
    const handles=items.map(item=>item.getAsFileSystemHandle?.()||Promise.resolve(null));
    const entries=items.map(item=>item.webkitGetAsEntry?.()).filter(Boolean),fallback=[...event.dataTransfer.files];
    if(hasEdits())return status(t('unsaved'),'error');
    try {
      const resolved=await Promise.all(handles.map(p=>p.catch(()=>null)));
      const files=resolved.length&&resolved.every(Boolean)?(await Promise.all(resolved.map(h=>sourceFiles.readHandle(h)))).flat():entries.length?(await Promise.all(entries.map(entry=>readEntry(entry)))).flat():fallback;
      await importFiles(files);
    }catch(error){fail(error);}
  });
  window.addEventListener('blur',()=>{dragDepth=0;$('drop-overlay').hidden=true;});

  function filePath(context) {
    return context.path || context.name;
  }
  function sourceContext(path, context) {
    if(context.assets) {
      return {name:path.split('/').pop(),path,assets:context.assets,sourceFile:context.assets.get(path),
        base:new URL(`/__local__/${path.slice(0,path.lastIndexOf('/')+1)}`,location.href).href};
    }
    const url=new URL(`urdf/${path}`,location.href);
    const sameFolder=path.slice(0,path.lastIndexOf('/'))===filePath(context).slice(0,filePath(context).lastIndexOf('/'));
    return {name:path.split('/').pop(),path,files:context.files,
      base:sameFolder?context.base:new URL('.',url).href};
  }
  async function openTreeFile(path) {
    if(path===filePath(state.context))return;
    if(hasEdits())return status(t('unsaved'),'error');
    const context=sourceContext(path,state.context);
    try {
      let text;
      if(context.assets)text=await sourceFiles.read(context.sourceFile);
      else {
        const url=new URL(`urdf/${path}`,location.href);
        const response=await fetch(url,{cache:'no-store'});
        if(!response.ok)throw new Error(`HTTP ${response.status}`);
        text=await response.text();context.sourceFile=sourceFiles.fromURL(url,text);
      }
      await loadText(text,context);
    }catch(error){fail(error);}
  }
  function buildFileTree() {
    const context=state.context, active=filePath(context);
    const paths=context.assets?[...context.assets.keys()]:(context.files || [active]);
    const root={children:new Map()};
    for(const path of paths) {
      let parent=root;
      const parts=path.split('/');
      parts.forEach((name,index)=>{
        if(!parent.children.has(name))parent.children.set(name,{name,path:parts.slice(0,index+1).join('/'),children:new Map(),directory:index<parts.length-1});
        parent=parent.children.get(name);
      });
    }
    $('file-tree').replaceChildren();
    function append(node,parent,depth) {
      const entries=[...node.children.values()].sort((a,b)=>Number(b.directory)-Number(a.directory)||a.name.localeCompare(b.name,undefined,{numeric:true}));
      for(const entry of entries) {
        if(entry.directory) {
          const folder=document.createElement('details');
          folder.className='file-folder';
          folder.open=depth===0 || active.startsWith(entry.path+'/');
          const summary=document.createElement('summary');
          summary.setAttribute('role','treeitem');summary.setAttribute('aria-label',entry.name);
          const icon=document.createElement('span');icon.className='folder-icon';icon.textContent='📁';
          const name=document.createElement('span');name.textContent=entry.name;
          summary.append(icon,name);folder.append(summary);parent.append(folder);append(entry,folder,depth+1);
        }else{
          const isRobot=/\.(urdf|xml)$/i.test(entry.name);
          const row=document.createElement(isRobot?'button':'div');
          row.className='file-entry';row.dataset.path=entry.path;row.setAttribute('role','treeitem');
          row.setAttribute('aria-selected',String(entry.path===active));row.classList.toggle('selected',entry.path===active);
          row.title=entry.path;
          const icon=document.createElement('span');icon.className='file-entry-icon';icon.textContent=isRobot?'▤':'◇';
          const name=document.createElement('span');name.className='file-entry-name';name.textContent=entry.name;
          const type=document.createElement('span');type.className='file-entry-type';type.textContent=isRobot?'URDF':entry.name.split('.').pop().toUpperCase();
          row.append(icon,name,type);if(isRobot)row.onclick=()=>openTreeFile(entry.path);parent.append(row);
        }
      }
    }
    append(root,$('file-tree'),0);
    $('file-tree').hidden=false;
    $('file-tools-toggle').hidden=false;
    $('files-panel').classList.remove('show-import');
    $('file-tools-toggle').setAttribute('aria-expanded','false');
  }
  $('file-tools-toggle').onclick=()=>{
    const visible=$('files-panel').classList.toggle('show-import');
    $('file-tools-toggle').setAttribute('aria-expanded',String(visible));
  };

  // Preserve source sibling order and traverse each complete kinematic chain.
  function buildHierarchy() {
    const linkNodes=new Map(nodes(state.xml.documentElement,'link').map(element=>{
      const name=element.getAttribute('name');
      return [name,{name,kind:'link',object:viewer.robot.links[name],children:[]}];
    }));
    const children=new Set();
    for(const element of nodes(state.xml.documentElement,'joint')) {
      const name=element.getAttribute('name'),parent=direct(element,'parent').getAttribute('link'),child=direct(element,'child').getAttribute('link');
      const joint={name,kind:'joint',object:viewer.robot.joints[name],element,children:[linkNodes.get(child)]};
      linkNodes.get(parent).children.push(joint);children.add(child);
    }
    const root=[...linkNodes.values()].find(link=>!children.has(link.name));
    state.hierarchy=root;
    const joints=[];
    function visit(node){if(node.kind==='joint')joints.push(node);node.children.forEach(visit);}
    visit(root);state.orderedJoints=joints;
  }

  function jointBounds(joint) {
    if(joint.jointType==='continuous')return [-Math.PI,Math.PI];
    if(state.limits)return [Number.isFinite(joint.limit.lower)?joint.limit.lower:0,Number.isFinite(joint.limit.upper)?joint.limit.upper:0];
    return joint.jointType==='prismatic'?[-1,1]:[-Math.PI*2,Math.PI*2];
  }
  const compactNumber=value=>Number.isFinite(value)?String(Number(value.toFixed(3))):'—';
  function buildJoints() {
    state.rows.clear();$('joint-list').replaceChildren();
    for(const node of state.orderedJoints) {
      const joint=node.object;
      const row=document.createElement('li');row.dataset.name=joint.name;row.dataset.type=joint.jointType;
      const heading=document.createElement('div');heading.className='joint-heading';
      const name=document.createElement('span');name.className='joint-name';name.textContent=joint.name;name.title=`${joint.name} · ${joint.jointType}`;
      heading.append(name);
      const limits=direct(node.element,'limit'),parameters=document.createElement('span');parameters.className='joint-parameters';
      for(const [key,label,unit]of [['effort','τ',joint.jointType==='prismatic'?'N':'N·m'],['velocity','v',joint.jointType==='prismatic'?'m/s':'rad/s']]) {
        if(!limits?.hasAttribute(key))continue;
        const parameter=document.createElement('span');parameter.className='joint-parameter';
        const value=document.createElement('span');value.className=`joint-${key}`;value.textContent=compactNumber(Number(limits.getAttribute(key)));
        parameter.append(document.createTextNode(`${label}: `),value);parameter.title=`${key}: ${value.textContent} ${unit}`;
        parameters.append(parameter);
      }
      heading.append(parameters);row.append(heading);
      if(['revolute','continuous','prismatic'].includes(joint.jointType) && !joint.mimicJoint) {
        const inputs=document.createElement('div');inputs.className='joint-inputs';
        const lower=document.createElement('span'),upper=document.createElement('span');
        lower.className='joint-bound joint-lower';upper.className='joint-bound joint-upper';
        const slider=document.createElement('input'),number=document.createElement('input'),unit=document.createElement('span');
        slider.type='range';number.type='number';slider.setAttribute('aria-label',joint.name);number.setAttribute('aria-label',`${joint.name} value`);unit.className='joint-unit';
        inputs.append(lower,slider,upper,number,unit);row.append(inputs);
        const factor=()=>joint.jointType==='prismatic'||state.radians?1:180/Math.PI;
        const update=()=>{
          const f=factor(),[lo,hi]=jointBounds(joint),value=joint.angle || 0;
          slider.min=Math.min(lo,value);slider.max=Math.max(hi,value);slider.step='any';slider.value=value;
          number.min=lo*f;number.max=hi*f;number.step=joint.jointType==='prismatic'?.001:state.radians?.01:.1;
          if(document.activeElement!==number)number.value=(value*f).toFixed(joint.jointType==='prismatic'?3:2);
          lower.textContent=Number.isFinite(lo)?(lo*f).toFixed(2):'−∞';upper.textContent=Number.isFinite(hi)?(hi*f).toFixed(2):'∞';
          lower.title=`min: ${lo*f}`;upper.title=`max: ${hi*f}`;
          unit.textContent=joint.jointType==='prismatic'?'m':state.radians?'rad':'°';
          slider.setAttribute('aria-valuetext',`${number.value} ${unit.textContent}`);
        };
        slider.oninput=()=>{stopAnimation();clearSurfaceMeasurement();viewer.setJointValue(joint.name,Number(slider.value));update();};
        number.onchange=()=>{stopAnimation();const value=Number(number.value)/factor();if(Number.isFinite(value)){clearSurfaceMeasurement();viewer.setJointValue(joint.name,value);}update();};
        state.rows.set(joint.name,{row,update});update();
      }else{
        const type=document.createElement('span');type.className='joint-type';type.textContent=joint.jointType;heading.append(type);
        state.rows.set(joint.name,{row,update:()=>{}});
        if(joint.jointType!=='fixed') {
          const note=document.createElement('span');note.className='muted';note.textContent=joint.mimicJoint?`mimic → ${joint.mimicJoint}`:lang==='zh'?'多自由度关节（仅显示）':'Multi-DOF joint (display only)';row.append(note);
        }
      }
      $('joint-list').append(row);
    }
    filterJoints();
  }
  $('joint-filter-toggle').onclick=()=>{
    const filters=$('joint-filters');filters.hidden=!filters.hidden;
    $('joint-filter-toggle').setAttribute('aria-expanded',String(!filters.hidden));
    $('joints-panel').classList.toggle('show-filters',!filters.hidden);
    if(!filters.hidden)$('joint-search').focus();
  };
  function filterJoints() {let visible=0;const query=$('joint-search').value.toLowerCase();for(const [name,{row}]of state.rows){row.hidden=($('hide-fixed').checked&&row.dataset.type==='fixed')||!name.toLowerCase().includes(query);if(!row.hidden)visible++;}$('joint-count').textContent=`${visible} / ${state.rows.size} ${t('jointTotal')}`;}
  $('joint-search').oninput=filterJoints;$('hide-fixed').onchange=filterJoints;
  for(const [id,radians]of [['unit-rad',true],['unit-deg',false]])$(id).onclick=()=>{state.radians=radians;for(const [other,value]of [['unit-rad',true],['unit-deg',false]]){$(other).classList.toggle('selected',value===radians);$(other).setAttribute('aria-pressed',String(value===radians));}state.rows.forEach(row=>row.update());queueJointAngleUpdate();};
  $('limits').onclick=()=>{state.limits=!state.limits;$('limits').classList.toggle('active',state.limits);$('limits').setAttribute('aria-pressed',String(state.limits));viewer.ignoreLimits=!state.limits;state.rows.forEach(row=>row.update());clearSurfaceMeasurement();queueLinkMeasurementUpdate();};
  $('reset-joints').onclick=()=>{stopAnimation();clearSurfaceMeasurement();Object.values(viewer.robot?.joints||{}).forEach(j=>viewer.setJointValue(j.name,...j.jointValue.map(()=>0)));};
  viewer.addEventListener('angle-change',event=>{state.rows.get(event.detail)?.update();viewer.redraw();});
  for(const [event,on]of [['joint-mouseover',true],['joint-mouseout',false]])viewer.addEventListener(event,e=>state.rows.get(e.detail)?.row.classList.toggle('hovered',on));
  viewer.addEventListener('manipulate-start',event=>{stopAnimation();clearSurfaceMeasurement();setActiveDragJoint(viewer.robot?.joints[event.detail] || null);});
  viewer.addEventListener('manipulate-end',()=>setActiveDragJoint(null));
  $('animate').onclick=()=>motion?.toggle();

  let structureGraph=null;
  let showFixedNodes=false;
  const svgNS='http://www.w3.org/2000/svg';
  function svgElement(tag,attributes={}) {
    const element=document.createElementNS(svgNS,tag);
    for(const [key,value]of Object.entries(attributes))element.setAttribute(key,String(value));
    return element;
  }
  function diagramHierarchy() {
    function linkBranch(link) {
      const children=[];
      function appendJoint(joint) {
        if(!showFixedNodes && joint.object.jointType==='fixed') {
          joint.children[0].children.forEach(appendJoint);
        }else children.push({...joint,children:[linkBranch(joint.children[0])]});
      }
      link.children.forEach(appendJoint);
      return {...link,children};
    }
    return {name:'Ground',kind:'ground',object:viewer.world,children:[linkBranch(state.hierarchy)]};
  }
  function drawGraphTransform() {
    const graph=structureGraph;
    if(graph)graph.layer.setAttribute('transform',`translate(${graph.x} ${graph.y}) scale(${graph.scale})`);
  }
  function fitGraph() {
    const graph=structureGraph,host=$('tree');
    if(!graph || !host.clientWidth || !host.clientHeight)return;
    const padding=22;
    graph.scale=Math.min((host.clientWidth-padding*2)/graph.width,(host.clientHeight-padding*2)/graph.height,1);
    graph.x=(host.clientWidth-graph.width*graph.scale)/2;
    graph.y=Math.min(32,(host.clientHeight-graph.height*graph.scale)/2);
    graph.autoFit=true;drawGraphTransform();
  }
  function zoomGraph(factor,x=$('tree').clientWidth/2,y=$('tree').clientHeight/2) {
    const graph=structureGraph;if(!graph)return;
    const scale=Math.max(.025,Math.min(5,graph.scale*factor));
    graph.x=x-(x-graph.x)*scale/graph.scale;graph.y=y-(y-graph.y)*scale/graph.scale;
    graph.scale=scale;graph.autoFit=false;drawGraphTransform();
  }
  function refreshGraphVisibility() {
    for(const {object,element}of structureGraph?.nodes || []) {
      let visible=true;
      for(let ancestor=object;ancestor;ancestor=ancestor.parent)if(!ancestor.visible)visible=false;
      element.classList.toggle('node-hidden',!visible);
      element.setAttribute('aria-label',`${object.name || 'Ground'} · ${visible?t('visible'):t('hidden')}`);
    }
  }
  function buildTree() {
    const host=$('tree');host.replaceChildren();host.hidden=false;
    const svg=svgElement('svg',{'aria-label':lang==='zh'?'机器人连杆与关节拓扑图':'Robot link and joint topology',tabindex:0});
    const layer=svgElement('g'),edges=svgElement('g',{'class':'structure-edges'}),vertices=svgElement('g');
    layer.append(edges,vertices);svg.append(layer);host.append(svg);
    const root=diagramHierarchy(),allNodes=[];
    function measure(node) {
      const lines=node.kind==='joint'?splitJointName(node.name):[node.name];
      node.lines=lines;node.width=Math.max(node.kind==='joint'?104:110,Math.min(230,Math.max(...lines.map(line=>line.length))*6.8+24));
      node.height=node.kind==='joint'?34:30;
      node.children.forEach(measure);
      node.span=Math.max(node.width+24,node.children.reduce((sum,child)=>sum+child.span,0));
    }
    function position(node,left,depth) {
      node.x=left+node.span/2;node.y=20+depth*58;allNodes.push(node);
      let offset=left+(node.span-node.children.reduce((sum,child)=>sum+child.span,0))/2;
      node.children.forEach(child=>{position(child,offset,depth+1);offset+=child.span;});
    }
    measure(root);position(root,0,0);
    const height=Math.max(...allNodes.map(node=>node.y+node.height/2))+10;
    structureGraph={svg,layer,nodes:[],width:root.span,height,x:0,y:0,scale:1,autoFit:true,dragged:false};
    for(const node of allNodes) {
      for(const child of node.children) {
        const start=node.y+node.height/2,end=child.y-child.height/2,middle=(start+end)/2;
        edges.append(svgElement('path',{d:`M ${node.x} ${start} C ${node.x} ${middle}, ${child.x} ${middle}, ${child.x} ${end}`}));
      }
      const group=svgElement('g',{'class':`structure-node node-${node.kind}`,transform:`translate(${node.x} ${node.y})`,role:'button',tabindex:0,'data-name':node.name,'data-kind':node.kind,'data-model-root':String(node.object===viewer.robot),'aria-pressed':'false'});
      const title=svgElement('title');title.textContent=`${node.name}${node.kind==='joint'?` · ${node.object.jointType}`:''}`;
      const rect=svgElement('rect',{x:-node.width/2,y:-node.height/2,width:node.width,height:node.height,rx:node.kind==='joint'?15:5});
      const text=svgElement('text',{'text-anchor':'middle'});
      node.lines.forEach((line,index)=>{const span=svgElement('tspan',{x:0,y:node.lines.length===1?4:index===0?-2:10});span.textContent=line;text.append(span);});
      group.append(title,rect,text);vertices.append(group);
      structureGraph.nodes.push({object:node.object,element:group});
      const choose=event=>{if(!structureGraph.dragged)selectStructureNode(node.object,group,event);};
      group.onclick=choose;
      group.oncontextmenu=event=>{event.preventDefault();node.object.visible=!node.object.visible;selectObject(node.object,group);refreshGraphVisibility();viewer.redraw();};
      group.onkeydown=event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();event.stopPropagation();choose(event);}else if(event.key.toLowerCase()==='h'){event.preventDefault();node.object.visible=!node.object.visible;selectObject(node.object,group);refreshGraphVisibility();viewer.redraw();}};
      group.onpointerenter=()=>state.rows.get(node.name)?.row.classList.add('hovered');
      group.onpointerleave=()=>state.rows.get(node.name)?.row.classList.remove('hovered');
    }
    let drag=null;
    svg.addEventListener('pointerdown',event=>{
      if(event.button!==0)return;
      structureGraph.dragged=false;
      drag={x:event.clientX,y:event.clientY,initialX:structureGraph.x,initialY:structureGraph.y};
      event.target.setPointerCapture(event.pointerId);
    });
    svg.addEventListener('pointermove',event=>{
      if(!drag)return;
      const dx=event.clientX-drag.x,dy=event.clientY-drag.y;
      if(Math.hypot(dx,dy)>3)structureGraph.dragged=true;
      if(structureGraph.dragged){structureGraph.x=drag.initialX+dx;structureGraph.y=drag.initialY+dy;structureGraph.autoFit=false;drawGraphTransform();}
    });
    const end=()=>{drag=null;setTimeout(()=>{if(structureGraph)structureGraph.dragged=false;},0);};
    svg.addEventListener('pointerup',end);svg.addEventListener('pointercancel',end);
    svg.addEventListener('wheel',event=>{event.preventDefault();const rect=host.getBoundingClientRect();zoomGraph(Math.exp(-event.deltaY*.0015),event.clientX-rect.left,event.clientY-rect.top);},{passive:false});
    svg.addEventListener('dblclick',event=>{if(!event.target.closest('.structure-node'))fitGraph();});
    svg.addEventListener('contextmenu',event=>event.preventDefault());
    svg.addEventListener('keydown',event=>{if(event.key==='0'||event.key==='Home'){event.preventDefault();fitGraph();}else if(event.key==='+'||event.key==='='){event.preventDefault();zoomGraph(1.2);}else if(event.key==='-'){event.preventDefault();zoomGraph(1/1.2);}});
    refreshGraphVisibility();refreshMeasuredNodes();fitGraph();
  }
  function splitJointName(name) {
    if(name.length<18)return [name];
    const middle=Math.floor(name.length/2),split=name.lastIndexOf('_',middle+3);
    if(split>0)return [name.slice(0,split+1),name.slice(split+1)];
    return [name.slice(0,middle),name.slice(middle)];
  }
  $('graph-fit').onclick=fitGraph;
  $('graph-zoom-in').onclick=()=>zoomGraph(1.25);
  $('graph-zoom-out').onclick=()=>zoomGraph(.8);
  $('graph-fixed').onclick=()=>{
    showFixedNodes=!showFixedNodes;
    $('graph-fixed').classList.toggle('active',showFixedNodes);$('graph-fixed').setAttribute('aria-pressed',String(showFixedNodes));
    if(viewer.robot){clearSelection();buildTree();}
  };
  new ResizeObserver(()=>{if(structureGraph?.autoFit)fitGraph();}).observe($('tree'));
  function clearSelection() {
    state.selected=null;$('selection-info').hidden=true;
    $$('#tree .selected').forEach(element=>{element.classList.remove('selected');element.setAttribute('aria-pressed','false');});
  }
  function selectObject(object,row) {
    clearSelection();state.selected=object;row?.classList.add('selected');row?.setAttribute('aria-pressed','true');
    $('selection-info').hidden=false;$('selection-info').textContent=`${object.name || 'Ground'} · ${object.jointType||'link'} · ${object.visible?t('visible'):t('hidden')}`;
    viewer.redraw();statics?.requestProjection();
  }
  function objectBounds(object) {const box=new THREE.Box3();object.updateWorldMatrix(true,true);object.traverse(child=>{if(child.isURDFVisual)box.expandByObject(child);});return box;}
  function selectedBounds(object) {
    if(!object.isURDFLink)return objectBounds(object);
    const box=new THREE.Box3();object.updateWorldMatrix(true,true);
    for(const child of object.children)if((child.isURDFVisual || child.isURDFCollider) && child.visible)box.expandByObject(child);
    return box;
  }

  // Pick rendered model surfaces only. Ground, helper meshes and hidden links cannot intercept the ray.
  function visibleInScene(object) {
    for(let parent=object;parent;parent=parent.parent)if(!parent.visible)return false;
    return true;
  }
  function owningLink(object) {
    for(let parent=object;parent;parent=parent.parent)if(parent.isURDFLink)return parent;
    return null;
  }
  function visibleModelHit(raycaster) {
    if(!viewer.robot || !visibleInScene(viewer.robot))return null;
    viewer.world.updateWorldMatrix(true,true);
    const hits=[];
    viewer.robot.traverseVisible(object=>{
      if(!object.isMesh || !raycaster.layers.test(object.layers))return;
      let surface=object.parent;
      while(surface && !surface.isURDFVisual && !surface.isURDFCollider && surface!==viewer.robot)surface=surface.parent;
      if(!surface?.isURDFVisual && !surface?.isURDFCollider)return;
      // The legacy viewer disables collider raycasts; use their actual mesh when collisions are displayed.
      const raycast=surface.isURDFCollider?THREE.Mesh.prototype.raycast:object.raycast;
      raycast.call(object,raycaster,hits);
    });
    hits.sort((a,b)=>a.distance-b.distance);
    const hit=hits.find(hit=>{
      const material=Array.isArray(hit.object.material)?hit.object.material[hit.face?.materialIndex || 0]:hit.object.material;
      return material?.visible!==false && !(material?.transparent && material.opacity===0);
    });
    return hit?{...hit,link:owningLink(hit.object)}:null;
  }
  const pointerRay=new THREE.Raycaster();
  function rayFromPointer(event) {
    const rect=viewer.renderer.domElement.getBoundingClientRect();
    const point=new THREE.Vector2((event.clientX-rect.left)/rect.width*2-1,1-(event.clientY-rect.top)/rect.height*2);
    viewer.camera.updateMatrixWorld(true);
    pointerRay.setFromCamera(point,viewer.camera);
    return pointerRay;
  }
  let hoveredLink=null;
  const hoverMaterials=new Map();
  let lastModelPointer=null,modelPointerDown=null,hoverFrame=0;
  function setHoveredLink(link) {
    if(hoveredLink===link)return;
    for(const [mesh,{original,highlighted}]of hoverMaterials) {
      if(mesh.material===highlighted)mesh.material=original;
      for(const material of Array.isArray(highlighted)?highlighted:[highlighted])material.dispose();
    }
    hoverMaterials.clear();hoveredLink=link;
    $$('#tree .hovered').forEach(element=>element.classList.remove('hovered'));
    const node=structureGraph?.nodes.find(node=>node.object===link);
    node?.element.classList.add('hovered');
    if(link)for(const child of link.children) {
      if((!child.isURDFVisual && !child.isURDFCollider) || !visibleInScene(child))continue;
      child.traverseVisible(mesh=>{
        if(!mesh.isMesh)return;
        const original=mesh.material;
        const highlighted=(Array.isArray(original)?original:[original]).map(material=>{
          const clone=material.clone();
          if(clone.emissive){clone.emissive.set(0x268fff);clone.emissiveIntensity=.5;}
          else if(clone.color)clone.color.lerp(new THREE.Color(0x70baff),.45);
          return clone;
        });
        mesh.material=Array.isArray(original)?highlighted:highlighted[0];
        hoverMaterials.set(mesh,{original,highlighted:mesh.material});
      });
    }
    const canvas=viewer.renderer.domElement;
    canvas.title=link?.name || '';
    canvas.style.cursor=link?'pointer':'';
    viewer.redraw();
  }
  function revealStructureLink(link) {
    let node=structureGraph?.nodes.find(node=>node.object===link);
    if(!node && !showFixedNodes) {
      showFixedNodes=true;$('graph-fixed').classList.add('active');$('graph-fixed').setAttribute('aria-pressed','true');
      buildTree();node=structureGraph.nodes.find(node=>node.object===link);
    }
    return node;
  }
  function selectLink(link) {selectObject(link,revealStructureLink(link)?.element);}
  function nearestMovableJoint(link) {
    for(let parent=link?.parent;parent;parent=parent.parent)if(parent.isURDFJoint && ['revolute','continuous','prismatic'].includes(parent.jointType))return parent;
    return null;
  }
  let linkLocateVersion=0,linkLocateTimer=0,locatedSource=null;
  const locateAnimations=new Map();
  function flashLocated(element,finished=()=>{}) {
    locateAnimations.get(element)?.cancel();
    const animation=element.animate([
      {backgroundColor:'rgba(107,163,200,0)',boxShadow:'inset 0 0 0 1px rgba(184,212,232,0)'},
      {backgroundColor:'rgba(107,163,200,.38)',boxShadow:'inset 0 0 0 1px rgba(184,212,232,.95)',offset:.35},
      {backgroundColor:'rgba(107,163,200,0)',boxShadow:'inset 0 0 0 1px rgba(184,212,232,0)'}
    ],{duration:420,iterations:2,easing:'ease-in-out'});
    locateAnimations.set(element,animation);
    animation.finished.then(()=>{if(locateAnimations.get(element)===animation){locateAnimations.delete(element);finished();}}).catch(()=>{});
  }
  function cancelLinkNavigation() {
    linkLocateVersion++;clearTimeout(linkLocateTimer);linkLocateTimer=0;
    for(const animation of locateAnimations.values())animation.cancel();locateAnimations.clear();
    locatedSource=null;$('source-locate-highlight').hidden=true;
  }
  function linkSourceRange(text,name) {
    // Tokenize tags with quoted attributes intact; comments and CDATA never name links.
    const tokens=/<!--[\s\S]*?-->|<!\[CDATA\[[\s\S]*?\]\]>|<\?[\s\S]*?\?>|<![^>]*>|<\/?[\w:.-]+\b(?:[^>"']|"[^"]*"|'[^']*')*>/g;
    let depth=0,start=null;
    for(const match of text.matchAll(tokens)) {
      const token=match[0];if(token.startsWith('<!') || token.startsWith('<?'))continue;
      const tag=token.match(/^<(\/?)([\w:.-]+)/),closing=!!tag[1],kind=tag[2],empty=/\/\s*>$/.test(token);
      if(closing) {
        depth--;
        if(start!==null && depth===1 && kind==='link')return {start,end:match.index+token.length};
      }else{
        if(depth===1 && kind==='link') {
          const element=new DOMParser().parseFromString(token.replace(/\/?\s*>$/,'/>'),'application/xml').documentElement;
          if(element.tagName==='link' && element.getAttribute('name')===name) {
            if(empty)return {start:match.index,end:match.index+token.length};
            start=match.index;
          }
        }
        if(!empty)depth++;
      }
    }
    return null;
  }
  function revealLocatePanel(name) {
    if(name==='editor')statics?.setRightView('code',false);
    setPanel(name,true);
    document.querySelector(`[data-collapse="${name}"]`).setAttribute('aria-expanded','true');
  }
  function locateLinkCode(name) {
    const source=$('source'),range=linkSourceRange(source.value,name);
    if(!range)return false;
    revealLocatePanel('editor');
    const firstLine=source.value.slice(0,range.start).split('\n').length-1,lastLine=source.value.slice(0,range.end).split('\n').length-1;
    const style=getComputedStyle(source),lineHeight=parseFloat(style.lineHeight),padding=parseFloat(style.paddingTop);
    source.focus({preventScroll:true});source.setSelectionRange(range.start,range.end);
    source.scrollLeft=0;source.scrollTop=Math.max(0,padding+firstLine*lineHeight-source.clientHeight*.2);
    locatedSource={firstLine,lastLine,lineHeight,padding};
    const highlight=$('source-locate-highlight');highlight.dataset.link=name;highlight.hidden=false;
    updateEditor();syncEditorScroll();
    flashLocated(highlight,()=>{highlight.hidden=true;locatedSource=null;});
    return true;
  }
  function navigateToLink(link) {
    cancelLinkNavigation();selectLink(link);
    const version=linkLocateVersion,modelVersion=state.version;
    const joint=link.parent?.isURDFJoint?link.parent:null,row=joint?state.rows.get(joint.name)?.row:null;
    if(row) {
      revealLocatePanel('joints');
      if(row.hidden){$('joint-search').value='';if(joint.jointType==='fixed')$('hide-fixed').checked=false;filterJoints();}
      const body=$('joints-panel').querySelector('.panel-body');
      body.scrollTop+=row.getBoundingClientRect().top-body.getBoundingClientRect().top-(body.clientHeight-row.offsetHeight)/2;
      flashLocated(row);
    }
    const showCode=()=>{
      if(version!==linkLocateVersion || modelVersion!==state.version)return;
      const found=locateLinkCode(link.name);
      status(`${t('locatedLink')} · ${link.name} → ${joint?.name || t('rootLink')}${found?'':` · ${t('linkCodeMissing')}`}`);
    };
    // Narrow layouts show one panel at a time, so let both double flashes be visible.
    if(row && innerWidth<=600)linkLocateTimer=setTimeout(showCode,900);
    else showCode();
  }
  function updateJointHover(joint) {
    const controls=viewer.dragControls;
    if(controls.hovered===joint)return;
    if(controls.hovered)controls.onUnhover(controls.hovered);
    controls.hovered=joint;
    if(joint)controls.onHover(joint);
  }
  function resetModelPointer() {
    const canvas=viewer.renderer.domElement,pointer=modelPointerDown;
    modelPointerDown=null;lastModelPointer=null;
    viewer.dragControls.setGrabbed(false);updateJointHover(null);setHoveredLink(null);
    if(pointer && canvas.hasPointerCapture(pointer.id))canvas.releasePointerCapture(pointer.id);
  }
  function refreshModelHover() {
    if(!lastModelPointer || modelPointerDown || hoverFrame)return;
    hoverFrame=requestAnimationFrame(()=>{
      hoverFrame=0;
      if(lastModelPointer && !modelPointerDown) {
        if(lastModelPointer.shiftKey)updateSurfaceSnap(lastModelPointer);
        else viewer.dragControls.moveRay(rayFromPointer(lastModelPointer).ray);
      }
    });
  }
  function installModelPicking() {
    const controls=viewer.dragControls,canvas=viewer.renderer.domElement;
    // Replace the old mouse listeners and broad joint-subtree material highlighting.
    controls.dispose();
    controls.onHover=joint=>viewer.dispatchEvent(new CustomEvent('joint-mouseover',{detail:joint.name}));
    controls.onUnhover=joint=>viewer.dispatchEvent(new CustomEvent('joint-mouseout',{detail:joint.name}));
    controls.update=function() {
      if(this.manipulating)return;
      const hit=visibleModelHit(this.raycaster);
      this.hitDistance=hit?.distance ?? -1;
      if(hit)this.initialGrabPoint.copy(hit.point);
      setHoveredLink(hit?.link || null);
      updateJointHover(lastModelPointer?.ctrlKey || lastModelPointer?.metaKey || linkMeasurement.pinned?null:nearestMovableJoint(hit?.link));
    };
    canvas.addEventListener('pointerdown',event=>{
      if(event.button!==0 || event.shiftKey || event.ctrlKey || event.metaKey || linkMeasurement.pinned)return;
      if(!event.isPrimary){resetModelPointer();return;}
      lastModelPointer=event.pointerType==='touch'?null:{clientX:event.clientX,clientY:event.clientY,shiftKey:event.shiftKey,ctrlKey:event.ctrlKey,metaKey:event.metaKey,altKey:event.altKey};
      controls.moveRay(rayFromPointer(event).ray);
      if(motion?.isBodyLink(hoveredLink))motion.pause();else motion?.hideBodyEditor();
      modelPointerDown={id:event.pointerId,x:event.clientX,y:event.clientY,link:hoveredLink,moved:false,joint:false};
      if(event.pointerType!=='touch' && controls.hovered && !viewer.disableDragging) {
        event.preventDefault();event.stopImmediatePropagation();
        modelPointerDown.joint=true;controls.setGrabbed(true);canvas.setPointerCapture(event.pointerId);canvas.style.cursor='grabbing';
      }
    },true);
    canvas.addEventListener('pointermove',event=>{
      lastModelPointer=event.pointerType==='touch'?null:{clientX:event.clientX,clientY:event.clientY,shiftKey:event.shiftKey,ctrlKey:event.ctrlKey,metaKey:event.metaKey,altKey:event.altKey};
      if(!modelPointerDown && event.shiftKey){setHoveredLink(null);updateJointHover(null);updateSurfaceSnap(lastModelPointer);return;}
      clearSurfaceSnap();
      if(modelPointerDown) {
        if(event.pointerId!==modelPointerDown.id)return;
        if(Math.hypot(event.clientX-modelPointerDown.x,event.clientY-modelPointerDown.y)>4)modelPointerDown.moved=true;
        if(modelPointerDown.joint){event.preventDefault();event.stopImmediatePropagation();controls.moveRay(rayFromPointer(event).ray);}
      }else if(event.pointerType!=='touch')controls.moveRay(rayFromPointer(event).ray);
    },true);
    canvas.addEventListener('pointerup',event=>{
      const pointer=modelPointerDown;
      if(!pointer || event.pointerId!==pointer.id)return;
      if(pointer.joint) {
        event.preventDefault();event.stopImmediatePropagation();
        controls.moveRay(rayFromPointer(event).ray);controls.setGrabbed(false);
      }
      modelPointerDown=null;
      if(canvas.hasPointerCapture(event.pointerId) && pointer.joint)canvas.releasePointerCapture(event.pointerId);
      const hit=visibleModelHit(rayFromPointer(event));
      if(!pointer.moved){
        if(pointer.link && hit?.link===pointer.link){selectLink(hit.link);if(!motion?.pickBody(hit.link,hit.point))motion?.hideBodyEditor();}
        else motion?.hideBodyEditor();
      }
      setHoveredLink(event.pointerType==='touch'?null:hit?.link || null);
      canvas.style.cursor=hoveredLink?'pointer':'';
    },true);
    canvas.addEventListener('pointerleave',()=>{clearSurfaceSnap();if(!modelPointerDown){lastModelPointer=null;setHoveredLink(null);updateJointHover(null);}});
    canvas.addEventListener('dblclick',event=>{
      if(event.button!==0 || event.ctrlKey || event.metaKey || event.shiftKey || linkMeasurement.pinned)return;
      const hit=visibleModelHit(rayFromPointer(event));if(!hit?.link)return;
      event.preventDefault();event.stopPropagation();motion?.hideBodyEditor();navigateToLink(hit.link);
    });
    canvas.addEventListener('pointercancel',resetModelPointer);
    canvas.addEventListener('lostpointercapture',()=>{if(modelPointerDown?.joint)resetModelPointer();});
    window.addEventListener('blur',()=>{resetModelPointer();clearSurfaceSnap();if(jointAngleUI.editing)finishJointAngleEdit();jointAngleUI.hovered=false;scheduleJointAngleHide();});
    viewer.controls.addEventListener('change',refreshModelHover);
    viewer.addEventListener('angle-change',refreshModelHover);
  }

  function applyOrigin(group, origin) {const xyz=vector(origin?.getAttribute('xyz')), rpy=vector(origin?.getAttribute('rpy'));group.position.set(...xyz);group.rotation.set(rpy[0],rpy[1],rpy[2],'ZYX');}
  // Jacobi diagonalization of a real symmetric 3×3 inertia tensor.
  function principalAxes(a) {
    const v=[[1,0,0],[0,1,0],[0,0,1]];
    for(let n=0;n<30;n++) {let p=0,q=1;for(const [i,j]of [[0,2],[1,2]])if(Math.abs(a[i][j])>Math.abs(a[p][q])){p=i;q=j;}if(Math.abs(a[p][q])<1e-12)break;
      const theta=.5*Math.atan2(2*a[p][q],a[q][q]-a[p][p]),c=Math.cos(theta),s=Math.sin(theta),app=a[p][p],aqq=a[q][q],apq=a[p][q];
      a[p][p]=c*c*app-2*s*c*apq+s*s*aqq;a[q][q]=s*s*app+2*s*c*apq+c*c*aqq;a[p][q]=a[q][p]=0;
      for(let k=0;k<3;k++){if(k!==p&&k!==q){const akp=a[k][p],akq=a[k][q];a[k][p]=a[p][k]=c*akp-s*akq;a[k][q]=a[q][k]=s*akp+c*akq;}const vp=v[k][p],vq=v[k][q];v[k][p]=c*vp-s*vq;v[k][q]=s*vp+c*vq;}
    }return {values:[a[0][0],a[1][1],a[2][2]],vectors:v};
  }
  function helper(object,kind,parent){parent.add(object);state.helpers.push({object,kind});return object;}
  function setActiveDragJoint(joint) {
    clearTimeout(jointAngleUI.hideTimer);
    if(joint) {
      activeDragJoint=joint;jointAngleUI.dragging=true;jointAngleUI.editing=false;jointAngleUI.hovered=false;
      $('joint-angle-feedback').textContent='';
    }else{
      jointAngleUI.dragging=false;scheduleJointAngleHide();
    }
    refreshJointAngleAxis();renderJointAngle();
  }
  function refreshJointAngleAxis() {
    for(const {object,kind}of state.helpers)if(kind==='jointAxes') {
      object.visible=state.display.jointAxes || (!!statics?.enabled && !object.userData.staticsInspectionHidden) || object.parent===activeDragJoint;
      object.userData.dragActive=jointAngleUI.dragging && object.parent===activeDragJoint;
    }
    viewer.redraw();
  }
  function scheduleJointAngleHide() {
    clearTimeout(jointAngleUI.hideTimer);
    if(activeDragJoint && !jointAngleUI.dragging && !jointAngleUI.editing && !jointAngleUI.hovered)jointAngleUI.hideTimer=setTimeout(clearJointAngleUI,3000);
  }
  function clearJointAngleUI() {
    clearTimeout(jointAngleUI.hideTimer);activeDragJoint=null;
    jointAngleUI.dragging=false;jointAngleUI.editing=false;jointAngleUI.hovered=false;
    $('joint-angle-tooltip').hidden=true;$('joint-angle-tooltip').removeAttribute('data-joint');
    refreshJointAngleAxis();
  }
  function renderJointAngle() {
    const joint=activeDragJoint,tooltip=$('joint-angle-tooltip');
    if(!joint || viewer.robot?.joints[joint.name]!==joint){tooltip.hidden=true;return;}
    const glyph=joint.children.find(object=>object.userData.axisKind==='jointAxes');
    if(!glyph){tooltip.hidden=true;return;}
    viewer.world.updateWorldMatrix(true,true);viewer.camera.updateMatrixWorld(true);
    const point=glyph.localToWorld(new THREE.Vector3(...glyph.userData.angleAnchor)),projected=projectMeasurementPoint(point);
    if(!visibleInScene(joint) || !projected.visible || projected.x<0 || projected.x>innerWidth || projected.y<0 || projected.y>innerHeight){tooltip.hidden=true;return;}
    tooltip.hidden=false;tooltip.dataset.joint=joint.name;
    $('joint-angle-name').textContent=joint.name;$('joint-angle-name').title=joint.name;
    const factor=joint.jointType==='prismatic' || state.radians?1:180/Math.PI,unit=joint.jointType==='prismatic'?'m':state.radians?'rad':'°';
    const button=$('joint-angle-readout');button.textContent=`${((joint.angle || 0)*factor).toFixed(joint.jointType==='prismatic'?3:state.radians?3:2)} ${unit}`;
    button.disabled=jointAngleUI.dragging;button.title=t('editJointAngle');
    button.hidden=jointAngleUI.editing;$('joint-angle-form').hidden=!jointAngleUI.editing;
    tooltip.classList.toggle('editing',jointAngleUI.editing);
    tooltip.style.left=`${Math.max(6,Math.min(innerWidth-tooltip.offsetWidth-6,projected.x+12))}px`;
    tooltip.style.top=`${Math.max(6,Math.min(innerHeight-tooltip.offsetHeight-6,projected.y-20))}px`;
  }
  function queueJointAngleUpdate() {
    if(!activeDragJoint || jointAngleUI.frame)return;
    jointAngleUI.frame=requestAnimationFrame(()=>{jointAngleUI.frame=0;renderJointAngle();});
  }
  function startJointAngleEdit() {
    const joint=activeDragJoint;if(!joint || jointAngleUI.dragging)return;
    stopAnimation();clearTimeout(jointAngleUI.hideTimer);jointAngleUI.editing=true;
    jointAngleUI.factor=joint.jointType==='prismatic' || state.radians?1:180/Math.PI;
    jointAngleUI.unit=joint.jointType==='prismatic'?'m':state.radians?'rad':'°';
    const input=$('joint-angle-input');input.value=String(Number(((joint.angle || 0)*jointAngleUI.factor).toFixed(6)));input.setCustomValidity('');
    input.removeAttribute('min');input.removeAttribute('max');
    if(joint.jointType!=='continuous')for(const [attribute,bound]of [['min',joint.limit.lower],['max',joint.limit.upper]])if(Number.isFinite(bound))input.setAttribute(attribute,bound*jointAngleUI.factor);
    $('joint-angle-unit').textContent=jointAngleUI.unit;$('joint-angle-feedback').textContent='';
    renderJointAngle();input.focus({preventScroll:true});input.select();
  }
  function finishJointAngleEdit() {
    jointAngleUI.editing=false;$('joint-angle-input').setCustomValidity('');
    renderJointAngle();scheduleJointAngleHide();
  }
  function submitJointAngle() {
    const joint=activeDragJoint,input=$('joint-angle-input');
    if(!joint || viewer.robot?.joints[joint.name]!==joint){clearJointAngleUI();return;}
    const value=input.valueAsNumber/jointAngleUI.factor;
    if(!input.value.trim() || !Number.isFinite(value)) {
      input.setCustomValidity(t('invalidJointAngle'));$('joint-angle-feedback').textContent=t('invalidJointAngle');input.reportValidity();return;
    }
    // This direct edit always respects declared limits, including when the toolbar toggle is off.
    const clamped=joint.jointType==='continuous'?value:THREE.MathUtils.clamp(value,joint.limit.lower,joint.limit.upper);
    stopAnimation();clearSurfaceMeasurement();viewer.setJointValue(joint.name,clamped);
    $('joint-angle-feedback').textContent=clamped!==value?t('jointAngleClamped'):'';
    finishJointAngleEdit();$('joint-angle-readout').focus({preventScroll:true});
  }
  $('joint-angle-readout').onclick=startJointAngleEdit;
  $('joint-angle-form').onsubmit=event=>{event.preventDefault();submitJointAngle();};
  $('joint-angle-cancel').onclick=()=>{$('joint-angle-feedback').textContent='';finishJointAngleEdit();$('joint-angle-readout').focus({preventScroll:true});};
  $('joint-angle-input').oninput=event=>{event.target.setCustomValidity('');$('joint-angle-feedback').textContent='';};
  $('joint-angle-input').onkeydown=event=>{if(event.key==='Escape'){event.preventDefault();event.stopPropagation();$('joint-angle-cancel').click();}else if(event.key==='Enter' && !event.isComposing){event.preventDefault();event.stopPropagation();submitJointAngle();}};
  $('joint-angle-tooltip').addEventListener('pointerenter',()=>{jointAngleUI.hovered=true;clearTimeout(jointAngleUI.hideTimer);});
  $('joint-angle-tooltip').addEventListener('pointerleave',()=>{jointAngleUI.hovered=false;scheduleJointAngleHide();});
  for(const type of ['pointerdown','pointerup','dblclick'])$('joint-angle-tooltip').addEventListener(type,event=>event.stopPropagation());
  window.addEventListener('pointerdown',event=>{
    if(!activeDragJoint || event.composedPath().includes($('joint-angle-tooltip')))return;
    const hit=event.composedPath().includes(viewer.renderer.domElement)?visibleModelHit(rayFromPointer(event)):null;
    if(!hit?.link || nearestMovableJoint(hit.link)!==activeDragJoint){clearJointAngleUI();return;}
    if(jointAngleUI.editing){$('joint-angle-feedback').textContent='';finishJointAngleEdit();}
  },true);
  viewer.controls.addEventListener('change',queueJointAngleUpdate);
  viewer.addEventListener('angle-change',queueJointAngleUpdate);
  window.addEventListener('resize',queueJointAngleUpdate);
  const axisUp=new THREE.Vector3(0,1,0),axisWorldPoint=new THREE.Vector3(),axisParentScale=new THREE.Vector3();
  // WebGL often clamps lineWidth to one pixel. Mesh shafts provide a real 3 CSS-pixel stroke.
  function axisShaft(direction,length,color) {
    const shaft=new THREE.Mesh(new THREE.CylinderGeometry(1,1,1,10),helperMaterial(color));
    shaft.renderOrder=21;
    shaft.position.copy(direction).multiplyScalar(length/2);
    shaft.quaternion.setFromUnitVectors(axisUp,direction);shaft.scale.set(.001,length,.001);
    shaft.userData.axisShaft=true;
    shaft.raycast=()=>{};
    shaft.onBeforeRender=function(renderer,scene,camera) {
      this.getWorldPosition(axisWorldPoint);axisWorldPoint.applyMatrix4(camera.matrixWorldInverse);
      this.parent.getWorldScale(axisParentScale);
      const depth=Math.max(camera.near,-axisWorldPoint.z),height=renderer.domElement.clientHeight || 1;
      const radius=depth*Math.tan(THREE.MathUtils.degToRad(camera.fov/2))*3/(height*camera.zoom);
      this.scale.x=this.scale.z=radius/Math.max(.0001,axisParentScale.x);
      this.updateMatrixWorld(true);
    };
    return shaft;
  }
  // Link frames use the URDF convention X=red, Y=green, Z=blue.
  function thickLinkAxes(length) {
    const group=new THREE.Group();group.userData.axisKind='axes';
    for(const [direction,color]of [[new THREE.Vector3(1,0,0),0xff0000],[new THREE.Vector3(0,1,0),0x00ff00],[new THREE.Vector3(0,0,1),0x0000ff]])group.add(axisShaft(direction,length,color));
    return group;
  }
  function helperMaterial(color) {
    return new THREE.MeshBasicMaterial({color,toneMapped:false,transparent:true,depthTest:false,depthWrite:false});
  }
  function helperCone(radius,length,color) {
    const mesh=new THREE.Mesh(new THREE.ConeGeometry(radius,length,16),helperMaterial(color));
    mesh.raycast=()=>{};mesh.renderOrder=22;return mesh;
  }
  function referenceJointAxis(direction,scale,jointType) {
    // Build around local +Y, then rotate the entire glyph to the declared joint axis.
    const group=new THREE.Group();group.userData.axisKind='jointAxes';group.userData.jointType=jointType;
    group.quaternion.setFromUnitVectors(axisUp,direction);
    const length=scale*1.65,headLength=scale*.32;
    group.userData.angleAnchor=[0,length*1.05,0];
    group.userData.labelAnchor=[0,length*.5,0];
    group.userData.tipAnchor=[0,length,0];
    group.add(axisShaft(axisUp,length-headLength,0xff0000));
    const head=helperCone(scale*.14,headLength,0xff0000);
    head.position.y=length-headLength/2;group.add(head);
    if(jointType==='revolute' || jointType==='continuous') {
      const radius=scale*.4,start=-Math.PI/3,sweep=Math.PI*1.68,end=start+sweep;
      const geometry=new THREE.TorusGeometry(radius,scale*.009,8,64,sweep);
      // A positive turn in this XZ plane follows the right-hand rule about +Y.
      geometry.rotateX(-Math.PI/2);geometry.rotateY(start);
      const arc=new THREE.Mesh(geometry,helperMaterial(0x00ff00));
      arc.position.y=length*.86;arc.userData.rotationIndicator=true;arc.raycast=()=>{};arc.renderOrder=23;
      group.add(arc);
      const tangent=new THREE.Vector3(-Math.sin(end),0,-Math.cos(end));
      const tip=helperCone(scale*.06,scale*.15,0x00ff00);
      tip.position.set(radius*Math.cos(end),arc.position.y,-radius*Math.sin(end)).addScaledVector(tangent,-scale*.075);
      tip.quaternion.setFromUnitVectors(axisUp,tangent);tip.userData.rotationArrowhead=true;
      group.add(tip);
    }
    return group;
  }
  function checkerCenterOfMass(radius) {
    const geometry=new THREE.SphereGeometry(radius,32,20);
    const positions=geometry.attributes.position,index=geometry.index;
    const patches=[[],[]];
    // Alternate all eight octants so the sphere stays recognizably black/white from any view.
    for(let offset=0;offset<index.count;offset+=3) {
      let x=0,y=0,z=0;
      for(let corner=0;corner<3;corner++){const vertex=index.getX(offset+corner);x+=positions.getX(vertex);y+=positions.getY(vertex);z+=positions.getZ(vertex);}
      const octant=Number(x<0)+Number(y<0)+Number(z<0);
      patches[octant%2].push(index.getX(offset),index.getX(offset+1),index.getX(offset+2));
    }
    geometry.setIndex([...patches[0],...patches[1]]);geometry.clearGroups();
    geometry.addGroup(0,patches[0].length,0);geometry.addGroup(patches[0].length,patches[1].length,1);
    const sphere=new THREE.Mesh(geometry,[helperMaterial(0xf4f4f4),helperMaterial(0x161616)]);
    sphere.userData.centerOfMassMarker=true;sphere.raycast=()=>{};sphere.renderOrder=30;
    return sphere;
  }
  const originalVisualStyle=new WeakMap();
  function updateInspectionAppearance() {
    const inspecting=state.display.com || state.display.axes || state.display.jointAxes;
    viewer.robot?.traverse(mesh=>{
      if(!mesh.isMesh)return;
      let parent=mesh.parent;
      while(parent && !parent.isURDFVisual && parent!==viewer.robot)parent=parent.parent;
      if(!parent?.isURDFVisual)return;
      for(const material of Array.isArray(mesh.material)?mesh.material:[mesh.material]) {
        if(!originalVisualStyle.has(material))originalVisualStyle.set(material,{opacity:material.opacity,transparent:material.transparent,depthWrite:material.depthWrite});
        const original=originalVisualStyle.get(material),transparent=inspecting || original.transparent;
        if(material.transparent!==transparent){material.transparent=transparent;material.needsUpdate=true;}
        material.opacity=inspecting?Math.min(original.opacity,.32):original.opacity;
        material.depthWrite=inspecting?false:original.depthWrite;
      }
    });
  }
  function buildHelpers() {
    const robot=viewer.robot, size=objectBounds(robot).getSize(new THREE.Vector3()).length()||1, scale=Math.max(.02,size*.07);
    for(const link of Object.values(robot.links))helper(thickLinkAxes(scale),'axes',link);
    for(const joint of Object.values(robot.joints))if(['revolute','continuous','prismatic'].includes(joint.jointType) && joint.axis?.lengthSq())helper(referenceJointAxis(joint.axis.clone().normalize(),scale,joint.jointType),'jointAxes',joint);
    for(const element of nodes(state.xml.documentElement,'link')){
      const link=robot.links[element.getAttribute('name')], inertial=direct(element,'inertial');if(!link||!inertial)continue;
      const origin=direct(inertial,'origin'), mass=Number(direct(inertial,'mass')?.getAttribute('value'));if(!(mass>0))continue;
      const com=new THREE.Group();com.userData.markerKind='com';applyOrigin(com,origin);com.add(checkerCenterOfMass(scale*.14));helper(com,'com',link);
      const tensor=direct(inertial,'inertia');if(!tensor)continue;const get=key=>Number(tensor.getAttribute(key)||0);
      const {values,vectors}=principalAxes([[get('ixx'),get('ixy'),get('ixz')],[get('ixy'),get('iyy'),get('iyz')],[get('ixz'),get('iyz'),get('izz')]]);
      // Full edge lengths of a uniform box: Ix = mass * (height² + depth²) / 12.
      const dimensions=values.map((value,i)=>Math.sqrt(6*(values[(i+1)%3]+values[(i+2)%3]-value)/mass));
      if(values.some(value=>value<0)||dimensions.some(value=>!Number.isFinite(value)||value<=0))continue;
      const box=new THREE.Mesh(new THREE.BoxGeometry(1,1,1),new THREE.MeshBasicMaterial({color:0x6ba3c8,transparent:true,opacity:.24,depthWrite:false}));box.scale.set(...dimensions);
      box.userData.inertiaBox=true;box.raycast=()=>{};
      const edges=new THREE.LineSegments(new THREE.EdgesGeometry(box.geometry),new THREE.LineBasicMaterial({color:0xb8d4e8,transparent:true,opacity:.55,depthWrite:false}));edges.raycast=()=>{};box.add(edges);
      const rotation=new THREE.Matrix4().set(vectors[0][0],vectors[0][1],vectors[0][2],0,vectors[1][0],vectors[1][1],vectors[1][2],0,vectors[2][0],vectors[2][1],vectors[2][2],0,0,0,0,1);box.quaternion.setFromRotationMatrix(rotation);
      const group=new THREE.Group();group.userData.markerKind='inertia';applyOrigin(group,origin);group.add(box);helper(group,'inertia',link);
    }
  }
  function applyDisplay(){setHoveredLink(null);clearSurfaceSnap();const d=state.display;viewer.showCollision=d.collision;viewer.displayShadow=d.shadow;viewer.directionalLight.castShadow=d.shadow;viewer.plane.visible=d.shadow;grid.visible=d.grid;viewer.ambientLight.intensity=d.lighting?.65:1.8;viewer.directionalLight.intensity=d.lighting?Math.PI:0;viewer.robot?.traverse(child=>{if(child.isURDFVisual)child.visible=d.visual;});state.helpers.forEach(({object,kind})=>object.visible=d[kind] || (kind==='jointAxes' && (object.parent===activeDragJoint || (statics?.enabled && !object.userData.staticsInspectionHidden))));updateInspectionAppearance();refreshModelHover();viewer.redraw();statics?.requestProjection();}
  $$('[data-display]').forEach(button=>button.onclick=()=>{const key=button.dataset.display;state.display[key]=!state.display[key];button.classList.toggle('active',state.display[key]);button.setAttribute('aria-pressed',String(state.display[key]));applyDisplay();});
  function updateCameraFrame() {
    let left=0,right=0,top=72,bottom=40;
    if(!$('motion-timeline').hidden)bottom=Math.max(bottom,innerHeight-$('motion-timeline').getBoundingClientRect().top+12);
    const visible=panel=>!panel.hidden && getComputedStyle(panel).display!=='none';
    if(innerWidth>600) {
      const sidebar=$('workspace-sidebar'),editor=$('editor-panel');
      if(visible(sidebar))left=sidebar.getBoundingClientRect().right;
      if(visible(editor))right=innerWidth-editor.getBoundingClientRect().left;
    }else{
      for(const panel of [$('workspace-sidebar'),$('editor-panel')])if(visible(panel))bottom=Math.max(bottom,innerHeight-panel.getBoundingClientRect().top+8);
    }
    const width=Math.max(innerWidth-left-right-32,innerWidth*.15),height=Math.max(innerHeight-top-bottom,innerHeight*.25);
    viewer.camera.setViewOffset(innerWidth,innerHeight,(right-left)/2,(bottom-top)/2,innerWidth,innerHeight);
    viewer.redraw();
    queueLinkMeasurementUpdate();
    queueJointAngleUpdate();statics?.requestProjection();
    return {width:width/innerWidth,height:height/innerHeight};
  }
  function fitView(bounds=null) {
    const box=bounds?.isBox3?bounds:viewer.robot?objectBounds(viewer.robot):new THREE.Box3(new THREE.Vector3(-.7,0,-.7),new THREE.Vector3(.7,1,.7));
    const center=box.isEmpty()?new THREE.Vector3(0,.3,0):box.getCenter(new THREE.Vector3()), size=box.isEmpty()?1.5:box.getSize(new THREE.Vector3()).length();
    const available=updateCameraFrame();
    const distance=Math.max(size, .25)/(2*Math.tan(THREE.MathUtils.degToRad(viewer.camera.fov/2)))*Math.max(1.2,1/(viewer.camera.aspect*available.width),1/available.height);
    viewer.controls.target.copy(center);viewer.camera.position.copy(center).add(new THREE.Vector3(1,.75,1).normalize().multiplyScalar(distance));viewer.controls.maxDistance=Math.max(50,distance*8);viewer.camera.far=Math.max(1000,distance*12);viewer.camera.updateProjectionMatrix();viewer.controls.update();
    if(!bounds?.isBox3 && !motion?.groundLocked && viewer.robot && !box.isEmpty()){grid.position.y=box.min.y-.003;viewer.plane.position.y=box.min.y-.002;viewer._updateEnvironment();}viewer.redraw();updateGizmo();
  }
  $('fit-view').onclick=fitView;
  $('up-select').onchange=()=>{viewer.up=$('up-select').value;fitView();queueLinkMeasurementUpdate();statics?.orientationChanged();motion?.orientationChanged();};
  function updateGizmo(){const lines=$('gizmo-lines'),labels=$('gizmo-labels');lines.replaceChildren();labels.replaceChildren();const inverse=viewer.camera.quaternion.clone().invert(),rotation=viewer.world.quaternion;
    const items=[['X',new THREE.Vector3(1,0,0),'#ff5269'],['Y',new THREE.Vector3(0,1,0),'#a4db3b'],['Z',new THREE.Vector3(0,0,1),'#439bff']].map(([name,direction,color])=>({name,direction,color,point:direction.clone().applyQuaternion(rotation).applyQuaternion(inverse)})).sort((a,b)=>a.point.z-b.point.z);
    for(const {name,direction,color,point}of items){const x=55+point.x*33,y=50-point.y*33;const line=document.createElementNS('http://www.w3.org/2000/svg','line');for(const [key,val]of Object.entries({x1:55,y1:50,x2:x,y2:y,stroke:color,'stroke-width':1.5}))line.setAttribute(key,val);lines.append(line);const g=document.createElementNS(line.namespaceURI,'g');g.setAttribute('role','button');g.setAttribute('tabindex','0');g.setAttribute('aria-label',`${name} view`);const circle=document.createElementNS(line.namespaceURI,'circle');for(const [key,val]of Object.entries({cx:x,cy:y,r:10,fill:color}))circle.setAttribute(key,val);const text=document.createElementNS(line.namespaceURI,'text');text.setAttribute('x',x);text.setAttribute('y',y+4);text.setAttribute('text-anchor','middle');text.textContent=name;g.append(circle,text);const click=()=>{const distance=viewer.camera.position.distanceTo(viewer.controls.target);const axis=direction.clone().applyQuaternion(rotation);viewer.camera.position.copy(viewer.controls.target).addScaledVector(axis,distance);if(Math.abs(axis.y)>.99)viewer.camera.position.z+=.0001;viewer.controls.update();viewer.redraw();updateGizmo();};g.onclick=click;g.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();click();}};labels.append(g);}
  }
  viewer.controls.addEventListener('change',()=>{updateGizmo();viewer.redraw();});
  updateGizmo();
  const linkMeasurement={enabled:false,pinned:false,links:[],unit:'mm',distance:0,delta:null,points:[],frame:0,lastStatus:'',annotations:null};
  const surfaceMeasurement={anchors:[],unit:'mm',distance:0,delta:null,points:[],annotations:null,epoch:0,lastStatus:''};
  const measurementFeatures=new URDFMeasurementFeatures(THREE);
  const surfaceSnap={link:null,sets:[],promise:null,epoch:0,pointer:null,hit:null,candidates:[],active:null,loading:false};
  let surfacePickQueue=Promise.resolve();
  function linkMeasureStatus(text) {
    linkMeasurement.lastStatus=text;status(text);
  }
  function measurementValue(meters,space=true,unit=linkMeasurement.unit) {
    const multiplier=unit==='mm'?1000:1,digits=unit==='mm'?1:4;
    let value=meters*multiplier;
    if(Math.abs(value)<.5*Math.pow(10,-digits))value=0;
    return `${value.toFixed(digits)}${space?' ':''}${unit}`;
  }
  function refreshMeasuredNodes() {
    for(const {object,element}of structureGraph?.nodes || []) {
      const index=linkMeasurement.links.indexOf(object);
      element.classList.toggle('measured-link',index!==-1);
      element.setAttribute('aria-pressed',String(index!==-1 || element.classList.contains('selected')));
      if(index!==-1)element.setAttribute('data-measure-endpoint',index===0?'A':'B');
      else element.removeAttribute('data-measure-endpoint');
    }
  }
  function updateLinkMeasureUI() {
    const measurement=linkMeasurement;
    $('graph-measure').classList.toggle('active',measurement.pinned);
    $('graph-measure').setAttribute('aria-pressed',String(measurement.pinned));
    $('link-measure-panel').hidden=!measurement.enabled;
    $('structure-panel').classList.toggle('measuring-links',measurement.enabled);
    for(const [index,id]of [[0,'link-measure-a'],[1,'link-measure-b']]) {
      const text=`${index===0?'A':'B'}: ${measurement.links[index]?.name || '—'}`;
      $(id).textContent=text;$(id).title=text;
    }
    const complete=measurement.links.length===2;
    $('link-measure-distance').textContent=complete?measurementValue(measurement.distance):'—';
    for(const [index,axis]of ['x','y','z'].entries()) {
      const value=complete?measurementValue(measurement.delta.getComponent(index)):'—';
      $(`link-measure-d${axis}`).textContent=`Δ${axis.toUpperCase()} ${value}`;
      $(`link-measure-d${axis}`).title=`Δ${axis.toUpperCase()} ${value}`;
    }
    const hint=$('structure-panel').querySelector('.structure-hint');
    hint.textContent=measurement.enabled?(measurement.pinned?t(measurement.links.length===0?'pickLinkA':measurement.links.length===1?'pickLinkB':'linkMeasureAgain'):t('ctrlLinkHint')):t('structureHint');
    refreshMeasuredNodes();
  }
  function clearLinkMeasurement(exitMode=false) {
    const measurement=linkMeasurement;
    measurement.links=[];measurement.points=[];measurement.delta=null;measurement.distance=0;
    if(exitMode){measurement.enabled=false;measurement.pinned=false;}
    $('link-measure-overlay').replaceChildren();$('link-measure-overlay').setAttribute('hidden','');
    measurement.annotations=null;
    updateLinkMeasureUI();
    if($('status-text').textContent===measurement.lastStatus)status(state.context?`${t('modelLoaded')} · ${state.context.name}`:t('ready'));
    measurement.lastStatus='';
  }
  function setLinkMeasureMode(enabled,pinned=true) {
    if(enabled && !viewer.robot)return;
    clearSurfaceMeasurement();clearLinkMeasurement(!enabled);
    linkMeasurement.enabled=enabled;linkMeasurement.pinned=enabled && pinned;
    if(enabled){clearSelection();linkMeasureStatus(t('pickLinkA'));}
    updateLinkMeasureUI();
  }
  function chooseMeasuredLink(object) {
    if(!object.isURDFLink){linkMeasureStatus(t('pickLinkOnly'));return;}
    clearSurfaceMeasurement();
    if(linkMeasurement.links.length===2)clearLinkMeasurement();
    if(linkMeasurement.links.includes(object)){linkMeasureStatus(t('pickOtherLink'));return;}
    clearSelection();revealStructureLink(object);linkMeasurement.links.push(object);
    updateLinkMeasurement();
    if(linkMeasurement.links.length===1)linkMeasureStatus(`${t('pickLinkB')} · A: ${object.name}`);
    else linkMeasureStatus(`${t('linkMeasure')}: ${linkMeasurement.links[0].name} → ${object.name}`);
  }
  function selectStructureNode(object,row,event) {
    if(event?.ctrlKey || event?.metaKey) {
      if(!object.isURDFLink){linkMeasureStatus(t('pickLinkOnly'));return;}
      if(!linkMeasurement.enabled)setLinkMeasureMode(true,false);
      chooseMeasuredLink(object);return;
    }
    if(linkMeasurement.pinned)chooseMeasuredLink(object);
    else selectObject(object,row);
  }
  function makeMeasurementAnnotations() {
    const overlay=$('link-measure-overlay'),lines=[],labels=[];
    for(const [index,color]of ['#ff3030','#12e52c','#4141ff','#ffff00'].entries()) {
      const line=svgElement('line',{stroke:color,'stroke-width':index===3?3.4:2.8,'stroke-linecap':'round'});
      if(index===3)line.setAttribute('stroke-dasharray','8 6');
      overlay.append(line);lines.push(line);
    }
    const endpoints=[0,1].map(()=>{
      const circle=svgElement('circle',{r:7,fill:'#ffff00',stroke:'#202020a0','stroke-width':1.5});overlay.append(circle);return circle;
    });
    for(const color of ['#ff3030','#12e52c','#4141ff','#ffff00']) {
      const label=svgElement('text',{fill:color,'text-anchor':'middle','class':'measurement-label'});
      overlay.append(label);labels.push(label);
    }
    return {lines,labels,endpoints};
  }
  function projectMeasurementPoint(point) {
    const view=point.clone().applyMatrix4(viewer.camera.matrixWorldInverse),projected=point.clone().project(viewer.camera);
    return {x:(projected.x+1)*innerWidth/2,y:(1-projected.y)*innerHeight/2,
      visible:-view.z>=viewer.camera.near && -view.z<=viewer.camera.far && Number.isFinite(projected.x) && Number.isFinite(projected.y)};
  }
  function clippedMeasurementSegment(a,b) {
    let start=a.clone(),end=b.clone();
    for(const [sign,limit]of [[-1,viewer.camera.near],[1,-viewer.camera.far]]) {
      const startView=start.clone().applyMatrix4(viewer.camera.matrixWorldInverse),endView=end.clone().applyMatrix4(viewer.camera.matrixWorldInverse);
      const da=sign*startView.z-limit,db=sign*endView.z-limit;
      if(da<0 && db<0)return null;
      if(da<0)start.lerp(end,da/(da-db));
      else if(db<0)end.copy(start.clone().lerp(end,da/(da-db)));
    }
    return [projectMeasurementPoint(start),projectMeasurementPoint(end)];
  }
  function renderMeasurementProjection(measurement=linkMeasurement) {
    const overlay=$('link-measure-overlay');
    if(measurement.points.length===0){overlay.setAttribute('hidden','');return;}
    viewer.camera.updateMatrixWorld(true);
    overlay.removeAttribute('hidden');overlay.setAttribute('viewBox',`0 0 ${innerWidth} ${innerHeight}`);
    measurement.annotations ||= makeMeasurementAnnotations();
    const annotations=measurement.annotations,points=measurement.points,complete=points.length===4;
    const segments=complete?[[points[0],points[1]],[points[1],points[2]],[points[2],points[3]],[points[0],points[3]]]:[];
    annotations.lines.forEach((line,index)=>{
      const projected=segments[index]?clippedMeasurementSegment(...segments[index]):null;
      line.style.display=projected?'':'none';
      if(projected)for(const [key,value]of Object.entries({x1:projected[0].x,y1:projected[0].y,x2:projected[1].x,y2:projected[1].y}))line.setAttribute(key,value);
    });
    annotations.endpoints.forEach((circle,index)=>{
      const position=index===0?points[0]:complete?points[3]:null;
      const projected=position?projectMeasurementPoint(position):null;
      circle.style.display=projected?.visible?'':'none';
      if(projected){circle.setAttribute('cx',projected.x);circle.setAttribute('cy',projected.y);}
    });
    const used=[],offsets=[[0,-17],[-24,0],[0,24],[24,-12]];
    annotations.labels.forEach((label,index)=>{
      if(!complete){label.style.display='none';return;}
      const text=index===3?measurementValue(measurement.distance,false,measurement.unit):`Δ${'XYZ'[index]} ${measurementValue(measurement.delta.getComponent(index),false,measurement.unit)}`;
      label.textContent=text;
      const midpoint=segments[index][0].clone().lerp(segments[index][1],.5),projected=projectMeasurementPoint(midpoint);
      if(!projected.visible || projected.x<-100 || projected.x>innerWidth+100 || projected.y<-100 || projected.y>innerHeight+100){label.style.display='none';return;}
      label.style.display='';
      const width=label.getComputedTextLength()+12,x=Math.max(width/2+6,Math.min(innerWidth-width/2-6,projected.x+offsets[index][0]));
      let y=Math.max(25,Math.min(innerHeight-12,projected.y+offsets[index][1]));
      // Separate labels when a component is zero or several segments project onto one point.
      for(const shift of [0,26,-26,52,-52,78,-78,104,-104]){const candidate=y+shift;if(candidate>=25 && candidate<=innerHeight-12 && !used.some(other=>Math.abs(x-other.x)<(width+other.width)/2 && Math.abs(candidate-other.y)<25)){y=candidate;break;}}
      used.push({x,y,width});label.setAttribute('x',x);label.setAttribute('y',y);
    });
  }
  function updateLinkMeasurement() {
    if(surfaceMeasurement.anchors.length)updateSurfaceMeasurement();
    const measurement=linkMeasurement;
    if(!measurement.enabled || !measurement.links.length)return;
    viewer.world.updateWorldMatrix(true,true);
    const start=measurement.links[0].getWorldPosition(new THREE.Vector3());
    measurement.points=[start];
    if(measurement.links.length===2) {
      const end=measurement.links[1].getWorldPosition(new THREE.Vector3());
      measurement.delta=end.clone().sub(start);measurement.distance=measurement.delta.length();
      measurement.points=[start,new THREE.Vector3(end.x,start.y,start.z),new THREE.Vector3(end.x,end.y,start.z),end];
    }
    updateLinkMeasureUI();renderMeasurementProjection();
  }
  function queueLinkMeasurementUpdate() {
    if((!linkMeasurement.enabled && !surfaceMeasurement.anchors.length) || linkMeasurement.frame)return;
    linkMeasurement.frame=requestAnimationFrame(()=>{linkMeasurement.frame=0;updateLinkMeasurement();});
  }
  $('graph-measure').onclick=()=>setLinkMeasureMode(!linkMeasurement.pinned);
  $('link-measure-clear').onclick=()=>{clearLinkMeasurement();linkMeasureStatus(t('pickLinkA'));};
  $('link-measure-unit').onchange=()=>{linkMeasurement.unit=$('link-measure-unit').value;updateLinkMeasureUI();renderMeasurementProjection();};
  viewer.controls.addEventListener('change',queueLinkMeasurementUpdate);
  viewer.addEventListener('angle-change',queueLinkMeasurementUpdate);
  window.addEventListener('resize',queueLinkMeasurementUpdate);

  function clearMeasurement(){clearSurfaceMeasurement();clearLinkMeasurement(true);}
  function clearSurfaceMeasurement(cancelPending=true){
    if(cancelPending)surfaceMeasurement.epoch++;
    if(surfaceMeasurement.anchors.length){$('link-measure-overlay').replaceChildren();$('link-measure-overlay').setAttribute('hidden','');}
    surfaceMeasurement.anchors=[];surfaceMeasurement.points=[];surfaceMeasurement.delta=null;surfaceMeasurement.distance=0;surfaceMeasurement.annotations=null;
    clearSurfaceSnap();
    $('measure-result').hidden=true;
    if($('status-text').textContent===surfaceMeasurement.lastStatus)status(state.context?`${t('modelLoaded')} · ${state.context.name}`:t('ready'));
    surfaceMeasurement.lastStatus='';
    viewer.redraw();
  }
  $('measure-result').querySelector('button').onclick=clearMeasurement;
  function snapPointName(candidate) {
    return `${t(candidate.kind)}${candidate.angle===null || candidate.angle===undefined?'':` ${candidate.angle}°`}${candidate.fitted?` (${t('meshFitted')})`:''}`;
  }
  function clearSurfaceSnap() {
    surfaceSnap.epoch++;surfaceSnap.link=null;surfaceSnap.sets=[];surfaceSnap.promise=null;surfaceSnap.pointer=null;surfaceSnap.hit=null;surfaceSnap.candidates=[];surfaceSnap.active=null;surfaceSnap.loading=false;
    $('measure-snap-overlay').replaceChildren();$('measure-snap-overlay').setAttribute('hidden','');$('measure-snap-hint').hidden=true;
    if(viewer.renderer.domElement.style.cursor==='crosshair')viewer.renderer.domElement.style.cursor='';
  }
  function prepareSnapLink(link) {
    if(surfaceSnap.link===link && surfaceSnap.promise)return surfaceSnap.promise;
    const epoch=++surfaceSnap.epoch,meshes=[];
    surfaceSnap.link=link;surfaceSnap.sets=[];surfaceSnap.loading=true;
    for(const child of link.children)if((child.isURDFVisual || child.isURDFCollider) && visibleInScene(child))child.traverseVisible(mesh=>{
      if(mesh.isMesh && mesh.geometry?.getAttribute('position'))meshes.push(mesh);
    });
    surfaceSnap.promise=Promise.all(meshes.map(async mesh=>({mesh,features:await measurementFeatures.get(mesh.geometry)}))).then(sets=>{
      if(surfaceSnap.epoch===epoch){surfaceSnap.sets=sets;surfaceSnap.loading=false;if(surfaceSnap.pointer)renderSurfaceSnap();}
      return sets;
    }).catch(error=>{
      // A malformed mesh must not disable ordinary surface-point measurement.
      console.warn('Measurement feature extraction failed',error);
      if(surfaceSnap.epoch===epoch){surfaceSnap.loading=false;surfaceSnap.sets=[];if(surfaceSnap.pointer)renderSurfaceSnap();}
      return [];
    });
    return surfaceSnap.promise;
  }
  function renderSurfaceSnap() {
    const pointer=surfaceSnap.pointer,link=surfaceSnap.link;
    if(!pointer || !link)return;
    viewer.world.updateWorldMatrix(true,true);viewer.camera.updateMatrixWorld(true);
    const all=[];
    for(const {mesh,features}of surfaceSnap.sets)if(visibleInScene(mesh))all.push(...measurementFeatures.worldCandidates(mesh,features));
    // Imported free-form surfaces still offer the three actual hit-triangle vertices.
    const hit=surfaceSnap.hit;
    if(hit?.face && hit.link===link && !['BoxGeometry','CylinderGeometry','SphereGeometry'].includes(hit.object.geometry.type))for(const id of [hit.face.a,hit.face.b,hit.face.c]) {
      const local=new THREE.Vector3().fromBufferAttribute(hit.object.geometry.getAttribute('position'),id);
      all.push({object:hit.object,local,world:hit.object.localToWorld(local.clone()),kind:'meshVertex',fitted:false});
    }
    const origin=link.getWorldPosition(new THREE.Vector3());
    all.push({object:link,local:new THREE.Vector3(),world:origin,kind:'linkOrigin',fitted:false});
    const keys=new Set(),candidates=[];
    for(const candidate of all) {
      const projected=projectMeasurementPoint(candidate.world);
      if(!projected.visible || projected.x<0 || projected.x>innerWidth || projected.y<0 || projected.y>innerHeight)continue;
      const key=candidate.world.toArray().map(n=>n.toFixed(7)).join(',');
      if(keys.has(key))continue;keys.add(key);
      candidates.push({...candidate,projected,distance:Math.hypot(projected.x-pointer.clientX,projected.y-pointer.clientY),depth:viewer.camera.position.distanceTo(candidate.world)});
    }
    const priority={circleCenter:0,geometryCenter:0,sphereCenter:0,linkOrigin:1,quadrant:1,rectangleCenter:2,rectangleVertex:3,boundsCenter:3,meshVertex:4};
    // A marker directly under the pointer wins. Depth only breaks true screen ties.
    candidates.sort((a,b)=>Math.abs(a.distance-b.distance)>.05 ? a.distance-b.distance : (priority[a.kind]-priority[b.kind] || a.depth-b.depth));
    // Dense CAD tessellation can yield many coincident-looking corners. Keep the
    // closest point selectable while leaving breathing room between shown markers.
    const active=!pointer.altKey && candidates[0]?.distance<=18?candidates[0]:null;
    const shown=active?[active]:[];
    // Keep small tessellated rectangles from crowding out centers and quadrant points.
    for(const candidate of candidates.slice().sort((a,b)=>(priority[a.kind]-priority[b.kind]) || a.distance-b.distance)) {
      if(shown.some(other=>Math.hypot(candidate.projected.x-other.projected.x,candidate.projected.y-other.projected.y)<10))continue;
      shown.push(candidate);if(shown.length===5)break;
    }
    surfaceSnap.candidates=shown;
    surfaceSnap.active=active;
    const overlay=$('measure-snap-overlay');overlay.replaceChildren();overlay.removeAttribute('hidden');overlay.setAttribute('viewBox',`0 0 ${innerWidth} ${innerHeight}`);
    for(const candidate of surfaceSnap.candidates) {
      const active=candidate===surfaceSnap.active;
      const marker=svgElement('circle',{cx:candidate.projected.x,cy:candidate.projected.y,r:active?8:4,
        class:`snap-point${active?' active':''}`,'data-kind':candidate.kind,'data-fitted':String(candidate.fitted),'data-link':link.name});
      overlay.append(marker);
      if(active)overlay.append(svgElement('path',{d:`M ${candidate.projected.x-12} ${candidate.projected.y} h 24 M ${candidate.projected.x} ${candidate.projected.y-12} v 24`,class:'snap-crosshair'}));
    }
    const hint=$('measure-snap-hint');
    hint.textContent=surfaceSnap.loading?t('snapLoading'):`${active?snapPointName(active):t('snapFree')} · ${link.name} · ${t('snapHint')}`;
    hint.classList.toggle('snapped',!!active);hint.hidden=false;
    hint.style.left=`${Math.max(8,Math.min(innerWidth-hint.offsetWidth-8,(active?.projected.x ?? pointer.clientX)+18))}px`;
    hint.style.top=`${Math.max(75,Math.min(innerHeight-hint.offsetHeight-12,(active?.projected.y ?? pointer.clientY)+20))}px`;
    viewer.renderer.domElement.style.cursor='crosshair';
  }
  function updateSurfaceSnap(event) {
    if(!viewer.robot || !event.shiftKey){clearSurfaceSnap();return;}
    const pointer={clientX:event.clientX,clientY:event.clientY,shiftKey:event.shiftKey,ctrlKey:event.ctrlKey,metaKey:event.metaKey,altKey:event.altKey};
    const hit=visibleModelHit(rayFromPointer(pointer));
    let link=hit?.link;
    if(!link && surfaceSnap.link && visibleInScene(surfaceSnap.link)) {
      // Keep a hole's center and internal center selectable inside the hovered link's projected bounds.
      const box=selectedBounds(surfaceSnap.link),xs=[],ys=[];
      if(!box.isEmpty())for(const x of [box.min.x,box.max.x])for(const y of [box.min.y,box.max.y])for(const z of [box.min.z,box.max.z]) {
        const p=projectMeasurementPoint(new THREE.Vector3(x,y,z));if(p.visible){xs.push(p.x);ys.push(p.y);}
      }
      if(xs.length && pointer.clientX>=Math.min(...xs)-20 && pointer.clientX<=Math.max(...xs)+20 && pointer.clientY>=Math.min(...ys)-20 && pointer.clientY<=Math.max(...ys)+20)link=surfaceSnap.link;
    }
    if(!link){clearSurfaceSnap();return;}
    prepareSnapLink(link);surfaceSnap.pointer=pointer;surfaceSnap.hit=hit;renderSurfaceSnap();
  }
  function updateSurfaceMeasurement() {
    const measurement=surfaceMeasurement;
    if(!measurement.anchors.length)return;
    viewer.world.updateWorldMatrix(true,true);
    const [start,end]=measurement.anchors.map(anchor=>anchor.object.localToWorld(anchor.local.clone()));
    measurement.points=[start];
    if(end) {
      measurement.delta=end.clone().sub(start);measurement.distance=measurement.delta.length();
      measurement.points=[start,new THREE.Vector3(end.x,start.y,start.z),new THREE.Vector3(end.x,end.y,start.z),end];
    }
    $('measure-result').hidden=false;
    const complete=measurement.anchors.length===2;
    $('surface-measure-distance').textContent=complete?measurementValue(measurement.distance,true,measurement.unit):'—';
    measurement.anchors.forEach((anchor,index)=>$(`surface-measure-${index===0?'a':'b'}`).textContent=`${index===0?'A':'B'}: ${anchor.link.name} · ${snapPointName(anchor)}`);
    if(!complete)$('surface-measure-b').textContent=`B: ${t('pickSecond')}`;
    for(const [index,axis]of ['x','y','z'].entries())$(`surface-measure-d${axis}`).textContent=`Δ${axis.toUpperCase()} ${complete?measurementValue(measurement.delta.getComponent(index),true,measurement.unit):'—'}`;
    renderMeasurementProjection(measurement);
  }
  $('surface-measure-unit').onchange=()=>{surfaceMeasurement.unit=$('surface-measure-unit').value;updateSurfaceMeasurement();};
  viewer.renderer.domElement.addEventListener('pointerdown',event=>{
    if(event.button!==0)return;
    if(!event.shiftKey) {
      if(!event.ctrlKey && !event.metaKey && !linkMeasurement.pinned)return;
      const hit=visibleModelHit(rayFromPointer(event));
      if(!hit?.link) {
        if(event.ctrlKey || event.metaKey){event.stopImmediatePropagation();event.preventDefault();}
        return;
      }
      event.stopImmediatePropagation();event.preventDefault();
      stopAnimation();clearSurfaceSnap();
      if(!linkMeasurement.enabled)setLinkMeasureMode(true,false);
      chooseMeasuredLink(hit.link);return;
    }
    event.stopImmediatePropagation();event.preventDefault();if(!viewer.robot)return;
    stopAnimation();
    const version=state.version,epoch=surfaceMeasurement.epoch;
    const pointer={clientX:event.clientX,clientY:event.clientY,shiftKey:event.shiftKey,ctrlKey:event.ctrlKey,metaKey:event.metaKey,altKey:event.altKey};
    lastModelPointer=pointer;
    // Preserve click order while a large imported mesh is still being analyzed.
    surfacePickQueue=surfacePickQueue.then(async()=>{
      if(version!==state.version || epoch!==surfaceMeasurement.epoch)return;
      updateSurfaceSnap(pointer);
      if(!pointer.altKey && surfaceSnap.promise)await surfaceSnap.promise;
      if(version!==state.version || epoch!==surfaceMeasurement.epoch)return;
      updateSurfaceSnap(pointer);
      const candidate=surfaceSnap.active,hit=surfaceSnap.hit;
      if(!candidate && !hit)return;
      const anchor=candidate?{...candidate,local:candidate.local.clone(),link:surfaceSnap.link}:
        {object:hit.object,local:hit.object.worldToLocal(hit.point.clone()),link:hit.link,kind:'surfacePoint',fitted:false};
      if(linkMeasurement.enabled || linkMeasurement.links.length)clearLinkMeasurement(true);
      if(surfaceMeasurement.anchors.length===2)clearSurfaceMeasurement(false);
      surfaceMeasurement.anchors.push(anchor);updateSurfaceMeasurement();
      surfaceMeasurement.lastStatus=surfaceMeasurement.anchors.length===1?t('pickSecond'):`${t('measure')}: ${measurementValue(surfaceMeasurement.distance,true,surfaceMeasurement.unit)}`;
      status(surfaceMeasurement.lastStatus);
      if(!lastModelPointer?.shiftKey)clearSurfaceSnap();
    }).catch(error=>{console.warn('Surface measurement failed',error);});
  },true);
  for(const type of ['keydown','keyup'])window.addEventListener(type,event=>{
    if(!['Shift','Control','Meta','Alt'].includes(event.key) || !lastModelPointer)return;
    if(/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName))return;
    Object.assign(lastModelPointer,{shiftKey:event.shiftKey,ctrlKey:event.ctrlKey,metaKey:event.metaKey,altKey:event.altKey});
    if(event.shiftKey){setHoveredLink(null);updateJointHover(null);updateSurfaceSnap(lastModelPointer);}
    else {clearSurfaceSnap();refreshModelHover();}
  });

  let highlightedSource=null;
  function highlightSource(text) {
    if(highlightedSource===text)return;
    highlightedSource=text;
    const fragment=document.createDocumentFragment();
    const tokens=/<!--[\s\S]*?(?:-->|$)|"[^"]*"|'[^']*'|<\?xml|<\/?[\w:.-]+|[A-Za-z_:][\w:.-]*(?=\s*=)|\?>|\/?>/g;
    let end=0;
    for(const match of text.matchAll(tokens)) {
      if(match.index>end)fragment.append(document.createTextNode(text.slice(end,match.index)));
      const value=match[0],span=document.createElement('span');
      span.className=value.startsWith('<!--')?'xml-comment':/^["']/.test(value)?'xml-value':value.startsWith('<')||value.endsWith('>')?'xml-tag':'xml-attribute';
      span.textContent=value;fragment.append(span);end=match.index+value.length;
    }
    fragment.append(document.createTextNode(text.slice(end)+'\n'));
    $('source-highlight').replaceChildren(fragment);
    syncEditorScroll();
  }
  function syncEditorScroll() {
    $('line-numbers').scrollTop=$('source').scrollTop;
    $('source-highlight').style.transform=`translate(${- $('source').scrollLeft}px,${- $('source').scrollTop}px)`;
    if(locatedSource) {
      const {firstLine,lastLine,lineHeight,padding}=locatedSource;
      Object.assign($('source-locate-highlight').style,{top:`${padding+firstLine*lineHeight-$('source').scrollTop}px`,height:`${(lastLine-firstLine+1)*lineHeight}px`});
    }
  }
  function updateEditor() {
    const text=$('source').value;highlightSource(text);
    const lineCount=text.split('\n').length;$('line-numbers').textContent=Array.from({length:lineCount},(_,i)=>i+1).join('\n');$('line-numbers').scrollTop=$('source').scrollTop;
    const before=text.slice(0,$('source').selectionStart),line=before.split('\n').length,column=before.length-before.lastIndexOf('\n');$('cursor-position').textContent=`Ln ${line}, Col ${column}`;
    const source=state.context?.sourceFile,saved=text===source?.savedText,applied=text===state.baseline;
    $('save-state').classList.toggle('dirty',!saved&&!!text);$('save-state').removeAttribute('data-i18n');
    $('save-state').textContent=state.saving?t('saving'):!text?t('emptyFile'):saved?(source.written?(applied?t('saved'):t('savedUnapplied')):t('applied')):applied?t('appliedUnsaved'):t('modified');
    $('filename').readOnly=!!source&&source.savedText!==null;
    $('reload').disabled=state.saving||!text.trim();$('save').disabled=state.saving||!text.trim();
    $('save').textContent=t(state.saving?'saving':'save');
    $('save').title=source&&source.savedText!==null&&!source.handle?`${t('chooseSource')}: ${source.name}`:'Ctrl + S';
  }
  $('source').oninput=()=>{cancelLinkNavigation();$('editor-error').hidden=true;updateEditor();};$('source').onscroll=syncEditorScroll;$('source').onkeyup=updateEditor;$('source').onclick=updateEditor;
  $('source').onkeydown=event=>{if(event.key==='Tab'){event.preventDefault();const source=$('source');source.setRangeText('  ',source.selectionStart,source.selectionEnd,'end');source.dispatchEvent(new Event('input'));}};
  $('reload').onclick=async()=>{if(state.saving)return;try{await loadText($('source').value,state.context||{name:$('filename').value||'newfile.urdf',base:new URL('.',location.href).href},{replaceEditor:false});}catch(error){fail(error);}};
  $('save').onclick=async()=>{
    const text=$('source').value;if(state.saving||!text.trim())return;
    const context=state.context,source=context?.sourceFile||{name:$('filename').value||'newfile.urdf',handle:null,savedText:null,written:false};
    state.saving=true;$('editor-error').hidden=true;updateEditor();
    try {
      validate(text);
      let name=$('filename').value.trim().replace(/[\\/:*?"<>|]/g,'_')||'robot.urdf';if(!/\.(urdf|xml)$/i.test(name))name+='.urdf';
      await sourceFiles.save(source,text,name);
      if(context){context.sourceFile=source;context.name=source.name;}
      else state.context={name:source.name,sourceFile:source,base:new URL('.',location.href).href};
      $('filename').value=source.name;status(`${t('saveHint')} · ${source.name}`);
    }catch(error){
      if(error.name!=='AbortError'){$('editor-error').hidden=false;$('editor-error').textContent=error.message||String(error);status(`${t('saveFailed')} · ${error.message||error}`,'error');}
    }finally{state.saving=false;updateEditor();}
  };
  window.addEventListener('beforeunload',event=>{if(hasEdits()){event.preventDefault();event.returnValue='';}});
  window.addEventListener('keydown',event=>{if(motion?.handleShortcut(event))return;const editing=/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName);if((event.ctrlKey||event.metaKey)&&event.key==='Enter'){event.preventDefault();$('reload').click();}else if((event.ctrlKey||event.metaKey)&&event.key==='s'){event.preventDefault();$('save').click();}else if(!editing&&event.key.toLowerCase()==='f'){fitView();}else if(!editing&&event.key==='Escape'){clearJointAngleUI();clearMeasurement();clearSelection();viewer.redraw();}});

  const leftPanels=['files','joints','structure','motion'];
  function refreshWorkspace() {
    const mobile=innerWidth<=600,sidebar=$('workspace-sidebar');
    sidebar.classList.toggle('collapsed',state.sidebarCollapsed);
    sidebar.hidden=mobile && state.mobile==='editor';
    document.body.classList.toggle('sidebar-collapsed',state.sidebarCollapsed);
    for(const name of leftPanels) {
      const panel=$(`${name}-panel`),selected=state.activeTab===name,visible=selected&&!state.sidebarCollapsed&&!sidebar.hidden;
      panel.hidden=!visible;
      panel.classList.remove('collapsed');
      panel.classList.toggle('mobile-active',visible);
      panel.classList.toggle('mobile-inactive',!visible);
      const tab=$(`tab-${name}`);
      tab.classList.toggle('active',visible);
      tab.setAttribute('aria-selected',String(selected));
      tab.setAttribute('aria-pressed',String(visible));
      tab.tabIndex=selected?0:-1;
    }
    const editor=$('editor-panel'),editorVisible=state.editorOpen&&(!mobile||state.mobile==='editor');
    editor.hidden=!editorVisible;
    document.querySelector('[data-collapse=editor]').setAttribute('aria-expanded',String(!editor.classList.contains('collapsed')));
    editor.classList.toggle('mobile-active',editorVisible);
    editor.classList.toggle('mobile-inactive',!editorVisible);
    document.body.classList.toggle('editor-hidden',!editorVisible);
    const editorButton=document.querySelector('[data-panel="editor"]');
    editorButton.classList.toggle('active',editorVisible);
    editorButton.setAttribute('aria-pressed',String(editorVisible));
    $('workspace-toggle').classList.toggle('active',!sidebar.hidden&&!state.sidebarCollapsed);
    $('workspace-toggle').setAttribute('aria-pressed',String(!sidebar.hidden&&!state.sidebarCollapsed));
    $('sidebar-toggle').textContent=state.sidebarCollapsed?'›':'‹';
    $('sidebar-toggle').setAttribute('aria-expanded',String(!state.sidebarCollapsed));
    $('sidebar-toggle').setAttribute('aria-label',lang==='zh'?(state.sidebarCollapsed?'展开工作栏':'收起工作栏'):(state.sidebarCollapsed?'Expand workspace':'Collapse workspace'));
    if(state.activeTab==='structure'&&!state.sidebarCollapsed&&structureGraph?.autoFit!==false)fitGraph();
    for(const [id,host] of [['sidebar-resize',sidebar],['editor-resize',editor]])$(id).setAttribute('aria-valuenow',String(Math.round(host.getBoundingClientRect().width)));
    if(viewer.robot)updateCameraFrame();
    queueLinkMeasurementUpdate();
  }
  function setPanel(name,visible) {
    if(name==='motion'&&visible&&motion&&!motion.isOpen)return motion.toggle(true);
    if(leftPanels.includes(name)) {
      if(visible){state.activeTab=name;state.sidebarCollapsed=false;if(innerWidth<=600)state.mobile=name;}
      else if(state.activeTab===name)state.sidebarCollapsed=true;
    }else if(name==='editor') {
      state.editorOpen=visible;
      if(visible){$('editor-panel').classList.remove('collapsed');if(innerWidth<=600)state.mobile='editor';}
      else if(state.mobile==='editor')state.mobile=state.activeTab;
    }
    refreshWorkspace();
  }
  $$('[data-panel]').forEach(button=>button.onclick=()=>{
    const name=button.dataset.panel;
    const visible=name==='editor'?$('editor-panel').hidden||statics?.view==='statics':true;
    if(name==='editor')statics?.setRightView('code',false);
    if(name==='motion'){motion?.toggle(true);return;}
    setPanel(name,visible);
  });
  $('workspace-toggle').onclick=()=>{
    if($('workspace-sidebar').hidden){state.mobile=state.activeTab;state.sidebarCollapsed=false;}
    else state.sidebarCollapsed=!state.sidebarCollapsed;
    refreshWorkspace();
  };
  $('sidebar-toggle').onclick=()=>{state.sidebarCollapsed=!state.sidebarCollapsed;refreshWorkspace();};
  document.querySelector('[role="tablist"]').addEventListener('keydown',event=>{
    if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','End'].includes(event.key))return;
    event.preventDefault();
    const index=leftPanels.indexOf(state.activeTab),count=leftPanels.length,next=event.key==='Home'?0:event.key==='End'?count-1:(index+(['ArrowLeft','ArrowUp'].includes(event.key)?count-1:1))%count;
    setPanel(leftPanels[next],true);$(`tab-${leftPanels[next]}`).focus();
  });
  $$('[data-close]').forEach(button=>button.onclick=()=>setPanel(button.dataset.close,false));
  $$('[data-collapse]').forEach(button=>button.onclick=()=>{
    const name=button.dataset.collapse;
    if(leftPanels.includes(name)){state.sidebarCollapsed=true;refreshWorkspace();}
    else {const panel=$(`${name}-panel`);panel.classList.toggle('collapsed');button.setAttribute('aria-expanded',String(!panel.classList.contains('collapsed')));if(viewer.robot)updateCameraFrame();}
  });
  function installDockResize(id,variable,direction,min,max,other) {
    const handle=$(id),host=handle.parentElement;
    let drag=null;
    const apply=width=>{
      const otherWidth=other==='sidebar'?(state.sidebarCollapsed?44:$('workspace-sidebar').getBoundingClientRect().width):($('editor-panel').hidden?0:$('editor-panel').getBoundingClientRect().width);
      const limit=Math.max(min,Math.min(max,innerWidth-otherWidth-344));
      document.body.style.setProperty(variable,`${Math.round(Math.max(min,Math.min(limit,width)))}px`);
      refreshWorkspace();
    };
    handle.addEventListener('pointerdown',event=>{
      if(event.button!==0||innerWidth<=600)return;
      event.preventDefault();drag={x:event.clientX,width:host.getBoundingClientRect().width};
      handle.setPointerCapture(event.pointerId);document.body.classList.add('resizing-dock');
    });
    handle.addEventListener('pointermove',event=>{if(drag)apply(drag.width+(event.clientX-drag.x)*direction);});
    const end=()=>{drag=null;document.body.classList.remove('resizing-dock');};
    handle.addEventListener('pointerup',end);handle.addEventListener('pointercancel',end);handle.addEventListener('lostpointercapture',end);
    handle.addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight','Home'].includes(event.key))return;event.preventDefault();if(event.key==='Home'){document.body.style.removeProperty(variable);refreshWorkspace();}else apply(host.getBoundingClientRect().width+(event.key==='ArrowRight'?16:-16)*direction);});
    handle.addEventListener('dblclick',()=>{document.body.style.removeProperty(variable);refreshWorkspace();});
  }
  installDockResize('sidebar-resize','--side',1,260,440,'editor');
  installDockResize('editor-resize','--right-side',-1,280,600,'sidebar');
  function restoreLayout() {
    document.body.style.removeProperty('--side');document.body.style.removeProperty('--right-side');
    for(const panel of $$('.panel')){panel.removeAttribute('style');panel.classList.remove('collapsed');}
    state.activeTab='files';state.sidebarCollapsed=false;state.editorOpen=innerWidth>1000;state.mobile='files';
    refreshWorkspace();
  }
  $('restore-layout').onclick=()=>{restoreLayout();$('help-dialog').close();};
  let layoutBand=innerWidth<=600?'mobile':innerWidth<=1000?'compact':'desktop';
  refreshWorkspace();
  window.addEventListener('resize',()=>{
    const band=innerWidth<=600?'mobile':innerWidth<=1000?'compact':'desktop';
    if(band!==layoutBand){
      state.editorOpen=band==='desktop';state.mobile=state.activeTab;layoutBand=band;
      document.body.style.removeProperty('--side');document.body.style.removeProperty('--right-side');
    }
    if(band==='desktop'&&state.editorOpen&&!state.sidebarCollapsed) {
      const left=$('workspace-sidebar').getBoundingClientRect().width,right=$('editor-panel').getBoundingClientRect().width;
      if(left+right>innerWidth-344) {
        const width=Math.max(260,Math.min(left,innerWidth-624));
        document.body.style.setProperty('--side',`${width}px`);
        document.body.style.setProperty('--right-side',`${Math.max(280,innerWidth-344-width)}px`);
      }
    }
    refreshWorkspace();viewer.redraw();
  });
  statics=createStaticsWorkbench({THREE,viewer,grid,refreshScene:()=>{queueLinkMeasurementUpdate();queueJointAngleUpdate();fitView();},getModel:()=>({xml:state.xml,version:state.version}),getLanguage:()=>lang,openRight:()=>setPanel('editor',true),refreshAxes:refreshJointAngleAxis,project:projectMeasurementPoint});
  motion=createTrajectoryWorkbench({THREE,viewer,grid,projectPoint:projectMeasurementPoint,fitBounds:fitView,getJointInfo:()=>state.orderedJoints.map(n=>({name:n.name,velocity:Number(direct(n.element,'limit')?.getAttribute('velocity'))})),getLanguage:()=>lang,showPanel:name=>setPanel(name,true),refreshLayout:refreshWorkspace,
    beforePose:()=>{clearSurfaceMeasurement();statics.prepareMotion({preserveGround:true});motion?.restoreGround();},
    onPose:()=>{viewer.redraw();queueLinkMeasurementUpdate();queueJointAngleUpdate();statics.requestProjection();},
    getGround:()=>{viewer.world.updateWorldMatrix(true,false);viewer.plane.updateWorldMatrix(true,false);const inverse=viewer.world.getWorldQuaternion(new THREE.Quaternion()).invert(),normal=new THREE.Vector3(0,0,1).applyQuaternion(viewer.plane.getWorldQuaternion(new THREE.Quaternion())).applyQuaternion(inverse).normalize(),point=viewer.plane.getWorldPosition(new THREE.Vector3());viewer.world.worldToLocal(point);return {normal:normal.toArray(),offset:normal.dot(point)};}
  });
  $('help').onclick=()=>{$('help-dialog').showModal();$('help-content').scrollTop=0;};$('close-help').onclick=()=>$('help-dialog').close();
  function translate(){document.documentElement.lang=lang==='zh'?'zh-CN':'en';$$('[data-i18n]').forEach(el=>el.textContent=t(el.dataset.i18n));$('language').textContent=lang==='zh'?'EN':'中文';$('joint-search').placeholder=lang==='zh'?'搜索关节…':'Search joints…';renderURDFHelp($('help-content'),lang);updateEditor();filterJoints();refreshWorkspace();statics?.languageChanged();motion?.languageChanged();updateLinkMeasureUI();updateLinkMeasurement();if(surfaceSnap.pointer)renderSurfaceSnap();if(viewer.robot){$('model-meta').textContent=`${Object.keys(viewer.robot.links).length} ${t('linkTotal')} · ${Object.keys(viewer.robot.joints).length} ${t('jointTotal')}`;}}
  $('language').onclick=()=>{lang=lang==='zh'?'en':'zh';translate();};
  $('theme').onclick=()=>{const light=document.body.classList.toggle('light');$('theme').textContent=light?'☀':'☾';document.querySelector('meta[name="theme-color"]').content=light?'#E8F0F8':'#2E6B8A';grid.material.color.set(light?0x2e6b8a:0x6ba3c8);grid.material.opacity=light?.26:.32;viewer.redraw();};
  translate();
  function animate(now){motion?.tick(now);requestAnimationFrame(animate);}requestAnimationFrame(animate);
  installModelPicking();
  // Small read-only debugging surface for local integration checks.
  window.urdfWorkbench={get viewer(){return viewer;},get display(){return {...state.display};},get hoveredLink(){return hoveredLink;},get selectedObject(){return state.selected;},get linkMeasurement(){return {enabled:linkMeasurement.enabled,links:linkMeasurement.links.map(link=>link.name),unit:linkMeasurement.unit,distance:linkMeasurement.distance,delta:linkMeasurement.delta?.toArray() || null};},get surfaceMeasurement(){return {anchors:surfaceMeasurement.anchors.map(a=>({link:a.link.name,kind:a.kind,fitted:a.fitted,local:a.local.toArray(),world:a.object.localToWorld(a.local.clone()).toArray()})),unit:surfaceMeasurement.unit,distance:surfaceMeasurement.distance,delta:surfaceMeasurement.delta?.toArray() || null};},get statics(){return statics.snapshot;},get motion(){return motion?.snapshot;},get loading(){return $('status').classList.contains('loading');}};
})();
