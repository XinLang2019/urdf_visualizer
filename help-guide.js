/* Illustrated help, using local screenshots and the workbench's current language. */
(() => {
  'use strict';
  const guides = {
    zh: {
      intro:'加载、调姿、测量与分析，一页看懂。',
      steps:['加载机器人','调整姿态','测量与分析'],
      cards:[
        {title:'加载与浏览',image:'workbench.png',alt:'Go2 模型、左侧工作栏和右侧代码编辑器',
          lines:['选择内置示例，或拖入 URDF / MJCF / USD 完整模型文件夹。MJCF / 文本 USDA 在浏览器本地处理，无需安装。','左侧模型标签可切换、关闭，默认沿 Y 轴并排显示多个模型；“位置”调整 XYZ 位置与间隙。点击三维模型切换其代码。']},
        {title:'关节与辅助显示',image:'joint-control.png',alt:'拖动关节后，在轴旁输入目标关节角度',
          lines:['拖动连杆或关节滑块调姿；点击轴旁数值输入目标角度，自动遵守限位。','双击连杆定位关节与代码；顶部可开启质心、惯量、坐标轴和关节轴。']},
        {title:'两种三维测量',image:'measurement.png',alt:'黄色总距离及红绿蓝 XYZ 分量标注',
          lines:['Shift + 点击两个表面点；靠近推荐点自动吸附，Alt 暂停吸附。','Ctrl + 点击两个连杆或结构节点，测量原点距离与 XYZ 分量；松开恢复拖动。']},
        {title:'静力学与斜坡',image:'slope-statics.png',alt:'四足贴合斜坡，显示关节力矩、机身角度与接触力',
          lines:['调好站姿，开启“受力分析”；点击力矩标签查看力与力臂，再次点击恢复。','设置坡度、坡向和摩擦系数，点击“一键贴合斜坡”，查看机身角度与轮轴力矩。']}
      ],
      formatsTitle:'格式转换与 STEP 装配',formatsNote:'文件栏选择 URDF / MJCF / USD 转换下载 ZIP。选择 STEP 可导出保留连杆层级的零位静态装配，单位为 mm，网格保留为三角面几何。在线版无需 Python；USD 导出为文本 .usda。MJCF / 文本 USDA 编辑原格式代码；保存写入授权源文件，重新加载后生效。',
      meshTitle:'Mesh Simplifier',meshNote:'文件栏选择已加载的 STL / OBJ / DAE，设置目标面数并对比预览，再覆盖源文件或下载。也可单独加载网格文件夹。',
      saveTitle:'编辑与保存',save:'应用代码到模型',write:'保存回源文件',
      saveNote:'编辑器内 Ctrl + S 保存源文件。首次保存可能需要授权，保存后需重新加载以应用修改。',
      controlsTitle:'视图快捷操作',
      controls:[['左键拖动空白','旋转'],['右键拖动','平移'],['滚轮','缩放'],['F','适应模型'],['Esc','清除测量 / 关闭帮助']],
      note:'静力学结果为静态估算；无法稳定支撑或摩擦不足时，请先调整姿态或参数。',
      footer:'点击图片查看大图 · Esc 关闭帮助',preview:'查看大图：',close:'关闭帮助'
    },
    en: {
      intro:'From loading a robot to inspecting its loads.',
      steps:['Load a robot','Set a pose','Measure & analyze'],
      cards:[
        {title:'Load & explore',image:'workbench.png',alt:'Go2 model with the left workspace and right code editor',
          lines:['Choose an example or drop a complete URDF / MJCF / USD folder. MJCF / text USDA runs locally in your browser.','Switch or close models with the left tabs; models appear side by side along the model Y axis by default. Position sets XYZ placement and spacing; click a 3D model to select its code.']},
        {title:'Joints & visual helpers',image:'joint-control.png',alt:'Enter a target joint angle beside the joint axis',
          lines:['Drag a link or joint slider to pose the robot. Click the axis readout to enter a target within joint limits.','Double-click a link to locate its joint and code. Toggle CoM, inertia, frames, and joint axes in the toolbar.']},
        {title:'Two ways to measure',image:'measurement.png',alt:'Total distance in yellow with red, green, and blue XYZ components',
          lines:['Shift + click two surface points. Nearby suggested points snap automatically; Alt bypasses snapping.','Ctrl + click two links or graph nodes for origin distance and XYZ components. Release Ctrl to resume dragging.']},
        {title:'Statics & slopes',image:'slope-statics.png',alt:'Robot fitted to a slope with joint torques, body attitude, and contact forces',
          lines:['Set a standing pose and enable Statics. Click a torque label for forces and moment arms; click again to restore labels.','Set slope angle, heading, and friction, then Fit supports to slope. Inspect body attitude and wheel holding torques.']}
      ],
      formatsTitle:'Format conversion & STEP assembly',formatsNote:'Convert URDF / MJCF / USD in Files and download a ZIP. STEP exports a static zero-pose assembly retaining the link tree, in mm; mesh geometry stays faceted. No Python is required online. USD exports as text .usda. MJCF / text USDA edits use original source; Save writes the authorized file and Reload applies edits.',
      meshTitle:'Mesh Simplifier',meshNote:'Select a loaded STL / OBJ / DAE in Files, set the target face count and compare previews, then overwrite the source or download. Standalone mesh folders work too.',
      saveTitle:'Edit & save',save:'Apply code to model',write:'Save back to source',
      saveNote:'Ctrl + S saves source in the editor. Saving may need permission. Reload to apply saved code to the model.',
      controlsTitle:'View controls',
      controls:[['Left-drag background','Orbit'],['Right-drag','Pan'],['Scroll','Zoom'],['F','Fit model'],['Esc','Clear measurement / close help']],
      note:'Static loads are estimates. Adjust the pose or parameters if support is unstable or friction is insufficient.',
      footer:'Click an image to enlarge · Esc closes help',preview:'View full image: ',close:'Close help'
    }
  };
  window.renderURDFHelp = (container, language) => {
    const guide=guides[language] || guides.en;
    const element=(tag,className,text)=>{
      const node=document.createElement(tag);
      if(className)node.className=className;
      if(text)node.textContent=text;
      return node;
    };
    const intro=element('div','help-intro');
    intro.append(element('p','',guide.intro));
    const steps=element('ol','help-steps');
    for(const text of guide.steps)steps.append(element('li','',text));
    intro.append(steps);
    const grid=element('div','help-grid');
    guide.cards.forEach((card,index)=>{
      const article=element('article','help-card'),link=element('a','help-picture');
      link.href=`docs/images/${card.image}`;link.target='_blank';link.rel='noopener';
      link.setAttribute('aria-label',guide.preview+card.title);
      const image=element('img');image.src=link.href;image.alt=card.alt;
      image.loading='lazy';image.decoding='async';image.width=1897;image.height=961;
      link.append(image);
      const body=element('div','help-card-body'),heading=element('h3');
      heading.append(element('span','help-number',String(index+1).padStart(2,'0')),document.createTextNode(card.title));
      body.append(heading);
      for(const text of card.lines)body.append(element('p','',text));
      article.append(link,body);grid.append(article);
    });
    const formats=element('section','help-save');formats.append(element('h3','',guide.formatsTitle),element('p','',guide.formatsNote));const setup=element('a','',language==='zh'?'格式配置与支持范围':'Format setup & support');setup.href='docs/formats.md';setup.target='_blank';setup.rel='noreferrer';formats.append(setup);
    const mesh=element('section','help-save');mesh.append(element('h3','',guide.meshTitle),element('p','',guide.meshNote));const meshLink=element('a','',language==='zh'?'网格简化使用说明':'Mesh simplification guide');meshLink.href='docs/mesh-simplifier.md';meshLink.target='_blank';meshLink.rel='noreferrer';mesh.append(meshLink);
    const save=element('section','help-save');save.append(element('h3','',guide.saveTitle));
    const actions=element('div','help-save-actions');
    for(const [key,label]of [['Ctrl + Enter',guide.save],['Ctrl + S',guide.write]]){
      const action=element('span');action.append(element('kbd','',key),document.createTextNode(label));actions.append(action);
    }
    save.append(actions,element('p','',guide.saveNote));
    const controls=element('section','help-controls');controls.append(element('h3','',guide.controlsTitle));
    const list=element('dl');
    for(const [key,value]of guide.controls){const item=element('div');item.append(element('dt','',key),element('dd','',value));list.append(item);}
    controls.append(list);
    container.replaceChildren(intro,grid,formats,mesh,save,controls,element('p','help-note',guide.note));
    document.getElementById('help-footer-hint').textContent=guide.footer;
    document.getElementById('close-help').setAttribute('aria-label',guide.close);
  };
})();
