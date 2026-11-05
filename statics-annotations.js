/* Joint label anchors and measurement-style lever diagrams for a selected joint. */
(function(){
  'use strict';
  window.URDFStaticsAnnotations=class {
    constructor({THREE,viewer,project,svg}) {
      Object.assign(this,{T:THREE,viewer,project,svg});
    }
    inScene(object) {for(let p=object;p;p=p.parent)if(!p.visible)return false;return true;}
    begin() {
      this.viewer.world.updateWorldMatrix(true,true);this.viewer.camera.updateMatrixWorld(true);
    }
    labelPoint(joint) {
      const glyph=joint.children.find(o=>o.userData.axisKind==='jointAxes');
      if(!glyph)return joint.getWorldPosition(new this.T.Vector3());
      const anchor=/hip/i.test(joint.name)?glyph.userData.tipAnchor:glyph.userData.labelAnchor;
      return glyph.localToWorld(new this.T.Vector3(...anchor));
    }
    draw(overlay,breakdown,glyph,bounds) {
      const T=this.T,V=a=>new T.Vector3(...a),project=this.project,svg=this.svg,format=(v,d=1)=>Number.isFinite(v)?(Math.abs(v)<.5*10**-d?0:v).toFixed(d):'—';
      const root=svg('g',{'class':'statics-arm-diagram','data-joint':breakdown.name}),used=(bounds.occupied||[]).map(r=>({x:(r.left+r.right)/2,y:(r.top+r.bottom)/2,width:r.width+8}));overlay.append(root);
      const colors={gravity:'#ffcc67',support:'#71e2e8',normal:'#71e2e8',friction:'#ff8db3'};
      const segment=(a,b,attrs={},parent=root)=>{
        const va=a.clone().applyMatrix4(this.viewer.camera.matrixWorldInverse),vb=b.clone().applyMatrix4(this.viewer.camera.matrixWorldInverse),near=this.viewer.camera.near;
        if(-va.z<near&&-vb.z<near)return null;
        let start=a.clone(),end=b.clone();if(-va.z<near)start.lerp(end,(-near-va.z)/(vb.z-va.z));else if(-vb.z<near)end.copy(a.clone().lerp(b,(-near-va.z)/(vb.z-va.z)));
        const p=project(start),q=project(end);if(!p.visible||!q.visible)return null;
        const line=svg('line',{x1:p.x,y1:p.y,x2:q.x,y2:q.y,...attrs});parent.append(line);return line;
      };
      const point=(p,attrs,parent=root)=>{const s=project(p);if(!s.visible)return;parent.append(svg('circle',{cx:s.x,cy:s.y,...attrs}));};
      const caption=(position,text,color,offset=[0,-12],parent=root)=>{
        const p=project(position);if(!p.visible||p.x<bounds.left-150||p.x>bounds.right+150)return;
        const node=svg('text',{'class':'statics-dimension-label',fill:color,'text-anchor':'middle'});node.textContent=text;parent.append(node);
        const width=node.getComputedTextLength()+10,x=Math.max(bounds.left+width/2,Math.min(bounds.right-width/2,p.x+offset[0]));let y=p.y+offset[1];
        for(const shift of [0,-22,22,-44,44,-66,66,-88,88]){const t=Math.max(bounds.top+16,Math.min(bounds.bottom-6,y+shift));if(!used.some(r=>Math.abs(x-r.x)<(width+r.width)/2&&Math.abs(t-r.y)<20)){y=t;break;}}
        node.setAttribute('x',x);node.setAttribute('y',Math.max(bounds.top+16,Math.min(bounds.bottom-6,y)));used.push({x,y,width});
      };
      const defs=svg('defs',{});root.append(defs);
      for(const [kind,color]of Object.entries(colors)){const marker=svg('marker',{id:`statics-force-${kind}`,viewBox:'0 0 10 10',refX:8,refY:5,markerWidth:6,markerHeight:6,orient:'auto'});marker.append(svg('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:color}));defs.append(marker);}
      const q=V(breakdown.position),axis=V(breakdown.axis);
      const length=Math.max(.025,Math.min(.25,glyph?q.distanceTo(glyph.localToWorld(new T.Vector3(...glyph.userData.angleAnchor)))*.9:.1));
      point(q,{r:5,fill:'#e8f0f8',stroke:'#2e6b8a','stroke-width':2});
      for(const component of breakdown.components) {
        const group=svg('g',{'class':`statics-force-component ${component.kind}`,'data-force':component.id,'data-effort':component.effort});root.append(group);
        const color=colors[component.kind],p=V(component.position),tip=p.clone().addScaledVector(V(component.force).normalize(),length);
        segment(p,tip,{'class':'statics-force-arrow',stroke:color,'stroke-width':3,'marker-end':`url(#statics-force-${component.kind})`},group);
        point(p,{r:5,fill:color,stroke:'#243746','stroke-width':1.2},group);
        caption(tip,`${component.id} ${format(component.magnitude)} N`,color,[16,component.kind==='gravity'?20:-12],group);
        if(!component.arm)continue;
        const a=V(component.arm.axisPoint),b=V(component.arm.linePoint),delta=V(component.arm.delta),step=new T.Vector3(b.x,a.y,a.z),stepY=new T.Vector3(b.x,b.y,a.z);
        segment(q,a,{'class':'statics-axis-extension',stroke:'#b8d4e8','stroke-width':1.5,'stroke-dasharray':'4 4'},group);
        segment(p,b,{'class':'statics-force-extension',stroke:color,'stroke-width':1.3,'stroke-dasharray':'4 4'},group);
        const xyz=[[a,step,'#ff5c5c','X',delta.x],[step,stepY,'#67e28d','Y',delta.y],[stepY,b,'#7895ff','Z',delta.z]],nonzero=xyz.filter(s=>Math.abs(s[4])>1e-6);
        for(const [start,end,c,name,value]of nonzero){segment(start,end,{stroke:c,'stroke-width':2.5,'class':'statics-arm-component'},group);if(nonzero.length>1)caption(start.clone().lerp(end,.5),`Δ${name} ${format(value*1000)} mm`,c,[0,name==='X'?-16:20],group);}
        segment(a,b,{'class':'statics-moment-arm',stroke:color,'stroke-width':3,'stroke-dasharray':'7 5','data-length':component.arm.distance,'data-effective-arm':component.effectiveArm},group);
        point(a,{r:3,fill:color},group);point(b,{r:3,fill:color},group);
        caption(a.clone().lerp(b,.5),`d_${component.id} ${format(component.arm.distance*1000)} mm`,color,[0,component.kind==='gravity'?-17:19],group);
        const offset=Math.min(length*.15,component.arm.distance*.2);
        if(offset>1e-6){const radial=b.clone().sub(a).normalize(),corner=a.clone().addScaledVector(axis,offset);segment(corner,corner.clone().addScaledVector(radial,offset),{stroke:color,'stroke-width':1.4},group);segment(corner.clone().addScaledVector(radial,offset),a.clone().addScaledVector(radial,offset),{stroke:color,'stroke-width':1.4},group);}
      }
    }
  };
})();
