"""Fixed-joint four-support placement: body attitude follows actual contact geometry."""
import json
import math
from model_interaction import rendered_frame
from slope_statics import change


def fixture(kind='wheel',front_radius=.2):
    xml='''<robot name="support_fit_fixture"><link name="base"><inertial><origin xyz="0 0 .8"/><mass value="10"/><inertia ixx="1" iyy="1" izz="1"/></inertial><visual><origin xyz="0 0 .8"/><geometry><box size=".8 .5 .2"/></geometry></visual></link>'''
    for i,(x,y) in enumerate([(.5,.4),(.5,-.4),(-.5,.4),(-.5,-.4)]):
        radius=front_radius if i<2 else .2
        geom=f'<cylinder radius="{radius}" length=".1"/>' if kind=='wheel' else f'<sphere radius="{radius}"/>'
        rotation='rpy="1.5707963267948966 0 0"' if kind=='wheel' else ''
        xml+=f'''<link name="suspension_{i}"><inertial><mass value="0"/><inertia ixx="0" iyy="0" izz="0"/></inertial></link>
        <joint name="height_{i}" type="prismatic"><origin xyz="{x} {y} .2"/><parent link="base"/><child link="suspension_{i}"/><axis xyz="0 0 1"/><limit lower="-.3" upper=".3" effort="500" velocity="2"/></joint>
        <link name="{kind}_{i}"><inertial><mass value="1"/><inertia ixx=".01" iyy=".01" izz=".01"/></inertial><visual><origin {rotation}/><geometry>{geom}</geometry></visual><collision><origin {rotation}/><geometry>{geom}</geometry></collision></link>
        <joint name="axle_{i}" type="continuous"><parent link="suspension_{i}"/><child link="{kind}_{i}"/><axis xyz="0 1 0"/><limit effort="50" velocity="10"/></joint>'''
    return xml+'</robot>'


ACTUAL_CONTACTS='''(()=>{const T=parcelRequire('../node_modules/three/build/three.module.js'),s=urdfWorkbench.statics,v=urdfWorkbench.viewer,G=new URDFStaticContacts(T),n=new T.Vector3(...s.terrain.normal),q=new T.Quaternion().setFromUnitVectors(n,new T.Vector3(0,1,0)),frame={toPlane:new T.Matrix4().makeRotationFromQuaternion(q)};v.world.updateWorldMatrix(true,true);return s.terrain.fit.links.length===4&&s.terrain.fit.links.every(name=>{const object=v.robot.links[name],meshes=[];for(const group of object.children.filter(c=>c.isURDFCollider))group.traverse(m=>{if(m.isMesh)meshes.push(m)});const b=G.bottom({link:name,object,meshes,fallback:false},frame);return Math.abs(b.min-s.ground)<1e-6})})()'''


def load(client,kind='wheel',front_radius=.2):
    client.js('window.previousFitRobot=urdfWorkbench.viewer.robot;document.getElementById("source").value='+json.dumps(fixture(kind,front_radius))+';document.getElementById("source").dispatchEvent(new Event("input"));document.getElementById("reload").click()')
    client.until('urdfWorkbench.viewer.robot?.robotName==="support_fit_fixture"&&urdfWorkbench.viewer.robot!==previousFitRobot&&!urdfWorkbench.loading')
    client.js('if(!urdfWorkbench.statics.enabled)document.getElementById("statics-toggle").click()')
    change(client,'statics-friction',.9)


def heights(client,values):
    client.js('for(const [i,value]of '+json.dumps(list(enumerate(values)))+')urdfWorkbench.viewer.setJointValue("height_"+i,value)')
    rendered_frame(client)


def place(client):
    client.js('document.getElementById("statics-place-slope").click()');rendered_frame(client)


def check_support_fitting(client,output,check):
    load(client)
    heights(client,[.15,.15,0,0])
    change(client,'statics-slope-angle',15);change(client,'statics-slope-heading',0)
    client.js('window.fitAngles=Object.fromEntries(Object.entries(urdfWorkbench.viewer.robot.joints).map(([n,j])=>[n,j.angle]));window.fitXML=document.getElementById("source").value;urdfWorkbench.viewer.robot.updateMatrix();window.fitOriginalRoot=urdfWorkbench.viewer.robot.matrix.toArray()')
    place(client)
    correction=-math.degrees(math.atan(.15))
    check(client,'(()=>{const s=urdfWorkbench.statics,a=s.terrain.bodyAttitude;return s.valid&&s.terrain.fit.success&&s.contacts.length===4&&s.terrain.fit.maxGap<1e-6&&Math.abs(a.horizontal.pitch-('+str(15+correction)+'))<1e-5&&Math.abs(a.slope.pitch-('+str(correction)+'))<1e-5&&Math.abs(a.horizontal.roll)<1e-5})()','front/rear height difference determines body pitch instead of forcing it parallel to the 15-degree slope')
    check(client,ACTUAL_CONTACTS,'all four real wheel surfaces touch the slope, including the previously raised front wheels')
    check(client,'Object.entries(urdfWorkbench.viewer.robot.joints).every(([n,j])=>j.angle===fitAngles[n])&&document.getElementById("source").value===fitXML&&document.getElementById("statics-placement-detail").textContent.includes("机身俯仰")','fitting preserves all suspension/axle positions and XML while displaying derived body angles')
    client.screenshot(str(output/'fitted-asymmetric-wheels.png'))
    client.js('urdfWorkbench.viewer.robot.updateMatrix();window.fitPlacedRoot=urdfWorkbench.viewer.robot.matrix.toArray();document.getElementById("statics-place-slope").click()');rendered_frame(client)
    check(client,'urdfWorkbench.viewer.robot.matrix.toArray().every((x,i)=>Math.abs(x-fitPlacedRoot[i])<1e-7)','contact fitting remains idempotent for an asymmetric stance')
    # Add roll as well as pitch. Four centers still lie on y = .1 x + .08 z.
    heights(client,[.15+.05-.032,.15+.05+.032,.15-.05-.032,.15-.05+.032])
    check(client,'!urdfWorkbench.statics.valid&&urdfWorkbench.statics.reason==="placementMismatch"','changing suspension positions invalidates four-support loads until contacts are fitted again')
    place(client)
    pitch=-math.degrees(math.asin(.1/math.sqrt(1+.1**2+.08**2)))
    roll=math.degrees(math.atan(.08))
    check(client,'urdfWorkbench.statics.valid&&Math.abs(urdfWorkbench.statics.terrain.bodyAttitude.slope.pitch-('+str(pitch)+'))<1e-5&&Math.abs(urdfWorkbench.statics.terrain.bodyAttitude.slope.roll-('+str(roll)+'))<1e-5','combined front/rear and left/right height differences produce the expected relative pitch and roll')
    check(client,ACTUAL_CONTACTS,'combined pitch/roll fitting still makes all four wheel surfaces contact')
    # Twisting one corner cannot be repaired by a rigid body transform.
    client.js('urdfWorkbench.viewer.robot.updateMatrix();window.beforeFailedFit=urdfWorkbench.viewer.robot.matrix.toArray()')
    heights(client,[.168,.232,.068,.212]);place(client)
    check(client,'!urdfWorkbench.statics.valid&&urdfWorkbench.statics.reason==="placementMismatch"&&!urdfWorkbench.statics.terrain.fit.success&&urdfWorkbench.statics.terrain.fit.maxGap>.005&&urdfWorkbench.viewer.robot.matrix.toArray().every((x,i)=>Math.abs(x-beforeFailedFit[i])<1e-7)&&document.querySelectorAll(".statics-axis-label:not([hidden])").length===0','a noncoplanar four-corner stance is rejected without moving the prior placement or fabricating four-support torques')
    client.screenshot(str(output/'fitted-noncoplanar-warning.png'))
    heights(client,[.15,.15,0,0]);place(client)
    check(client,'urdfWorkbench.statics.valid&&urdfWorkbench.statics.terrain.fit.success','correcting the stance recovers from a failed fit')
    change(client,'statics-slope-angle',0);place(client)
    check(client,'urdfWorkbench.statics.valid&&Math.abs(urdfWorkbench.statics.terrain.bodyAttitude.horizontal.pitch-('+str(correction)+'))<1e-5','zero slope also fits all four contacts and reports a nonzero body angle when leg heights differ')
    client.js('document.getElementById("statics-flat").click()');rendered_frame(client)
    check(client,'!urdfWorkbench.statics.terrain.active&&urdfWorkbench.viewer.robot.matrix.toArray().every((x,i)=>Math.abs(x-fitOriginalRoot[i])<1e-7)&&urdfWorkbench.viewer.robot.joints.height_0.angle===.15','restore-flat reverses rigid fitting while retaining configured joint values')
    # Spherical feet with unequal radii require recomputing tangent points;
    # fitting the original lowest points alone gives the wrong attitude.
    load(client,'foot',.15);heights(client,[.15,.15,0,0])
    change(client,'statics-slope-angle',15);place(client)
    sphere_angle=math.degrees(math.asin(-.05/math.sqrt(1+.15**2))-math.atan(.15))
    check(client,'urdfWorkbench.statics.valid&&urdfWorkbench.statics.terrain.fit.links.length===4&&urdfWorkbench.statics.terrain.fit.iterations>0&&Math.abs(urdfWorkbench.statics.terrain.bodyAttitude.slope.pitch-('+str(sphere_angle)+'))<1e-5&&document.querySelectorAll(".statics-wheel-row").length===0','unequal-radius spherical feet use their true tangent geometry and produce the analytic body angle')
    check(client,ACTUAL_CONTACTS,'all four unequal-radius spherical surfaces touch the slope after attitude solving')
    client.screenshot(str(output/'fitted-unequal-radius-feet.png'))
    # The rendered pitch is checked independently from the actual URDF axes.
    check(client,'(()=>{const T=parcelRequire("../node_modules/three/build/three.module.js"),v=urdfWorkbench.viewer;v.world.updateWorldMatrix(true,true);const f=new T.Vector3(1,0,0).transformDirection(v.robot.matrixWorld),right=new T.Vector3(0,-1,0).transformDirection(v.robot.matrixWorld),up=new T.Vector3(0,0,1).transformDirection(v.robot.matrixWorld),a=urdfWorkbench.statics.terrain.bodyAttitude.horizontal;return Math.abs(Math.asin(f.y)*180/Math.PI-a.pitch)<1e-6&&Math.abs(Math.atan2(-right.y,up.y)*180/Math.PI-a.roll)<1e-6})()','reported body pitch and roll match the physical URDF base axes in the 3D scene')
    client.js('document.getElementById("statics-flat").click();document.getElementById("statics-toggle").click();document.getElementById("right-code-tab").click();document.getElementById("statics-friction").value=.7')


def check_go2_support_fit(client,output,check):
    client.js('document.getElementById("example-select").value="urdf/go2/urdf/go2.urdf";document.getElementById("load-example").click()')
    client.until('urdfWorkbench.viewer.robot?.robotName==="go2_description"&&!urdfWorkbench.loading',55)
    client.js('if(!urdfWorkbench.statics.enabled)document.getElementById("statics-toggle").click();for(const side of ["FL","FR","RL","RR"]){urdfWorkbench.viewer.setJointValue(side+"_thigh_joint",side[0]==="F"?.4:.7);urdfWorkbench.viewer.setJointValue(side+"_calf_joint",side[0]==="F"?-1.8:-1.4)}')
    rendered_frame(client)
    client.js('''(()=>{const T=parcelRequire('../node_modules/three/build/three.module.js'),v=urdfWorkbench.viewer;v.world.updateWorldMatrix(true,true);const front=new T.Vector3(),rear=new T.Vector3();for(const side of ['FL','FR','RL','RR']){const link=v.robot.links[side+'_foot'],mesh=link.children.find(o=>o.isURDFCollider).children.find(o=>o.isMesh);(side[0]==='F'?front:rear).add(mesh.getWorldPosition(new T.Vector3()))}const d=front.sub(rear);window.go2ExpectedCorrection=-Math.atan2(d.y,d.x)*180/Math.PI;window.go2FixedStance=Object.fromEntries(Object.entries(v.robot.joints).map(([n,j])=>[n,j.angle]));window.go2FitSource=document.getElementById('source').value})()''')
    change(client,'statics-slope-angle',15);change(client,'statics-slope-heading',0);change(client,'statics-friction',.9)
    place(client)
    check(client,'(()=>{const s=urdfWorkbench.statics;return s.valid&&s.terrain.fit.success&&s.terrain.fit.links.length===4&&s.contacts.length===4&&s.terrain.fit.maxGap<1e-6&&Math.abs(s.terrain.bodyAttitude.slope.pitch-go2ExpectedCorrection)<1e-5&&Math.abs(go2ExpectedCorrection)>5&&Math.abs(s.terrain.bodyAttitude.horizontal.pitch-(15+go2ExpectedCorrection))<1e-5})()','Go2 with different front/rear leg angles fits all four feet and solves its actual body pitch')
    check(client,ACTUAL_CONTACTS,'all four actual Go2 foot collision spheres are on the slope after asymmetric stance fitting')
    check(client,'Object.entries(urdfWorkbench.viewer.robot.joints).every(([n,j])=>j.angle===go2FixedStance[n])&&document.getElementById("source").value===go2FitSource','real robot fitting preserves the complete configured joint stance and its URDF text')
    client.screenshot(str(output/'fitted-asymmetric-go2.png'))
    client.js('document.getElementById("statics-flat").click();document.getElementById("statics-toggle").click();document.getElementById("right-code-tab").click()')
