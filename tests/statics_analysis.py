"""Hand-calculated floating-base statics and rendered live-analysis integration."""
import json
from model_interaction import rendered_frame

FIXTURE='''<robot name="statics_fixture">
<link name="base"><inertial><origin xyz="0 0 1"/><mass value="10"/><inertia ixx="1" iyy="1" izz="1"/></inertial><visual><origin xyz="0 0 1"/><geometry><box size=".5 .3 .2"/></geometry></visual><collision><origin xyz="0 0 1"/><geometry><box size=".5 .3 .2"/></geometry></collision></link>
<link name="left"><inertial><origin xyz="-.3 0 -.95"/><mass value="1"/><inertia ixx=".1" iyy=".1" izz=".1"/></inertial><visual><origin xyz="-.3 0 -.95"/><geometry><box size=".2 .2 .1"/></geometry></visual><collision><origin xyz="-.3 0 -.95"/><geometry><box size=".2 .2 .1"/></geometry></collision></link>
<link name="right"><inertial><origin xyz=".3 0 -.95"/><mass value="1"/><inertia ixx=".1" iyy=".1" izz=".1"/></inertial><visual><origin xyz=".3 0 -.95"/><geometry><box size=".2 .2 .1"/></geometry></visual><collision><origin xyz=".3 0 -.95"/><geometry><box size=".2 .2 .1"/></geometry></collision></link>
<joint name="left_hip" type="revolute"><origin xyz="-.2 0 1"/><parent link="base"/><child link="left"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="10" velocity="2"/></joint>
<joint name="right_hip" type="revolute"><origin xyz=".2 0 1"/><parent link="base"/><child link="right"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="30" velocity="2"/></joint>
</robot>'''


def check_statics_analysis(client,output,check):
    # Independent known loads: asymmetric two-support beam, unilateral boundary, no support.
    checks=client.js('''(()=>{const E=URDFStatics,near=(a,b)=>Math.abs(a-b)<1e-6,out={};
    const beam=E.contactLoads([[-1,0,0],[1,0,0]],[.25,1,0],100);out.beam=beam.valid&&near(beam.forces[0],37.5)&&near(beam.forces[1],62.5);
    const square=[[-1,0,-1],[-1,0,1],[1,0,-1],[1,0,1]],even=E.contactLoads(square,[0,1,0],100);out.even=even.valid&&even.forces.every(f=>near(f,25))&&near(even.margin,1);
    const edge=E.contactLoads(square,[1,1,1],100);out.edge=edge.valid&&edge.forces.every(f=>f>=0)&&near(edge.forces[3],100)&&edge.forces.slice(0,3).every(f=>near(f,0));
    const unilateral=E.contactLoads(square,[.8,1,.8],100);out.unilateral=unilateral.valid&&unilateral.forces.every((f,i)=>near(f,[0,10,10,80][i]));
    out.outside=!E.contactLoads(square,[2,1,0],100).valid;
    out.singular=E.contactLoads([[2,0,3]],[2,1,3],50).valid&&!E.contactLoads([[2,0,3]],[2.1,1,3],50).valid;
    const masses=[{link:'root',mass:10,position:[0,1,0]},{link:'leg',mass:1,position:[-.5,0,0]}],joints=[{name:'hip',type:'revolute',position:[-.2,1,0],axis:[0,0,-1],descendants:['leg'],limit:10}],contacts=[{link:'leg',position:[-.5,0,0]},{link:'root',position:[.5,0,0]}];
    const r=E.analyze({masses,joints,contacts}),j=r.joints[0];out.wrench=r.valid&&near(j.gravityEffort,2.943)&&near(j.contactEffort,-17.658)&&near(j.effort,-14.715)&&j.overLimit&&r.forceResidual<1e-6&&r.momentResidual<1e-6;
    const slider=E.analyze({masses,joints:[{...joints[0],type:'prismatic',axis:[0,1,0]}],contacts});out.slider=slider.valid&&near(slider.joints[0].effort,-49.05)&&slider.joints[0].unit==='N';
    const shifted=E.analyze({masses:masses.map(m=>({...m,position:m.position.map((x,i)=>x+[12,7,-8][i])})),joints:joints.map(j=>({...j,position:j.position.map((x,i)=>x+[12,7,-8][i])})),contacts:contacts.map(c=>({...c,position:c.position.map((x,i)=>x+[12,7,-8][i])}))});out.translation=shifted.valid&&near(shifted.joints[0].effort,j.effort);
    const flipped=E.analyze({masses,joints:[{...joints[0],axis:[0,0,1]}],contacts});out.axis=near(flipped.joints[0].effort,-j.effort);
    const mimic=E.analyze({masses,joints:[...joints,{...joints[0],name:'slave',mimic:{joint:'hip',multiplier:-2}}],contacts});out.mimic=near(mimic.joints[0].effort,-j.effort);
    return out;})()''')
    for name,passed in checks.items():assert passed,('analytical statics',name)
    print('PASS analytic forces, unilateral contacts, singular supports, wrench signs, prismatic units, translation and mimic virtual work',flush=True)
    client.js('document.getElementById("source").value='+json.dumps(FIXTURE)+';document.getElementById("source").dispatchEvent(new Event("input"));document.getElementById("reload").click()')
    client.until('urdfWorkbench.viewer.robot?.robotName==="statics_fixture"&&!urdfWorkbench.loading')
    client.js('if(!urdfWorkbench.statics.enabled)document.getElementById("statics-toggle").click()')
    rendered_frame(client)
    check(client,'(()=>{const s=urdfWorkbench.statics;return s.valid&&s.reason==="balanced"&&Math.abs(s.mass-12)<1e-8&&s.contacts.length===8&&s.missingMasses.length===0&&Math.abs(s.joints[0].effort+14.715)<1e-5&&Math.abs(s.joints[1].effort-14.715)<1e-5&&s.forceResidual<1e-6&&s.momentResidual<1e-6})()','current URDF pose reproduces hand-calculated 14.715 N·m hip torques including foot self-weight')
    check(client,'document.querySelectorAll(".statics-axis-label:not([hidden])").length===2&&!document.getElementById("statics-overlay").hasAttribute("hidden")&&document.querySelectorAll("#statics-overlay .statics-contact").length===8&&document.querySelector("#statics-overlay polygon")!==null','visible joint loads and support markers render at projected 3D positions')
    client.js('document.querySelector(`.statics-joint-row[data-joint="left_hip"]`).click()')
    check(client,'document.getElementById("statics-joint-details").open&&document.getElementById("statics-joint-detail").textContent.includes("left_hip")&&document.getElementById("statics-joint-detail").textContent.includes("0.3000")&&document.querySelector(`.statics-joint-row[data-joint="left_hip"]`).classList.contains("over-limit")','axis labels open force-arm details and flag URDF effort exceedance')
    client.screenshot(str(output/'statics-detail.png'))
    client.js('document.getElementById("statics-gravity").value="4.905";document.getElementById("statics-gravity").dispatchEvent(new Event("input"))')
    rendered_frame(client)
    check(client,'urdfWorkbench.statics.valid&&Math.abs(urdfWorkbench.statics.joints[0].effort+7.3575)<1e-5','editing gravity scales equilibrium torques immediately')
    client.js('document.getElementById("statics-gravity").value="";document.getElementById("statics-gravity").dispatchEvent(new Event("input"))');rendered_frame(client)
    check(client,'!urdfWorkbench.statics.valid&&document.querySelectorAll(".statics-axis-label:not([hidden])").length===0','invalid inputs invalidate stale results and hide load labels')
    client.js('document.getElementById("statics-gravity").value="9.81";document.getElementById("statics-gravity").dispatchEvent(new Event("input"));urdfWorkbench.viewer.setJointValue("left_hip",.2)');rendered_frame(client)
    check(client,'!urdfWorkbench.statics.valid&&["penetration","outsideSupport","airborne"].includes(urdfWorkbench.statics.reason)&&document.querySelectorAll(".statics-axis-label:not([hidden])").length===0','joint motion re-evaluates ground geometry and rejects unsupported poses without fabricated torques')
    client.js('urdfWorkbench.viewer.setJointValue("left_hip",0)');rendered_frame(client)
    check(client,'urdfWorkbench.statics.valid','restoring the stance restores equilibrium automatically')
    client.js('document.getElementById("statics-contact-mode").value="manual";document.getElementById("statics-contact-mode").dispatchEvent(new Event("change"))');rendered_frame(client)
    client.js('document.querySelector(".statics-contact-row button").click()');rendered_frame(client)
    check(client,'!urdfWorkbench.statics.valid&&urdfWorkbench.statics.reason==="outsideSupport"','manual support selection detects a CoM outside the remaining foot')
    client.js('document.getElementById("statics-contact-mode").value="auto";document.getElementById("statics-contact-mode").dispatchEvent(new Event("change"));document.getElementById("right-code-tab").click()');rendered_frame(client)
    check(client,'urdfWorkbench.statics.valid&&!document.getElementById("code-workspace").hidden&&urdfWorkbench.statics.enabled','editor remains accessible while live load labels stay enabled')
    client.js('document.getElementById("statics-toggle").click()');rendered_frame(client)
    check(client,'!urdfWorkbench.statics.enabled&&document.getElementById("statics-labels").hidden&&(()=>{const a=[];urdfWorkbench.viewer.robot.traverse(o=>{if(o.userData.axisKind==="jointAxes")a.push(o)});return a.every(o=>!o.visible)})()','turning off analysis restores the joint-axis display setting')
    client.js('document.querySelector("[data-display=jointAxes]").click();document.getElementById("statics-toggle").click();document.getElementById("statics-toggle").click()');rendered_frame(client)
    check(client,'urdfWorkbench.display.jointAxes&&(()=>{const a=[];urdfWorkbench.viewer.robot.traverse(o=>{if(o.userData.axisKind==="jointAxes")a.push(o)});return a.every(o=>o.visible)})()','disabling analysis preserves explicitly enabled joint axes')
    client.js('document.querySelector("[data-display=jointAxes]").click();document.getElementById("statics-toggle").click()');rendered_frame(client)
    no_mass='<robot name="statics_no_mass"><link name="base"><visual><geometry><box size="1 1 1"/></geometry></visual></link></robot>'
    client.js('document.getElementById("source").value='+json.dumps(no_mass)+';document.getElementById("source").dispatchEvent(new Event("input"));document.getElementById("reload").click()')
    client.until('urdfWorkbench.viewer.robot?.robotName==="statics_no_mass"&&!urdfWorkbench.loading');rendered_frame(client)
    check(client,'urdfWorkbench.statics.enabled&&!urdfWorkbench.statics.valid&&urdfWorkbench.statics.reason==="noMass"&&document.querySelectorAll(".statics-axis-label").length===0','reloading clears old joint labels and rejects a model without mass data')
    client.js('document.getElementById("statics-toggle").click();document.getElementById("right-code-tab").click()')
