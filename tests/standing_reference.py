"""Initial stance, contact-ground calibration, and ground-relative motion data."""
from motion_body import GROUND, change
from model_interaction import rendered_frame


def check_standing_reference(client, output, check):
    client.js('window.standOriginal={p:urdfWorkbench.viewer.robot.position.toArray(),q:urdfWorkbench.viewer.robot.quaternion.toArray(),j:Object.fromEntries(Object.entries(urdfWorkbench.viewer.robot.joints).map(([n,j])=>[n,j.angle]))};document.getElementById("animate").click();document.getElementById("motion-record").click()')
    check(client, 'urdfWorkbench.motion.project.keyframes.length===0&&document.getElementById("motion-status").classList.contains("error")', 'new motion requires a standing ground reference before recording')
    client.js('document.getElementById("motion-default-stand").click();window.standGround='+GROUND+';window.standReference=urdfWorkbench.motion.project.reference')
    check(client, '''(()=>{const p=urdfWorkbench.motion.project,r=p.reference,j=urdfWorkbench.viewer.robot.joints;return r?.supports.length===4&&r.contacts.every(c=>Math.abs(c.position[2])<1e-7)&&r.maxGap<1e-7&&r.initialStandingPose.base.position[2]>.3&&r.initialStandingPose.base.position[2]<.4&&Math.abs(Number(document.getElementById('motion-root-2').value)-r.initialStandingPose.base.position[2])<1e-5&&['FL','FR','RL','RR'].every(s=>j[s+'_hip_joint'].angle===0&&j[s+'_thigh_joint'].angle===.8&&j[s+'_calf_joint'].angle===-1.5)&&JSON.stringify(urdfWorkbench.viewer.robot.position.toArray())===JSON.stringify(standOriginal.p)})()''', 'default stand sets all twelve joints, fixes four foot contacts at ground Z=0, and reports true base height')
    client.js('document.getElementById("motion-undo").click()')
    check(client, '!urdfWorkbench.motion.project.reference&&Object.entries(standOriginal.j).every(([n,a])=>urdfWorkbench.viewer.robot.joints[n].angle===a)', 'undo standing setup restores the previous physical pose and coordinate frame')
    client.js('document.getElementById("motion-redo").click()')
    check(client, 'JSON.stringify('+GROUND+')===JSON.stringify(standGround)&&JSON.stringify(urdfWorkbench.motion.project.reference)===JSON.stringify(standReference)', 'redo restores the exact standing ground reference')
    client.js('urdfWorkbench.viewer.setJointValue("FL_thigh_joint",1.4);document.getElementById("motion-set-ground").click()')
    check(client, 'document.getElementById("motion-status").classList.contains("error")&&Math.abs(urdfWorkbench.viewer.robot.joints.FL_thigh_joint.angle-1.4)<1e-9&&JSON.stringify('+GROUND+')===JSON.stringify(standGround)&&JSON.stringify(urdfWorkbench.motion.project.reference)===JSON.stringify(standReference)', 'non-coplanar stance is rejected without changing the ground or losing the current draft pose')
    client.js('document.getElementById("motion-default-stand").click();document.getElementById("motion-record").click()')
    change(client, 'motion-root-2', .9)
    change(client, 'motion-root-4', 180)
    client.js('document.getElementById("motion-record").click();window.standKeys=JSON.stringify(urdfWorkbench.motion.project.keyframes);document.getElementById("motion-default-stand").click()')
    check(client, 'JSON.stringify(urdfWorkbench.motion.project.keyframes)===standKeys&&JSON.stringify('+GROUND+')===JSON.stringify(standGround)&&document.getElementById("motion-set-ground").disabled&&Math.abs(Number(document.getElementById("motion-root-2").value)-standReference.initialStandingPose.base.position[2])<1e-5', 'default stand on an existing clip returns to its ground without shifting the reference or recorded keys')
    client.js('window.standSaved=urdfWorkbench.motion.project;document.getElementById("motion-cursor").value=".25";document.getElementById("motion-cursor").dispatchEvent(new Event("input"));urdfWorkbench.viewer._updateEnvironment();document.getElementById("motion-fit").click()')
    check(client, 'JSON.stringify('+GROUND+')===JSON.stringify(standGround)&&Math.abs(urdfWorkbench.viewer.robot.position.z-(standReference.origin[2]+(standSaved.keyframes[0].base.position[2]+.9)/2))<1e-7', 'airborne interpolation converts ground-relative base coordinates to the native model without moving the floor')
    check(client, '''(async()=>{const C=createTrajectoryCore(parcelRequire('../node_modules/three/build/three.module.js')),d=await C.sample(standSaved,10);return d.metadata.ground_plane.offset===0&&d.metadata.ground_plane.normal[2]===1&&d.metadata.reference_transform.origin[2]===standReference.origin[2]&&d.frames[0].base_position[2]>.3&&d.frames.at(-1).base_position[2]===.9})()''', 'sampled references carry Z-up ground coordinates and the native-frame reference transform')
    client.js('''document.getElementById('motion-close').click();urdfWorkbench.viewer.plane.position.y+=2;(()=>{const d=new DataTransfer();d.items.add(new File([JSON.stringify(standSaved)],'standing.motion.json'));const i=document.getElementById('motion-import');i.files=d.files;i.dispatchEvent(new Event('change'))})()''');rendered_frame(client)
    check(client, 'JSON.stringify('+GROUND+')===JSON.stringify(standGround)&&Math.abs(urdfWorkbench.viewer.robot.position.z-standOriginal.p[2])<1e-7', 'import restores the saved reference floor and first physical pose even after the scene floor moved')
    client.js('''document.getElementById('animate').click();window.standConfirm=window.confirm;window.confirm=()=>true;document.getElementById('motion-new').click();window.confirm=standConfirm;document.getElementById('motion-default-stand').click()''')
    client.screenshot(str(output/'motion-standing-reference.png'))
    client.js('document.getElementById("motion-close").click()')
    print('PASS standing pose, contact calibration, reference height, fixed-ground interpolation/export/import and setup undo/redo', flush=True)
