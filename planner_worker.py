"""Web adapter for go2_flip_TO quad_backflip.py. Run only in an isolated process.

The phase schedule and two-stage crouch solve are adapted from that BSD-2-Clause
project; see third_party/go2_flip_TO-LICENSE. No upstream files are modified.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from planner_protocol import validate_config, description_hash, model_xml

def event(**data):
    print('@planner '+json.dumps(data,allow_nan=False),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--config',required=True);parser.add_argument('--result',required=True);parser.add_argument('--check-jacobian',action='store_true')
    args=parser.parse_args();cfg=validate_config(json.loads(Path(args.config).read_text()))
    root=Path(args.root).resolve();planning_hash=description_hash(model_xml(root))
    for relative in ['src','src/nltrajopt','src/terrain','src/examples/agile_exps']:sys.path.insert(0,str(root/relative))
    # Upstream params parses argv at import; never pass the web worker's flags to it.
    sys.argv=[sys.argv[0]]
    import numpy as np
    import pinocchio as pin
    from trajectory_optimization import NLTrajOpt
    from node import Node
    from constraint_models import (WholeBodyDynamics,TimeConstraint,SemiEulerIntegration,TerrainGridContactConstraints,TerrainBodyClearanceConstraints,Go2LeftRightLegSymmetryConstraints,TerrainGridFrictionConstraints,Go2MeshCollisionConstraints,BaseAngularVelocityLimitConstraints)
    from robots.go2.Go2Wrapper import Go2
    from terrain.terrain_grid import TerrainGrid
    from _go2_flip_ground_clearance import go2_foot_mesh_standoff_m,apply_joint_velocity_cap,stand_dwell_costs,LEG_SEGMENT_FRAMES
    import utils as reprutils
    import params
    from planner_sparse import SparseJacobian
    params.MAX_CPU_TIME=cfg['wallTime']
    started=time.monotonic();event(stage='building',message='Building Go2 whole-body dynamics and contact constraints')
    robot=Go2();model=robot.model
    model.gravity.linear=np.array([0,0,-cfg['gravity']]);model.effortLimit=np.asarray(model.effortLimit)*cfg['torqueScale']
    q=robot.go_neutral()
    for side in ['FL','FR','RL','RR']:
        for part,setting in [('hip','initialHip'),('thigh','initialThigh'),('calf','initialCalf')]:
            j=model.joints[model.getJointId(side+'_'+part+'_joint')];q[j.idx_q]=cfg[setting]
    # Calibrate stance frame height using the actual foot meshes and the selected pose.
    robot.fk_all(q);stance=go2_foot_mesh_standoff_m()+.002
    foot_names=robot.left_foot_frames+robot.right_foot_frames+robot.left_gripper_frames+robot.right_gripper_frames
    heights=[robot.data.oMf[model.getFrameId(n)].translation[2] for n in foot_names]
    if max(heights)-min(heights)>.005:raise ValueError('Initial feet are not coplanar')
    q[2]+=stance-min(heights)
    terrain=TerrainGrid(10,10,cfg['friction'],-5,-5,5,5);terrain.set_zero()
    mesh=Go2MeshCollisionConstraints(robot,q,self_margin=cfg['selfMargin'],ground_margin=cfg['groundMargin'],auto_slack=.0005,ground_z=0) if cfg['meshCollision'] else None
    clearance={'base':cfg['bodyClearance'],**{n:cfg['legClearance'] for n in LEG_SEGMENT_FRAMES}}
    sequence=[];ranges={}
    for phase,duration in cfg['phases'].items():
        begin=len(sequence);sequence.extend([([] if phase=='flight' else foot_names)]*round(duration/cfg['dt']));ranges[phase]=(begin,len(sequence))
    flight_start,flight_end=ranges['flight'];mid=(flight_start+flight_end-1)//2
    class WebNode(Node):
        def init_node_ids(self,v_id,c_id,k):
            super().init_node_ids(v_id,c_id,k)
            self.web_pose=slice(c_id+self.c_dim,c_id+self.c_dim+4) if cfg['waypointEnabled'] and k==mid else None
            if self.web_pose:self.c_dim+=4
    target_rotation=pin.rpy.rpyToMatrix(np.deg2rad([cfg['waypointRoll'],cfg['waypointPitch'],cfg['waypointYaw']]))
    class AttitudeWaypoint:
        def compute_constraints(self,node,next_node,w,c,model,data):
            if node.web_pose is None:return
            pose=reprutils.rep2pin(w[node.q_id]);R=pin.Quaternion(pose[3:7]).matrix()
            c[node.web_pose]=np.r_[pin.log3(target_rotation.T@R),pose[2]]
        def compute_jacobians(self,node,next_node,w,jac,model,data):
            if node.web_pose is None:return
            pose=reprutils.rep2pin(w[node.q_id]);R=pin.Quaternion(pose[3:7]).matrix();J=pin.Jexp6(w[node.q_id][:6])
            jac[node.web_pose.start:node.web_pose.start+3,node.q_id.start:node.q_id.start+6]=pin.Jlog3(target_rotation.T@R)@J[3:,:]
            jac[node.web_pose.stop-1,node.q_id.start:node.q_id.start+6]=(R@J[:3,:])[2]
        def get_structure_ids(self,node,next_node,rows,cols):
            if node.web_pose:
                for i in range(node.web_pose.start,node.web_pose.stop):
                    for j in range(node.q_id.start,node.q_id.start+6):rows.append(i);cols.append(j)
        def get_bounds(self,node,lb,ub,clb,cub,model):
            if node.web_pose:
                tol=np.deg2rad(cfg['waypointTolerance']);clb[node.web_pose]=[-tol,-tol,-tol,cfg['waypointHeight']];cub[node.web_pose]=[tol,tol,tol,None]
    stages=[]
    for names in sequence:
        node=WebNode(nv=model.nv,contact_phase_fnames=names,contact_fnames=foot_names,terrain_body_clearance=clearance,go2_lr_leg_symmetry=cfg['symmetry'],go2_mesh_collision_rows=mesh.n_rows if mesh else None,base_angular_velocity_limit=True)
        dyn=WholeBodyDynamics();node.dynamics_type=dyn.name
        node.constraints_list=[dyn,TimeConstraint(cfg['dt'],cfg['dt'],None),SemiEulerIntegration(),TerrainGridContactConstraints(terrain,swing_min_clearance=cfg['swingClearance'],stance_min_clearance=stance),TerrainBodyClearanceConstraints(terrain),TerrainGridFrictionConstraints(terrain),BaseAngularVelocityLimitConstraints(cfg['angularSpeed']),AttitudeWaypoint()]
        if cfg['symmetry']:node.constraints_list.append(Go2LeftRightLegSymmetryConstraints(model))
        if mesh:node.constraints_list.append(mesh)
        stages.append(node)
    costs=stand_dwell_costs(model,q)
    for i in list(range(*ranges['stand']))+list(range(*ranges['settle'])):stages[i].costs_list.extend(costs)
    class WebOpt(NLTrajOpt):
        stage='A';round=1
        def jacobian(self,w):
            if not hasattr(self,'web_jacobian'):self.web_jacobian=SparseJacobian(self.row_ids,self.col_ids,(self.cons_dim,self.vars_dim))
            jac=self.web_jacobian;jac.clear()
            for k,node in enumerate(self.nodes):
                next_node=self.nodes[k+1] if k+1<len(self.nodes) else None
                for constraint in node.constraints_list:constraint.compute_jacobians(node,next_node,w,jac,self.model,self.data)
            return jac.result()
        def intermediate(self,alg_mod,iter_count,obj_value,inf_pr,inf_du,*other):
            self.iter_count=iter_count
            def finite(v):return float(v) if np.isfinite(v) else None
            event(stage=self.stage,round=self.round,iteration=int(iter_count),objective=finite(obj_value),primal=finite(inf_pr),dual=finite(inf_du),elapsed=time.monotonic()-started)
    opt=WebOpt(model,stages,cfg['dt']);opt.set_initial_pose(q)
    qf=q.copy();qf[:3]+=np.array([cfg['targetX'],cfg['targetY'],cfg['targetZ']]);qf[3:7]=pin.Quaternion(pin.rpy.rpyToMatrix(np.deg2rad([cfg['targetRoll'],cfg['targetPitch'],cfg['targetYaw']]))).coeffs()
    opt.set_target_pose(qf)
    if not cfg['lockLandingJoints']:
        last=opt.nodes[-1];opt.lb[last.q_id.start+6:last.q_id.stop]=model.lowerPositionLimit[7:];opt.ub[last.q_id.start+6:last.q_id.stop]=model.upperPositionLimit[7:]
    apply_joint_velocity_cap(opt,cfg['jointSpeed'])
    for i,node in enumerate(opt.nodes):
        if flight_start<=i<=flight_end and flight_end>flight_start:
            s=(i-flight_start)/(flight_end-flight_start);pitch=(2*np.pi*s if cfg['motion']=='frontflip' else -2*np.pi*s) if cfg['motion'] in ['frontflip','backflip'] else 0
            guess=q.copy();guess[:3]=(1-s)*q[:3]+s*qf[:3];opt.x0[node.q_id]=reprutils.rpy2rep(guess,[0,pitch,0])
    if args.check_jacobian:
        sparse=opt.jacobian(opt.x0);dense=NLTrajOpt.jacobian(opt,opt.x0)
        difference=float(np.max(np.abs(sparse-dense)))
        attitude_error=0.
        pose_node=next((n for n in opt.nodes if n.web_pose),None)
        if pose_node is not None:
            full=opt.jac_test(opt.x0)[pose_node.web_pose,pose_node.q_id.start:pose_node.q_id.start+6]
            numerical=np.zeros_like(full)
            for i in range(6):
                j=pose_node.q_id.start+i;step=1e-6;a=opt.x0.copy();b=opt.x0.copy();a[j]+=step;b[j]-=step
                numerical[:,i]=(opt.constraints(a)[pose_node.web_pose]-opt.constraints(b)[pose_node.web_pose])/(2*step)
            attitude_error=float(np.max(np.abs(numerical-full)))
        if difference>1e-10 or attitude_error>1e-5:raise RuntimeError(f'Jacobian audit failed: {difference}, {attitude_error}')
        Path(args.result).write_text(json.dumps({'sparseDenseMaxError':difference,'attitudeFiniteDifferenceError':attitude_error,'nonzeros':len(sparse),'denseEntries':opt.cons_dim*opt.vars_dim}))
        event(stage='complete',message='Sparse/dense and attitude finite-difference Jacobian checks passed');return
    rounds=[]
    for stage in (['A','B'] if cfg['motion']!='stand' and ranges['crouch'][1]>ranges['crouch'][0] else ['A']):
        opt.stage=stage
        if stage=='B':
            start,end=ranges['crouch']
            for k in range(start,end):
                thigh=cfg['initialThigh']+(cfg['crouchThigh']-cfg['initialThigh'])*(k-start+1)/(end-start)
                for side in ['FL','FR','RL','RR']:
                    j=model.joints[model.getJointId(side+'_thigh_joint')];idx=opt.nodes[k].q_id.start+6+j.idx_q-7
                    opt.lb[idx]=max(opt.lb[idx] if opt.lb[idx] is not None else -np.inf,thigh)
            opt.x0=np.clip(np.asarray(opt.sol),np.array([v if v is not None else -np.inf for v in opt.lb]),np.array([v if v is not None else np.inf for v in opt.ub]))
        for round_number in range(1,cfg['rounds']+1):
            opt.round=round_number;event(stage=stage,round=round_number,iteration=0,message='Solving whole-body NLP')
            result=opt.solve(cfg['maxIterations'],cfg['tolerance'],True,print_level=0,accept_max_iter_exceeded=True)
            rounds.append({'stage':stage,'round':round_number,'iterations':opt.iter_count,'seconds':result['solve_time'],'status':int(opt.info['status'])})
            if not result.get('warning'):break
            opt.x0=np.asarray(opt.sol).copy()
    event(stage='validating',message='Checking all variable and dynamics/contact/collision constraint residuals')
    c=opt.constraints(opt.sol)
    def violation(x,lo,hi):
        lo=np.array([-np.inf if v is None else v for v in lo]);hi=np.array([np.inf if v is None else v for v in hi]);return float(max(0,np.max(lo-x),np.max(x-hi)))
    residual=max(violation(c,opt.clb,opt.cub),violation(opt.sol,opt.lb,opt.ub))
    if not np.all(np.isfinite(opt.sol)) or not np.all(np.isfinite(c)) or int(opt.info['status']) not in [0,1] or residual>cfg['tolerance']:raise RuntimeError(f'No feasible converged plan: IPOPT status {opt.info["status"]}, maximum residual {residual:.6g}')
    frames=[];elapsed=0.;rotation_sum=np.zeros(3);lastR=None;last_rpy=np.zeros(3)
    for node,decoded in zip(opt.nodes,result['nodes']):
        qn=decoded['q'];R=pin.Quaternion(qn[3:7]).matrix();rpy=pin.rpy.matrixToRpy(R)
        alternatives=[rpy,np.array([rpy[0]+np.pi,np.pi-rpy[1],rpy[2]+np.pi])]
        alternatives=[a+2*np.pi*np.round((last_rpy-a)/(2*np.pi)) for a in alternatives];rpy=min(alternatives,key=lambda a:np.linalg.norm(a-last_rpy));last_rpy=rpy
        if lastR is not None:rotation_sum+=pin.log3(R@lastR.T)
        lastR=R
        pin.forwardKinematics(model,robot.data,qn);pin.updateFramePlacements(model,robot.data)
        forces={name:(robot.data.oMf[model.getFrameId(name)].rotation@force).tolist() for name,force in decoded['forces'].items()}
        tau=c[node.c_dh_id][6:]
        v=decoded['v'];a=decoded['a'];linear=R@v[:3];angular=R@v[3:6]
        frames.append({'time':elapsed,'base_position':qn[:3].tolist(),'base_quaternion':qn[3:7].tolist(),'base_rpy':rpy.tolist(),'joint_positions':qn[7:].tolist(),'joint_velocities':v[6:].tolist(),'joint_accelerations':a[6:].tolist(),'base_linear_velocity':linear.tolist(),'base_angular_velocity':angular.tolist(),'base_linear_acceleration':(R@(a[:3]+np.cross(v[3:6],v[:3]))).tolist(),'base_angular_acceleration':(R@a[3:6]).tolist(),'contact_forces':forces,'joint_torques':tau.tolist()});elapsed+=decoded['dt']
    expected=-2*np.pi if cfg['motion']=='backflip' else 2*np.pi
    if cfg['motion'] in ['backflip','frontflip'] and abs(rotation_sum[1]-expected)>.35:raise RuntimeError('Solver converged without the requested flip winding; adjust flight or attitude waypoint')
    if description_hash(model_xml(root))!=planning_hash:raise RuntimeError('Robot description changed during planning; restart the plan')
    payload={'schema':'urdf-viewer.whole-body.v1','modelHash':planning_hash,'jointNames':[str(model.names[i]) for i in range(2,model.njoints)],'frames':frames,'settings':cfg,'report':{'solver':'Pinocchio / cyipopt IPOPT','sourceScript':'src/examples/agile_exps/quad_backflip.py','status':int(opt.info['status']),'maximumConstraintViolation':residual,'tolerance':cfg['tolerance'],'solveSeconds':time.monotonic()-started,'rounds':rounds,'pitchRotation':float(rotation_sum[1]),'meshCollision':cfg['meshCollision'],'discreteDynamicsFeasible':True,'continuousDynamicsValidated':False}}
    Path(args.result).write_text(json.dumps(payload,allow_nan=False));event(stage='complete',message='Feasible discrete whole-body plan ready',result=True)

if __name__=='__main__':
    try:main();sys.stdout.flush();os._exit(0)
    except Exception as error:
        import traceback
        traceback.print_exc();event(stage='failed',message=f'{type(error).__name__}: {error}');sys.stdout.flush();os._exit(1)
