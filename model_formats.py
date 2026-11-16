"""Portable rigid-body model interchange. Native readers: MuJoCo and OpenUSD.

Canonical frames use metres, kilograms, radians and Z-up. Only tree joints can
be represented by URDF; unsupported constraints are reported or rejected.
"""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import math
import re
import warnings
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


def vec(value, default=(0., 0., 0.)):
    a = np.asarray(default if value is None else ([float(x) for x in value.split()] if isinstance(value, str) else value), dtype=float)
    if not np.isfinite(a).all():
        raise ValueError('Non-finite numeric model data')
    return a


def transform(position=None, quaternion=None, rpy=None):
    t = np.eye(4)
    if position is not None:
        t[:3, 3] = vec(position)
    if quaternion is not None:
        q = vec(quaternion, (1, 0, 0, 0))
        t[:3, :3] = Rotation.from_quat([*q[1:], q[0]]).as_matrix()
    elif rpy is not None:
        t[:3, :3] = Rotation.from_euler('xyz', vec(rpy)).as_matrix()
    return t


def numbers(a):
    return ' '.join(format(float(x), '.12g') for x in np.asarray(a).ravel())


def quat(t):
    q = Rotation.from_matrix(np.asarray(t)[:3, :3]).as_quat()
    return [q[3], *q[:3]]


def xml_origin(parent, t):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        angles = Rotation.from_matrix(t[:3, :3]).as_euler('xyz')
    ET.SubElement(parent, 'origin', xyz=numbers(t[:3, 3]), rpy=numbers(angles))


def safe_name(name):
    return re.sub(r'[^A-Za-z0-9_]', '_', name) or 'model'


@dataclass
class Shape:
    kind: str
    size: object = None
    pose: object = field(default_factory=lambda: np.eye(4))
    color: object = field(default_factory=lambda: np.array([.72, .78, .84, 1.]))
    mesh: object = None


@dataclass
class Link:
    name: str
    mass: float = 0.
    com: object = field(default_factory=lambda: np.zeros(3))
    inertia: object = field(default_factory=lambda: np.zeros((3, 3)))
    visuals: list = field(default_factory=list)
    collisions: list = field(default_factory=list)


@dataclass
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    pose: object = field(default_factory=lambda: np.eye(4))
    axis: object = field(default_factory=lambda: np.array([1., 0., 0.]))
    lower: float = 0.
    upper: float = 0.
    effort: float = 0.
    velocity: float = 0.
    damping: float = 0.
    friction: float = 0.
    mimic: object = None


@dataclass
class Model:
    name: str
    links: dict = field(default_factory=dict)
    joints: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def note(self, text):
        if text not in self.notes:
            self.notes.append(text)

    def validate(self):
        if not self.links:
            raise ValueError('No rigid bodies or geometry found')
        children = set()
        names = set()
        for j in self.joints:
            if j.name in names or j.child in children or j.parent == j.child or j.parent not in self.links or j.child not in self.links:
                raise ValueError('Duplicate, cyclic or multiple-parent joints cannot form a URDF tree')
            names.add(j.name)
            children.add(j.child)
            if j.kind not in ['fixed', 'revolute', 'continuous', 'prismatic', 'floating']:
                raise ValueError('Unsupported joint type: ' + j.kind)
            if j.kind in ['revolute', 'prismatic'] and (not np.isfinite([j.lower,j.upper]).all() or j.lower > j.upper):
                raise ValueError('Invalid joint limits: ' + j.name)
            if not np.isfinite(j.pose).all():
                raise ValueError('Non-finite joint pose: '+j.name)
            if j.kind in ['fixed','floating'] and np.linalg.norm(j.axis)<1e-10:
                j.axis=np.array([1.,0.,0.])
            if np.linalg.norm(j.axis) < 1e-10:
                raise ValueError('Zero joint axis: ' + j.name)
            j.axis = vec(j.axis) / np.linalg.norm(j.axis)
        roots = set(self.links) - children
        if len(roots) != 1:
            raise ValueError('The model must have one connected root')
        root = next(iter(roots))
        reached = {root}
        while True:
            new = {j.child for j in self.joints if j.parent in reached} - reached
            if not new:
                break
            reached.update(new)
        if len(reached) != len(self.links):
            raise ValueError('Closed loops / disconnected joints cannot be converted to a tree')
        for link in self.links.values():
            if not np.isfinite([link.mass, *link.com, *link.inertia.ravel()]).all() or link.mass < 0:
                raise ValueError('Invalid mass or inertia: ' + link.name)
        return root

    def world_poses(self):
        root = self.validate()
        poses = {root: np.eye(4)}
        while len(poses) < len(self.links):
            for j in self.joints:
                if j.parent in poses and j.child not in poses:
                    poses[j.child] = poses[j.parent] @ j.pose
        return poses


class Resources:
    def __init__(self, root, entry):
        self.root = Path(root).resolve()
        self.entry = self.checked(entry)
        self._meshes = {}

    def checked(self, path):
        p = Path(path).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError('Resources must remain inside the uploaded model folder')
        return p

    def resolve(self, reference, base=None):
        reference = reference.replace('\\', '/')
        if reference.startswith('package://'):
            key = reference[len('package://'):]
            choices = [p for p in self.root.rglob('*') if p.is_file() and (p.relative_to(self.root).as_posix() == key or p.relative_to(self.root).as_posix().endswith('/' + key))]
            if len(choices) == 1:
                return choices[0]
        elif not re.match(r'^[A-Za-z]+:', reference):
            p = self.checked((base or self.entry.parent) / reference)
            if p.is_file():
                return p
        if reference.startswith('package://') or not re.match(r'^[A-Za-z]+:',reference):
            key=reference.removeprefix('package://').lstrip('/')
            choices=[p for p in self.root.rglob('*') if p.is_file() and p.relative_to(self.root).as_posix().endswith('/'+key)]
            if len(choices)==1:return choices[0]
            choices=[p for p in self.root.rglob(Path(key).name) if p.is_file()]
            if len(choices)==1:return choices[0]
        raise ValueError('Missing or unsupported resource: ' + reference + '. Load the complete model folder.')

    def mesh(self, reference, scale):
        p = self.resolve(reference)
        key=(str(p),tuple(vec(scale,(1,1,1))))
        if key in self._meshes:return self._meshes[key]
        self.check_mesh_dependencies(p)
        loaded = trimesh.load(str(p), force='scene', process=False)
        mesh = loaded.to_mesh() if isinstance(loaded, trimesh.Scene) else loaded
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            raise ValueError('Empty or unsupported mesh: ' + reference)
        mesh = mesh.copy()
        if p.suffix.lower()=='.dae':
            asset=ET.parse(p).getroot()
            namespace=asset.tag.split('}')[0]+'}' if '}' in asset.tag else ''
            unit=asset.find('.//'+namespace+'asset/'+namespace+'unit')
            if unit is not None:mesh.apply_scale(float(unit.get('meter',1)))
            # URDFLoader clears the COLLADA scene-root quaternion. URDF mesh
            # vertices stay in their authored link coordinates; retain only units.
        mesh.apply_scale(vec(scale, (1, 1, 1)))
        if not np.isfinite(mesh.vertices).all():raise ValueError('Non-finite mesh vertices: '+reference)
        self._meshes[key]=mesh
        return mesh


    def check_mesh_dependencies(self,path):
        suffix=path.suffix.lower()
        if suffix=='.dae':
            document=ET.parse(path).getroot()
            for image in document.iter():
                if image.tag.split('}')[-1]=='image':
                    for e in image.iter():
                        if e.tag.split('}')[-1]=='init_from' and e.text and not e.text.strip().startswith('#'):
                            self.resolve(e.text.strip(),path.parent)
        elif suffix=='.obj':
            import shlex
            for line in path.read_text(errors='replace').splitlines():
                if line.strip().startswith('mtllib '):
                    for name in shlex.split(line.strip())[1:]:
                        material=self.resolve(name,path.parent)
                        for record in material.read_text(errors='replace').splitlines():
                            if record.lstrip().startswith(('map_','bump ','disp ','decal ')):
                                words=shlex.split(record)
                                if len(words)>1:self.resolve(words[-1],material.parent)
        elif suffix in ['.gltf','.glb']:
            if suffix=='.gltf':document=json.loads(path.read_text())
            else:
                import struct
                raw=path.read_bytes();size,kind=struct.unpack('<II',raw[12:20])
                if kind!=0x4e4f534a:raise ValueError('Invalid GLB JSON chunk')
                document=json.loads(raw[20:20+size])
            for asset in document.get('buffers',[])+document.get('images',[]):
                uri=asset.get('uri')
                if uri and not uri.startswith('data:'):self.resolve(uri,path.parent)


def read_urdf(resources):
    root = ET.parse(resources.entry).getroot()
    if root.tag != 'robot':
        raise ValueError('Expected a URDF <robot> root')
    model = Model(root.get('name', resources.entry.stem))
    materials = {e.get('name'): e for e in root.findall('material')}
    def pose(e):
        o = e.find('origin')
        return transform(o.get('xyz') if o is not None else None, rpy=o.get('rpy') if o is not None else None)
    for e in root.findall('link'):
        name = e.get('name')
        if not name or name in model.links:
            raise ValueError('Link names must be unique and nonempty')
        link = model.links[name] = Link(name)
        inertial = e.find('inertial')
        if inertial is not None:
            mass, tensor = inertial.find('mass'), inertial.find('inertia')
            if mass is None or tensor is None:
                raise ValueError('Incomplete inertial: ' + name)
            link.mass = float(mass.get('value'))
            t = pose(inertial)
            link.com = t[:3, 3]
            get = lambda k: float(tensor.get(k, 0))
            i = np.array([[get('ixx'),get('ixy'),get('ixz')],[get('ixy'),get('iyy'),get('iyz')],[get('ixz'),get('iyz'),get('izz')]])
            link.inertia = t[:3, :3] @ i @ t[:3, :3].T
        for role in ['visual', 'collision']:
            for g in e.findall(role):
                container = g.find('geometry')
                if container is None or len(container) != 1:
                    raise ValueError('Invalid link geometry: ' + name)
                geo = container[0]
                material = g.find('material')
                if material is not None and material.find('color') is None:
                    material = materials.get(material.get('name'), material)
                color = material.find('color') if material is not None else None
                shape = Shape(geo.tag, pose=pose(g), color=vec(color.get('rgba'), (.72,.78,.84,1)) if color is not None else vec(None, (.72,.78,.84,1)))
                if material is not None and material.find('texture') is not None:
                    model.note('Texture maps are baked to a constant display color; texture shaders are not transferred.')
                if geo.tag == 'mesh':
                    shape.mesh = resources.mesh(geo.get('filename', ''), geo.get('scale'))
                    model.note('Mesh-internal material groups use the link display RGBA; per-face materials and texture shaders are not transferred.')
                elif geo.tag == 'box':
                    shape.size = vec(geo.get('size'))
                elif geo.tag in ['sphere', 'cylinder']:
                    shape.size = [float(geo.get('radius'))] + ([float(geo.get('length'))] if geo.tag == 'cylinder' else [])
                else:
                    raise ValueError('Unsupported URDF geometry: ' + geo.tag)
                getattr(link, role + 's' if role == 'visual' else 'collisions').append(shape)
    for e in root.findall('joint'):
        kind = e.get('type')
        if kind == 'planar':
            raise ValueError('Planar joints cannot be mapped to a single portable joint')
        axis, limit, dynamics, mimic = e.find('axis'), e.find('limit'), e.find('dynamics'), e.find('mimic')
        j = Joint(e.get('name'), kind, e.find('parent').get('link'), e.find('child').get('link'), pose(e), vec(axis.get('xyz') if axis is not None else None, (1,0,0)))
        for attr in ['lower','upper','effort','velocity']:
            setattr(j, attr, float(limit.get(attr, 0)) if limit is not None else 0.)
        if dynamics is not None:
            j.damping, j.friction = float(dynamics.get('damping', 0)), float(dynamics.get('friction', 0))
        if mimic is not None:
            j.mimic = {'joint':mimic.get('joint'), 'multiplier':float(mimic.get('multiplier',1)), 'offset':float(mimic.get('offset',0))}
        model.joints.append(j)
    if any(root.find(tag) is not None for tag in ['transmission','gazebo','ros2_control']):
        model.note('URDF transmission, Gazebo and ros2_control extensions are not transferred.')
    model.validate()
    return model


def xml_text(root):
    ET.indent(root, space='  ')
    return ET.tostring(root, encoding='unicode', xml_declaration=True) + '\n'


def write_urdf(model, directory):
    model.validate()
    root = ET.Element('robot', name=model.name)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    meshes = {}
    for link in model.links.values():
        e = ET.SubElement(root, 'link', name=link.name)
        if link.mass > 0:
            inertial = ET.SubElement(e, 'inertial')
            xml_origin(inertial, transform(link.com))
            ET.SubElement(inertial, 'mass', value=numbers([link.mass]))
            i = link.inertia
            ET.SubElement(inertial, 'inertia', **{k: numbers([v]) for k,v in zip(['ixx','iyy','izz','ixy','ixz','iyz'],[i[0,0],i[1,1],i[2,2],i[0,1],i[0,2],i[1,2]])})
        for role, shapes in [('visual',link.visuals), ('collision',link.collisions)]:
            for shape in shapes:
                g = ET.SubElement(e, role)
                xml_origin(g, shape.pose)
                geo = ET.SubElement(g, 'geometry')
                if shape.kind == 'mesh':
                    digest = hashlib.sha256(shape.mesh.vertices.tobytes() + shape.mesh.faces.tobytes()).hexdigest()[:20]
                    path = 'meshes/' + digest + '.obj'
                    if path not in meshes:
                        (directory/'meshes').mkdir(exist_ok=True)
                        shape.mesh.export(str(directory/path), include_color=False, include_texture=False)
                        meshes[path] = True
                    ET.SubElement(geo,'mesh',filename=path)
                elif shape.kind == 'box':
                    ET.SubElement(geo,'box',size=numbers(shape.size))
                elif shape.kind == 'sphere':
                    ET.SubElement(geo,'sphere',radius=numbers([shape.size[0]]))
                elif shape.kind == 'cylinder':
                    ET.SubElement(geo,'cylinder',radius=numbers([shape.size[0]]),length=numbers([shape.size[1]]))
                if role == 'visual':
                    ET.SubElement(ET.SubElement(g,'material',name='display'),'color',rgba=numbers(shape.color))
    for j in model.joints:
        e = ET.SubElement(root,'joint',name=j.name,type=j.kind)
        ET.SubElement(e,'parent',link=j.parent)
        ET.SubElement(e,'child',link=j.child)
        xml_origin(e,j.pose)
        if j.kind not in ['fixed','floating']:
            ET.SubElement(e,'axis',xyz=numbers(j.axis))
            limits={'effort':numbers([j.effort]),'velocity':numbers([j.velocity])}
            if j.kind != 'continuous':
                limits.update(lower=numbers([j.lower]), upper=numbers([j.upper]))
            ET.SubElement(e,'limit',**limits)
            if j.damping or j.friction:
                ET.SubElement(e,'dynamics',damping=numbers([j.damping]),friction=numbers([j.friction]))
        if j.mimic:
            ET.SubElement(e,'mimic',**{k:str(v) for k,v in j.mimic.items()})
    path = directory/'model.urdf'
    path.write_text(xml_text(root))
    return path


def check_mjcf_dependencies(resources):
    seen=set();assets=[];compiler={}
    def visit(path):
        path=resources.checked(path)
        if path in seen:return
        seen.add(path)
        root=ET.parse(path).getroot()
        if root.find('extension') is not None:
            raise ValueError('MJCF plugins are not supported by portable interchange')
        for e in root.iter():
            if e.tag=='compiler':compiler.update(e.attrib)
            ref=e.get('file')
            if not ref:continue
            if e.tag=='include':visit(resources.resolve(ref,path.parent))
            elif e.tag=='model':raise ValueError('MJCF asset-model attachments must be flattened before interchange')
            else:assets.append((e.tag,ref))
        return root
    root=visit(resources.entry)
    # MuJoCo asset paths are resolved from the main XML, including assets
    # declared by include files; include paths are relative to the including file.
    for kind,ref in assets:
        directory=compiler.get('assetdir','') or compiler.get('texturedir' if kind=='texture' else 'meshdir','')
        base=resources.checked(resources.entry.parent/directory)
        resources.resolve(ref,base)
    return root


def read_mjcf(resources):
    import mujoco as mj
    xml = check_mjcf_dependencies(resources)
    m = mj.MjModel.from_xml_path(str(resources.entry))
    data = mj.MjData(m)
    mj.mj_forward(m,data)
    model = Model(xml.get('model',resources.entry.stem))
    metadata = {}
    for e in xml.findall('./custom/text'):
        if e.get('name') == 'urdf_viewer_metadata':
            metadata = json.loads(e.get('data','{}'))
    body_names = {i:(mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,i) or 'body_'+str(i)) for i in range(m.nbody)}
    body_names[0] = metadata.get('worldRoot') or body_names[0]
    while body_names[0] in [body_names[i] for i in body_names if i]:body_names[0]='_'+body_names[0]
    if len(set(body_names.values())) != len(body_names):
        raise ValueError('Duplicate MuJoCo body names')
    body_world = {i:transform(data.xpos[i]) for i in range(m.nbody)}
    for i in body_world:
        body_world[i][:3,:3] = data.xmat[i].reshape(3,3)
    frames = {i:body_world[i].copy() for i in body_world}
    body_joints = {i:list(range(int(m.body_jntadr[i]),int(m.body_jntadr[i]+m.body_jntnum[i]))) if m.body_jntnum[i] else [] for i in range(m.nbody)}
    for i, joints in body_joints.items():
        if any(int(m.jnt_type[k]) == int(mj.mjtJoint.mjJNT_BALL) for k in joints):
            raise ValueError('Ball joints require a spherical-joint representation and cannot be converted to URDF faithfully')
        if len(joints) == 1 and int(m.jnt_type[joints[0]]) != int(mj.mjtJoint.mjJNT_FREE):
            frames[i] = body_world[i] @ transform(m.jnt_pos[joints[0]])
    def joint(k,parent,child,t):
        jt = int(m.jnt_type[k])
        name = mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,k) or 'joint_'+str(k)
        kind = 'floating' if jt==int(mj.mjtJoint.mjJNT_FREE) else 'prismatic' if jt==int(mj.mjtJoint.mjJNT_SLIDE) else 'revolute' if m.jnt_limited[k] else 'continuous'
        j = Joint(name,kind,parent,child,t,vec(m.jnt_axis[k]) if kind!='floating' else vec(None,(1,0,0)))
        if kind!='floating':
            ref = m.qpos0[m.jnt_qposadr[k]]
            j.lower,j.upper = (m.jnt_range[k]-ref).tolist() if m.jnt_limited[k] else (0.,0.)
            j.damping = float(m.dof_damping[m.jnt_dofadr[k]])
            j.friction = float(m.dof_frictionloss[m.jnt_dofadr[k]])
            if hasattr(m,'jnt_actfrclimited') and m.jnt_actfrclimited[k]:
                j.effort = float(max(abs(m.jnt_actfrcrange[k])))
            elif hasattr(m,'jnt_actfrclimited'):
                for a in range(m.nu):
                    if int(m.actuator_trntype[a])==int(mj.mjtTrn.mjTRN_JOINT) and int(m.actuator_trnid[a,0])==k and m.actuator_forcelimited[a]:
                        j.effort=max(j.effort,float(max(abs(m.actuator_forcerange[a]))*abs(m.actuator_gear[a,0])))
        saved = metadata.get('joints',{}).get(name,{})
        j.velocity=float(saved.get('velocity',0));j.effort=float(saved.get('effort',j.effort));j.mimic=saved.get('mimic')
        model.joints.append(j)
    for i, name in body_names.items():
        link=model.links[name]=Link(name,float(m.body_mass[i]))
        offset=np.linalg.inv(frames[i])@body_world[i]
        link.com=(offset@np.r_[m.body_ipos[i],1])[:3]
        r=offset[:3,:3]@transform(quaternion=m.body_iquat[i])[:3,:3]
        link.inertia=r@np.diag(m.body_inertia[i])@r.T
        if i==0:
            continue
        parent=int(m.body_parentid[i]); js=body_joints[i]
        if len(js)<=1:
            pose=np.linalg.inv(frames[parent])@frames[i]
            if js:
                joint(js[0],body_names[parent],name,pose)
            else:
                model.joints.append(Joint(metadata.get('fixedJointNames',{}).get(name,name+'__fixed'),'fixed',body_names[parent],name,pose))
        else:
            previous=body_names[parent];previous_frame=frames[parent]
            for n,k in enumerate(js):
                child=name+'__dof_'+str(n)
                if child in body_names.values() or child in model.links:
                    raise ValueError('Generated joint-frame name collision')
                model.links[child]=Link(child)
                frame=body_world[i]@transform(m.jnt_pos[k])
                joint(k,previous,child,np.linalg.inv(previous_frame)@frame)
                previous,previous_frame=child,frame
            model.joints.append(Joint(name+'__body','fixed',previous,name,np.linalg.inv(previous_frame)@frames[i]))
            model.note('Multiple joints on one MuJoCo body were expanded into serial URDF joint frames.')
    mesh_cache={}
    for k in range(m.ngeom):
        body=int(m.geom_bodyid[k]);link=model.links[body_names[body]];kind=int(m.geom_type[k]);size=m.geom_size[k]
        if kind==int(mj.mjtGeom.mjGEOM_PLANE):
            model.note('Infinite ground planes are excluded from the robot model.')
            continue
        pose=np.linalg.inv(frames[body])@body_world[body]@transform(m.geom_pos[k],m.geom_quat[k])
        rgba=vec(m.geom_rgba[k])
        material=int(m.geom_matid[k])
        if material>=0 and np.allclose(rgba,[.5,.5,.5,1]):rgba=vec(m.mat_rgba[material])
        s=Shape('mesh',pose=pose,color=rgba)
        if kind==int(mj.mjtGeom.mjGEOM_BOX):s.kind,s.size='box',size*2
        elif kind==int(mj.mjtGeom.mjGEOM_SPHERE):s.kind,s.size='sphere',[size[0]]
        elif kind==int(mj.mjtGeom.mjGEOM_CYLINDER):s.kind,s.size='cylinder',[size[0],2*size[1]]
        elif kind==int(mj.mjtGeom.mjGEOM_MESH):
            idx=int(m.geom_dataid[k])
            if idx not in mesh_cache:
                va,vn=int(m.mesh_vertadr[idx]),int(m.mesh_vertnum[idx]);fa,fn=int(m.mesh_faceadr[idx]),int(m.mesh_facenum[idx])
                mesh_cache[idx]=trimesh.Trimesh(vertices=m.mesh_vert[va:va+vn].copy(),faces=m.mesh_face[fa:fa+fn].copy(),process=False)
            s.mesh=mesh_cache[idx]
        elif kind==int(mj.mjtGeom.mjGEOM_CAPSULE):
            s.mesh=trimesh.creation.capsule(radius=size[0],height=2*size[1],count=[16,16])
            model.note('Capsules / ellipsoids are tessellated into triangle meshes.')
        elif kind==int(mj.mjtGeom.mjGEOM_ELLIPSOID):
            s.mesh=trimesh.creation.icosphere(subdivisions=3);s.mesh.apply_scale(size)
            model.note('Capsules / ellipsoids are tessellated into triangle meshes.')
        else:raise ValueError('Unsupported MuJoCo geometry: '+str(kind))
        if m.geom_group[k]!=3 and s.color[3]>0:link.visuals.append(s)
        if m.geom_contype[k] or m.geom_conaffinity[k]:link.collisions.append(s)
        if int(m.geom_matid[k])>=0:model.note('MuJoCo materials use their compiled RGBA; texture shaders are not transferred.')
    if m.nu or m.nsensor or m.ntendon or m.neq:
        model.note('Controllers, sensors, tendons and solver-specific equality constraints are not transferred; portable mimic metadata is retained when available.')
    if m.nkey:model.note('MuJoCo keyframes / simulation state are not transferred.')
    model.note('MJCF joint velocity limits are unspecified unless supplied by this tool; zero in URDF means unspecified in the conversion report.')
    root0=body_names[0]
    anchors=[j for j in model.joints if j.parent==root0]
    if metadata.get('rootName') and not metadata.get('worldRoot') and len(anchors)==1 and anchors[0].child==metadata['rootName'] and anchors[0].kind=='fixed' and np.allclose(anchors[0].pose,np.eye(4),atol=1e-8) and not model.links[root0].visuals and not model.links[root0].collisions:
        model.joints.remove(anchors[0]);del model.links[root0]
    model.validate()
    return model


def write_mjcf(model,directory):
    import mujoco as mj
    root_name=model.validate();directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    root=ET.Element('mujoco',model=model.name)
    ET.SubElement(root,'compiler',angle='radian',fusestatic='false',inertiafromgeom='false')
    ET.SubElement(root,'option',gravity='0 0 -9.81')
    root_link=model.links[root_name]
    empty_world=not root_link.mass and not root_link.visuals and not root_link.collisions
    meta={'rootName':root_name,'worldRoot':root_name if empty_world else None,'fixedJointNames':{j.child:j.name for j in model.joints if j.kind=='fixed'},'joints':{j.name:{'velocity':j.velocity,'effort':j.effort,'mimic':j.mimic} for j in model.joints}}
    ET.SubElement(ET.SubElement(root,'custom'),'text',name='urdf_viewer_metadata',data=json.dumps(meta,separators=(',',':')))
    assets=ET.SubElement(root,'asset');world=ET.SubElement(root,'worldbody');motors=ET.SubElement(root,'actuator');equality=ET.SubElement(root,'equality')
    meshes={};children={n:[] for n in model.links}
    for j in model.joints:children[j.parent].append(j)
    def build(name,parent,joint=None):
        link=model.links[name];t=joint.pose if joint else np.eye(4)
        body=ET.SubElement(parent,'body',name=name,pos=numbers(t[:3,3]),quat=numbers(quat(t)))
        if joint and joint.kind!='fixed':
            if joint.kind=='floating':
                if parent is not world:raise ValueError('MuJoCo free joints must be at the world root')
                ET.SubElement(body,'freejoint',name=joint.name)
            else:
                attrs={'name':joint.name,'type':'slide' if joint.kind=='prismatic' else 'hinge','axis':numbers(joint.axis),'limited':'false' if joint.kind=='continuous' else 'true','damping':numbers([joint.damping]),'frictionloss':numbers([joint.friction])}
                if joint.kind!='continuous':attrs['range']=numbers([joint.lower,joint.upper])
                ET.SubElement(body,'joint',**attrs)
                if joint.effort>0:ET.SubElement(motors,'motor',joint=joint.name,name=joint.name+'__motor',gear='1',ctrllimited='true',ctrlrange=numbers([-joint.effort,joint.effort]),forcelimited='true',forcerange=numbers([-joint.effort,joint.effort]))
        if link.mass>0:
            vals,axes=np.linalg.eigh(link.inertia)
            if min(vals)<=0 or max(vals)>sum(vals)-max(vals)+1e-8:raise ValueError('Nonphysical inertia tensor: '+name)
            if np.linalg.det(axes)<0:axes[:,0]*=-1
            rot=np.eye(4);rot[:3,:3]=axes
            ET.SubElement(body,'inertial',pos=numbers(link.com),mass=numbers([link.mass]),diaginertia=numbers(vals),quat=numbers(quat(rot)))
        elif joint and joint.kind!='fixed':
            # A serial kinematic frame has no own mass in URDF. MuJoCo requires
            # positive inertia on every moving body; do not silently invent it.
            raise ValueError('MuJoCo requires positive mass and inertia on moving link '+name+'; provide inertial data before conversion')
        for role,shapes in [('visual',link.visuals),('collision',link.collisions)]:
            for shape in shapes:
                attrs={'pos':numbers(shape.pose[:3,3]),'quat':numbers(quat(shape.pose)),'rgba':numbers(shape.color if role=='visual' else [*shape.color[:3],0]),'group':'2' if role=='visual' else '3','contype':'0' if role=='visual' else '1','conaffinity':'0' if role=='visual' else '1'}
                if shape.kind=='mesh':
                    digest=hashlib.sha256(shape.mesh.vertices.tobytes()+shape.mesh.faces.tobytes()).hexdigest()[:20]
                    name_mesh='mesh_'+digest
                    if name_mesh not in meshes:
                        (directory/'meshes').mkdir(exist_ok=True);filename='meshes/'+name_mesh+'.obj'
                        shape.mesh.export(str(directory/filename),include_color=False,include_texture=False)
                        ET.SubElement(assets,'mesh',name=name_mesh,file=filename);meshes[name_mesh]=True
                    attrs.update(type='mesh',mesh=name_mesh)
                elif shape.kind=='box':attrs.update(type='box',size=numbers(np.asarray(shape.size)/2))
                elif shape.kind=='sphere':attrs.update(type='sphere',size=numbers(shape.size))
                elif shape.kind=='cylinder':attrs.update(type='cylinder',size=numbers([shape.size[0],shape.size[1]/2]))
                ET.SubElement(body,'geom',**attrs)
        for child in children[name]:build(child.child,body,child)
    root_link=model.links[root_name]
    # An empty world root allows actual floating joints to stay top-level.
    if not root_link.mass and not root_link.visuals and not root_link.collisions:
        for j in children[root_name]:build(j.child,world,j)
    else:build(root_name,world)
    for j in model.joints:
        if j.mimic:
            m=j.mimic;ET.SubElement(equality,'joint',joint1=j.name,joint2=m['joint'],polycoef=numbers([m.get('offset',0),m.get('multiplier',1),0,0,0]))
    path=directory/'model.xml';path.write_text(xml_text(root))
    mj.MjModel.from_xml_path(str(path))
    model.note('MJCF export uses unit-gear torque motors where effort limits are known; original controllers and simulation settings are not reconstructed.')
    return path


def check_usd_dependencies(resources):
    from pxr import Sdf
    seen=set()
    def visit(path):
        path=resources.checked(path)
        if path in seen:return
        seen.add(path)
        layer=Sdf.Layer.FindOrOpen(str(path))
        if not layer:raise ValueError('OpenUSD could not read layer '+path.name)
        dependencies=set(layer.GetCompositionAssetDependencies())|set(layer.GetExternalAssetDependencies())
        for ref in dependencies:
            if not ref:continue
            dep=resources.resolve(ref,path.parent)
            if dep.suffix.lower() in ['.usd','.usda','.usdc']:visit(dep)
    visit(resources.entry)


def usd_matrix(gf_matrix):
    return np.asarray(gf_matrix,dtype=float).T.copy()


def usd_quaternion(value):
    return [value.GetReal(),*value.GetImaginary()]


def usd_attribute(prim,name,default):
    attr=prim.GetAttribute(name)
    value=attr.Get() if attr else None
    return default if value is None else value


def shape_mesh(shape):
    if shape.kind=='mesh':return shape.mesh.copy()
    if shape.kind=='box':return trimesh.creation.box(extents=shape.size)
    if shape.kind=='sphere':return trimesh.creation.icosphere(subdivisions=3,radius=shape.size[0])
    if shape.kind=='cylinder':return trimesh.creation.cylinder(radius=shape.size[0],height=shape.size[1],sections=48)
    raise ValueError('Unsupported geometry '+shape.kind)


def read_usd(resources):
    from pxr import Usd,UsdGeom,UsdPhysics,UsdShade
    check_usd_dependencies(resources)
    stage=Usd.Stage.Open(str(resources.entry))
    if not stage:raise ValueError('OpenUSD could not open the stage')
    unit=float(UsdGeom.GetStageMetersPerUnit(stage));mass_unit=float(UsdPhysics.GetStageKilogramsPerUnit(stage))
    if not np.isfinite([unit,mass_unit]).all() or min(unit,mass_unit)<=0:raise ValueError('Invalid USD stage units')
    up=str(UsdGeom.GetStageUpAxis(stage))
    coordinate=np.eye(4)
    if up=='Y':coordinate[:3,:3]=Rotation.from_euler('x',math.pi/2).as_matrix()
    cache=UsdGeom.XformCache(Usd.TimeCode.Default())
    prims=list(stage.Traverse(Usd.TraverseInstanceProxies()))
    bodies=[p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    default=stage.GetDefaultPrim()
    model=Model(str(usd_attribute(default,'urdf:modelName',default.GetName())) if default else resources.entry.stem)
    root_name='__world'
    occupied=set()
    def name_for(p):
        requested=str(usd_attribute(p,'urdf:name',p.GetName()))
        name=requested
        suffix=1
        while name in occupied:
            name=requested+'_'+str(suffix);suffix+=1
        occupied.add(name)
        return name
    names={str(p.GetPath()):name_for(p) for p in bodies}
    while root_name in occupied:root_name='_'+root_name
    model.links[root_name]=Link(root_name)
    frames={root_name:np.eye(4)}
    path_frames={}
    for p in bodies:
        path=str(p.GetPath());name=names[path];raw=usd_matrix(cache.GetLocalToWorldTransform(p))
        raw[:3,3]*=unit;raw=coordinate@raw
        r=raw[:3,:3]
        if not np.allclose(r.T@r,np.eye(3),atol=1e-5) or np.linalg.det(r)<0:
            raise ValueError('Scaled / reflected rigid-body frames must be baked before conversion: '+path)
        frames[name]=raw;path_frames[path]=raw
        link=model.links[name]=Link(name)
        mass=UsdPhysics.MassAPI(p)
        if p.HasAPI(UsdPhysics.MassAPI):
            link.mass=float(mass.GetMassAttr().Get() or 0)*mass_unit
            center=mass.GetCenterOfMassAttr().Get()
            if center is not None and np.isfinite(center).all():link.com=vec(center)*unit
            diag=mass.GetDiagonalInertiaAttr().Get()
            if diag is not None and all(float(v)>0 for v in diag):
                q=mass.GetPrincipalAxesAttr().Get()
                r=transform(quaternion=usd_quaternion(q))[:3,:3] if q is not None else np.eye(3)
                link.inertia=r@np.diag(vec(diag)*mass_unit*unit**2)@r.T
            elif link.mass>0:
                model.note('Some USD bodies have mass but no explicit inertia. Mass is retained with an unspecified zero tensor; MJCF export requires authored positive inertia.')
        if link.mass<=0 and not UsdPhysics.RigidBodyAPI(p).GetKinematicEnabledAttr().Get():model.note('Some USD rigid bodies have no explicit mass/inertia; implicit density is not guessed. Static mass results may be incomplete.')
    def color(p):
        rgba=np.array([.72,.78,.84,1.])
        g=UsdGeom.Gprim(p)
        colors=g.GetDisplayColorAttr().Get()
        opacity=g.GetDisplayOpacityAttr().Get()
        if colors:rgba[:3]=vec(colors[0])
        if opacity:rgba[3]=float(opacity[0])
        try:
            material,_=UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()
            if material:
                surface=material.ComputeSurfaceSource()[0]
                if surface:
                    diffuse=surface.GetInput('diffuseColor');alpha=surface.GetInput('opacity')
                    if diffuse and diffuse.Get() is not None:rgba[:3]=vec(diffuse.Get())
                    if alpha and alpha.Get() is not None:rgba[3]=float(alpha.Get())
                    if diffuse and diffuse.HasConnectedSource():model.note('USD texture / material networks are reduced to constant display colors.')
        except Exception:model.note('Some USD material bindings could not be reduced to a display color.')
        return rgba
    def geometry(p):
        if p.IsA(UsdGeom.Mesh):
            g=UsdGeom.Mesh(p);vertices=g.GetPointsAttr().Get();counts=g.GetFaceVertexCountsAttr().Get();indices=g.GetFaceVertexIndicesAttr().Get()
            if vertices is None or counts is None or indices is None:return None
            faces=[];offset=0
            for count in counts:
                count=int(count);face=list(indices[offset:offset+count]);offset+=count
                if count<3:continue
                if count>4:raise ValueError('USD mesh polygons with more than four vertices must be triangulated first: '+str(p.GetPath()))
                for i in range(1,count-1):faces.append([face[0],face[i],face[i+1]])
            if not faces:return None
            if g.GetOrientationAttr().Get()=='leftHanded':faces=[f[::-1] for f in faces]
            if g.GetSubdivisionSchemeAttr().Get() not in [None,'none']:model.note('USD subdivision surfaces are imported as their polygon control cage.')
            if g.GetHoleIndicesAttr().Get():raise ValueError('USD meshes with holes must be baked / triangulated before conversion')
            return trimesh.Trimesh(vertices=np.asarray(vertices,dtype=float)*unit,faces=faces,process=False)
        if p.IsA(UsdGeom.Cube):return trimesh.creation.box(extents=[float(UsdGeom.Cube(p).GetSizeAttr().Get())*unit]*3)
        if p.IsA(UsdGeom.Sphere):return trimesh.creation.icosphere(subdivisions=3,radius=float(UsdGeom.Sphere(p).GetRadiusAttr().Get())*unit)
        if p.IsA(UsdGeom.Cylinder) or p.IsA(UsdGeom.Capsule):
            g=UsdGeom.Cylinder(p) if p.IsA(UsdGeom.Cylinder) else UsdGeom.Capsule(p)
            radius=float(g.GetRadiusAttr().Get())*unit;height=float(g.GetHeightAttr().Get())*unit
            mesh=trimesh.creation.cylinder(radius=radius,height=height,sections=48) if p.IsA(UsdGeom.Cylinder) else trimesh.creation.capsule(radius=radius,height=height,count=[16,16])
            axis={'X':[1,0,0],'Y':[0,1,0],'Z':[0,0,1]}[str(g.GetAxisAttr().Get())]
            mesh.apply_transform(trimesh.geometry.align_vectors([0,0,1],axis))
            return mesh
        return None
    for p in prims:
        if not p.IsA(UsdGeom.Gprim):continue
        if UsdGeom.Imageable(p).ComputeVisibility()=='invisible' and not p.HasAPI(UsdPhysics.CollisionAPI):continue
        mesh=geometry(p)
        if mesh is None:
            if p.IsA(UsdGeom.Points) or p.IsA(UsdGeom.BasisCurves):model.note('USD points / curves are excluded; only rigid solid geometry is converted.')
            continue
        ancestor=p
        while ancestor and str(ancestor.GetPath()) not in names:ancestor=ancestor.GetParent()
        owner=names[str(ancestor.GetPath())] if ancestor else root_name
        world=usd_matrix(cache.GetLocalToWorldTransform(p));world[:3,3]*=unit;world=coordinate@world
        mesh.apply_transform(np.linalg.inv(frames[owner])@world)
        shape=Shape('mesh',mesh=mesh,color=color(p))
        role=str(usd_attribute(p,'urdf:role',''))
        if role!='collision' and UsdGeom.Imageable(p).ComputeVisibility()!='invisible':model.links[owner].visuals.append(shape)
        if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()!=False:model.links[owner].collisions.append(shape)
    authored_frames={name:t.copy() for name,t in frames.items()}
    new_frames={name:t.copy() for name,t in frames.items()}
    children=set()
    joint_names=set()
    for p in prims:
        if not p.IsA(UsdPhysics.Joint):continue
        api=UsdPhysics.Joint(p)
        if api.GetJointEnabledAttr().Get()==False:continue
        if not any(p.IsA(k) for k in [UsdPhysics.FixedJoint,UsdPhysics.RevoluteJoint,UsdPhysics.PrismaticJoint]):
            raise ValueError('Unsupported USD joint '+p.GetTypeName()+': '+str(p.GetPath())+'; only fixed, revolute and prismatic joints are portable')
        targets0=api.GetBody0Rel().GetTargets();targets1=api.GetBody1Rel().GetTargets()
        path0=str(targets0[0]) if targets0 else None;path1=str(targets1[0]) if targets1 else None
        # Non-rigid Xforms are static frames in USD. Include their transform.
        def body(path):
            if path in names:return names[path],frames[names[path]]
            if path:
                prim=stage.GetPrimAtPath(path)
                if not prim:raise ValueError('Unresolved USD joint body: '+path)
                frame=usd_matrix(cache.GetLocalToWorldTransform(prim));frame[:3,3]*=unit
                return root_name,coordinate@frame
            return root_name,np.eye(4)
        parent,pw=body(path0);child,cw=body(path1)
        if child==root_name:
            raise ValueError('USD joints with a world-side body1 must be reoriented to body0=parent / body1=child before conversion')
        if child in children:raise ValueError('USD closed-loop / multiple-parent constraints cannot be converted to URDF')
        children.add(child)
        f0=transform(vec(api.GetLocalPos0Attr().Get())*unit,usd_quaternion(api.GetLocalRot0Attr().Get()))
        f1=transform(vec(api.GetLocalPos1Attr().Get())*unit,usd_quaternion(api.GetLocalRot1Attr().Get()))
        joint_world=cw@f1
        kind='fixed' if p.IsA(UsdPhysics.FixedJoint) else 'prismatic' if p.IsA(UsdPhysics.PrismaticJoint) else 'revolute'
        axis_name=str(usd_attribute(p,'physics:axis','X'));axis={'X':vec(None,(1,0,0)),'Y':vec(None,(0,1,0)),'Z':vec(None,(0,0,1))}[axis_name]
        relative=np.linalg.inv(pw@f0)@joint_world
        if kind=='prismatic':
            reference=float(np.dot(relative[:3,3],axis))
            if not np.allclose(relative[:3,:3],np.eye(3),atol=1e-4) or np.linalg.norm(relative[:3,3]-reference*axis)>1e-5:raise ValueError('USD prismatic frames are inconsistent: '+str(p.GetPath()))
        elif kind=='revolute':
            q=Rotation.from_matrix(relative[:3,:3]).as_rotvec();reference=float(np.dot(q,axis))
            if np.linalg.norm(q-reference*axis)>1e-4 or np.linalg.norm(relative[:3,3])>1e-5:raise ValueError('USD revolute frames are inconsistent: '+str(p.GetPath()))
        else:reference=0.
        raw_name=str(usd_attribute(p,'urdf:name',p.GetName()))
        if raw_name in joint_names:raise ValueError('Duplicate USD joint name: '+raw_name)
        joint_names.add(raw_name)
        # URDF links are joint frames. An explicit fixed frame preserves the
        # original rigid-body frame, geometry, COM and all child joints.
        new_frames[child]=joint_world
        j=Joint(raw_name,kind,parent,child,np.eye(4),axis)
        if kind!='fixed':
            scale=unit if kind=='prismatic' else math.pi/180
            lo=float(usd_attribute(p,'physics:lowerLimit',-math.inf))*scale-reference
            hi=float(usd_attribute(p,'physics:upperLimit',math.inf))*scale-reference
            if kind=='revolute' and not np.isfinite([lo,hi]).any():j.kind='continuous'
            elif not np.isfinite([lo,hi]).all():raise ValueError('One-sided or unbounded prismatic limits require finite portable limits: '+raw_name)
            else:j.lower,j.upper=lo,hi
            drive=UsdPhysics.DriveAPI(p,'linear' if kind=='prismatic' else 'angular')
            force=drive.GetMaxForceAttr().Get() if drive else None
            if drive and drive.GetTypeAttr().Get()=='acceleration':model.note('Acceleration-drive force limits are not torque limits; actuator effort remains unspecified.')
            elif force is not None and np.isfinite(force):j.effort=float(force)*mass_unit*(unit if kind=='prismatic' else unit**2)
            j.velocity=float(usd_attribute(p,'urdf:velocity',0.))
            j.damping=float(usd_attribute(p,'urdf:damping',0.));j.friction=float(usd_attribute(p,'urdf:friction',0.))
            mimic=usd_attribute(p,'urdf:mimic','')
            if mimic:j.mimic=json.loads(mimic)
        model.joints.append(j)
    for name,link in model.links.items():
        offset=np.linalg.inv(new_frames[name])@authored_frames[name]
        # Shared visual/collision shapes must only be reframed once.
        for shape in {id(s):s for s in link.visuals+link.collisions}.values():shape.pose=offset@shape.pose
        link.com=(offset@np.r_[link.com,1])[:3]
        link.inertia=offset[:3,:3]@link.inertia@offset[:3,:3].T
    for j in model.joints:j.pose=np.linalg.inv(new_frames[j.parent])@new_frames[j.child]
    for path,name in names.items():
        if name not in children:
            # Unconstrained rigid roots remain floating; fixed world roots are
            # represented by their authored fixed joint above.
            prim=stage.GetPrimAtPath(path);joint_name=str(usd_attribute(prim,'urdf:floatingJointName',name+'__root'))
            model.joints.append(Joint(joint_name,'floating',root_name,name,frames[name]))
    if not bodies:model.note('USD has no rigid-body articulation; imported as a static geometry model.')
    if any(p.GetAttribute('xformOp:transform').ValueMightBeTimeVarying() for p in prims if p.GetAttribute('xformOp:transform')):model.note('USD animation is sampled at the default time; animation curves are not transferred.')
    model.note('USD geometry is flattened to triangle meshes; renderer-specific shaders, scene lights and simulator extensions are not transferred.')
    anchors=[j for j in model.joints if j.parent==root_name]
    expected=str(usd_attribute(default,'urdf:rootLink','')) if default else ''
    anchor=next((j for j in anchors if j.child==expected and j.kind=='fixed' and np.allclose(j.pose,np.eye(4),atol=1e-6)),None)
    if expected and anchor and not model.links[root_name].visuals and not model.links[root_name].collisions:
        model.joints.remove(anchor)
        for j in model.joints:
            if j.parent==root_name:j.parent=expected
        del model.links[root_name]
    model.validate()
    return model


def write_usd(model,directory,text=False):
    from pxr import Usd,UsdGeom,UsdPhysics,Gf,Sdf,Vt
    model.validate();directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    path=directory/('model.usda' if text else 'model.usd')
    stage=Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    UsdPhysics.SetStageKilogramsPerUnit(stage,1.)
    root=UsdGeom.Xform.Define(stage,'/Robot');stage.SetDefaultPrim(root.GetPrim());root.GetPrim().SetMetadata('documentation','Portable robot geometry / articulation; see conversion-report.json for differences.')
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    poses=model.world_poses();paths={};body_prims={}
    def qf(t):
        q=quat(t);return Gf.Quatf(float(q[0]),Gf.Vec3f(*map(float,q[1:])))
    def custom(prim,name,value,kind=Sdf.ValueTypeNames.String):prim.CreateAttribute('urdf:'+name,kind,custom=True).Set(value)
    custom(root.GetPrim(),'modelName',model.name);custom(root.GetPrim(),'rootLink',model.validate())
    for i,link in enumerate(model.links.values()):
        path_body='/Robot/Links/L'+str(i)+'_'+safe_name(link.name);paths[link.name]=path_body
        body=UsdGeom.Xform.Define(stage,path_body);p=body.GetPrim();body_prims[link.name]=p
        custom(p,'name',link.name)
        floating=next((j for j in model.joints if j.child==link.name and j.kind=='floating'),None)
        if floating:custom(p,'floatingJointName',floating.name)
        body.AddTranslateOp().Set(Gf.Vec3d(*map(float,poses[link.name][:3,3])))
        body.AddOrientOp().Set(Gf.Quatf(qf(poses[link.name])))
        # Static zero-mass frame links must not acquire implicit collision mass.
        if link.mass>0:
            UsdPhysics.RigidBodyAPI.Apply(p)
            api=UsdPhysics.MassAPI.Apply(p);api.CreateMassAttr(float(link.mass));api.CreateCenterOfMassAttr(Gf.Vec3f(*map(float,link.com)))
            values,axes=np.linalg.eigh(link.inertia)
            if min(values)<=0:raise ValueError('Nonpositive inertia: '+link.name)
            if np.linalg.det(axes)<0:axes[:,0]*=-1
            t=np.eye(4);t[:3,:3]=axes
            api.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float,values)));api.CreatePrincipalAxesAttr(qf(t))
        else:
            # Keep an explicit zero-mass kinematic body so the USD reader can
            # retain virtual frames instead of treating them as unrelated Xforms.
            rb=UsdPhysics.RigidBodyAPI.Apply(p);rb.CreateKinematicEnabledAttr(True)
        for role,shapes in [('visual',link.visuals),('collision',link.collisions)]:
            for k,shape in enumerate(shapes):
                mesh=shape_mesh(shape);mesh.apply_transform(shape.pose)
                geom=UsdGeom.Mesh.Define(stage,path_body+'/'+role+'_'+str(k));gp=geom.GetPrim()
                geom.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertices,dtype=np.float32)))
                geom.CreateFaceVertexCountsAttr([3]*len(mesh.faces));geom.CreateFaceVertexIndicesAttr(np.asarray(mesh.faces).ravel().tolist());geom.CreateSubdivisionSchemeAttr('none')
                geom.CreateDisplayColorAttr([Gf.Vec3f(*map(float,shape.color[:3]))]);geom.CreateDisplayOpacityAttr([float(shape.color[3])]);custom(gp,'role',role)
                if role=='collision':
                    UsdPhysics.CollisionAPI.Apply(gp);geom.CreateVisibilityAttr('invisible')
    def frames(api,t0,t1):
        api.CreateLocalPos0Attr(Gf.Vec3f(*map(float,t0[:3,3])));api.CreateLocalRot0Attr(qf(t0))
        api.CreateLocalPos1Attr(Gf.Vec3f(*map(float,t1[:3,3])));api.CreateLocalRot1Attr(qf(t1))
    for i,j in enumerate(model.joints):
        if j.kind=='floating':continue
        path_joint='/Robot/Joints/J'+str(i)+'_'+safe_name(j.name)
        schema=UsdPhysics.FixedJoint if j.kind=='fixed' else UsdPhysics.PrismaticJoint if j.kind=='prismatic' else UsdPhysics.RevoluteJoint
        api=schema.Define(stage,path_joint);p=api.GetPrim();custom(p,'name',j.name)
        api.CreateBody0Rel().SetTargets([paths[j.parent]]);api.CreateBody1Rel().SetTargets([paths[j.child]])
        align=trimesh.geometry.align_vectors([1,0,0],j.axis)
        frames(api,j.pose@align,align)
        if j.kind!='fixed':
            api.CreateAxisAttr('X')
            if j.kind!='continuous':
                scale=1 if j.kind=='prismatic' else 180/math.pi
                api.CreateLowerLimitAttr(float(j.lower*scale));api.CreateUpperLimitAttr(float(j.upper*scale))
            custom(p,'velocity',float(j.velocity),Sdf.ValueTypeNames.Double);
            custom(p,'damping',float(j.damping),Sdf.ValueTypeNames.Double);custom(p,'friction',float(j.friction),Sdf.ValueTypeNames.Double)
            if j.mimic:custom(p,'mimic',json.dumps(j.mimic))
            if j.effort>0 or j.damping>0:
                drive=UsdPhysics.DriveAPI.Apply(p,'linear' if j.kind=='prismatic' else 'angular')
                if j.effort>0:drive.CreateMaxForceAttr(float(j.effort))
                drive.CreateStiffnessAttr(0.);drive.CreateDampingAttr(float(j.damping*(1 if j.kind=='prismatic' else math.pi/180)))
    # Pin the canonical root to the world; floating children remain unconstrained.
    root_name=model.validate();api=UsdPhysics.FixedJoint.Define(stage,'/Robot/Joints/WorldAnchor');api.CreateBody1Rel().SetTargets([paths[root_name]]);frames(api,np.eye(4),np.eye(4));custom(api.GetPrim(),'name','__world_anchor')
    stage.GetRootLayer().Save()
    if any(j.mimic for j in model.joints):model.note('USD mimic / velocity / passive-friction values are preserved as urdf:* metadata; simulator-specific mimic APIs are not authored.')
    model.note('USD export uses the standard UsdPhysics schema; simulator-specific articulation tuning and controller gains are not reconstructed.')
    return path


def read_model(root,entry):
    resources=Resources(root,entry);suffix=resources.entry.suffix.lower()
    if suffix in ['.usd','.usda','.usdc']:return read_usd(resources),'usd'
    document=ET.parse(resources.entry).getroot()
    if document.tag=='robot':return read_urdf(resources),'urdf'
    if document.tag=='mujoco':return read_mjcf(resources),'mjcf'
    raise ValueError('Expected URDF <robot>, MJCF <mujoco>, or a USD stage')


def write_model(model,target,directory):
    if target=='urdf':return write_urdf(model,directory)
    if target=='mjcf':return write_mjcf(model,directory)
    if target in ['usd','usda']:return write_usd(model,directory,text=target=='usda')
    raise ValueError('Unknown target format')
