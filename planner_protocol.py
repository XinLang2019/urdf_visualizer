"""Validated, dependency-free protocol shared by the HTTP service and solver worker."""
from pathlib import Path
import copy
import math
import xml.etree.ElementTree as ET

DEFAULTS = {
    'motion': 'backflip', 'dt': .02,
    'phases': {'stand': .4, 'crouch': .5, 'load': .1, 'flight': .5, 'landing': .6, 'settle': .4},
    'friction': .9, 'gravity': 9.81, 'jointSpeed': 18., 'angularSpeed': 24., 'torqueScale': 1.,
    'swingClearance': .14, 'bodyClearance': .14, 'legClearance': .09,
    'meshCollision': True, 'symmetry': True, 'selfMargin': .002, 'groundMargin': .001,
    'initialHip': 0., 'initialThigh': .8, 'initialCalf': -1.5, 'crouchThigh': 1.3,
    'targetX': -.4, 'targetY': 0., 'targetZ': 0., 'targetRoll': 0., 'targetPitch': 0., 'targetYaw': 0.,
    'lockLandingJoints': True, 'waypointEnabled': False, 'waypointRoll': 0., 'waypointPitch': -180., 'waypointYaw': 0.,
    'waypointTolerance': 15., 'waypointHeight': .4,
    'maxIterations': 300, 'rounds': 3, 'tolerance': .007, 'wallTime': 600.,
}
LIMITS = {
    'dt': (.01, .2), 'friction': (.05, 3), 'gravity': (.1, 30), 'jointSpeed': (.1, 40),
    'angularSpeed': (1, 50), 'torqueScale': (.05, 1),
    'swingClearance': (0, .5), 'bodyClearance': (.03, .5), 'legClearance': (0, .3),
    'selfMargin': (0, .02), 'groundMargin': (0, .02), 'initialHip': (-.7, .7),
    'initialThigh': (-.5, 2), 'initialCalf': (-2.7, -.84), 'crouchThigh': (.8, 2),
    'targetX': (-2, 2), 'targetY': (-1, 1), 'targetZ': (-.1, .3),
    'targetRoll': (-180, 180), 'targetPitch': (-180, 180), 'targetYaw': (-180, 180),
    'waypointRoll': (-360, 360), 'waypointPitch': (-360, 360), 'waypointYaw': (-360, 360),
    'waypointTolerance': (1, 45), 'waypointHeight': (.15, 1.5),
    'maxIterations': (1, 1000), 'rounds': (1, 5), 'tolerance': (1e-6, .01), 'wallTime': (5, 1800),
}

def validate_config(value):
    if not isinstance(value, dict) or set(value)-set(DEFAULTS):
        raise ValueError('Unknown planner settings')
    cfg=copy.deepcopy(DEFAULTS)
    for k,v in value.items():
        if k=='phases':
            if not isinstance(v,dict) or set(v)-set(DEFAULTS['phases']):raise ValueError('Invalid contact phases')
            cfg[k].update(v)
        else:cfg[k]=v
    if cfg['motion'] not in ['backflip','frontflip','jump','stand']:raise ValueError('Unsupported motion')
    for k,(lo,hi) in LIMITS.items():
        v=cfg[k]
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not lo<=v<=hi:raise ValueError(f'{k} must be in [{lo}, {hi}]')
    for k in LIMITS:
        if k not in ['maxIterations','rounds']:cfg[k]=float(cfg[k])
    for k in ['meshCollision','symmetry','waypointEnabled','lockLandingJoints']:
        if not isinstance(cfg[k],bool):raise ValueError(k+' must be boolean')
    for k in ['maxIterations','rounds']:
        if int(cfg[k])!=cfg[k]:raise ValueError(k+' must be an integer')
        cfg[k]=int(cfg[k])
    for k,v in cfg['phases'].items():
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=2:raise ValueError('Invalid phase duration: '+k)
        cfg['phases'][k]=round(v/cfg['dt'])*cfg['dt']
    counts={k:round(v/cfg['dt']) for k,v in cfg['phases'].items()}
    if not 4<=sum(counts.values())<=160:raise ValueError('Use 4–160 solver nodes; increase dt or shorten phases')
    if counts['stand']<1 or counts['settle']<1:raise ValueError('Initial and final stand phases require at least one node')
    if cfg['motion']!='stand' and counts['flight']<3:raise ValueError('Flight requires at least three nodes')
    if cfg['motion']=='stand' and counts['flight']:raise ValueError('Standing motion must have zero flight duration')
    if cfg['waypointEnabled'] and counts['flight']<3:raise ValueError('Attitude waypoint requires a flight phase')
    return cfg

def model_xml(root):
    """Viewer variant of the actual planner model; keep rotor inertias and limits."""
    path=Path(root)/'src/robots/go2/go2/urdf/go2.urdf'
    robot=ET.parse(path).getroot()
    for joint in list(robot.findall('joint')):
        if joint.get('type')=='floating':
            parent=joint.find('parent').get('link');robot.remove(joint)
            link=robot.find(f"link[@name='{parent}']")
            if link is not None:robot.remove(link)
    for mesh in robot.findall('.//mesh'):
        filename=mesh.get('filename','').replace('package://go2/','').lstrip('./')
        mesh.set('filename','/api/planner/assets/'+filename)
    return '<?xml version="1.0"?>\n'+ET.tostring(robot,encoding='unicode')

def description_hash(text):
    h=2166136261
    for char in text:h=((h^ord(char))*16777619)&0xffffffff
    return format(h,'x')
