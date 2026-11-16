#!/usr/bin/env python3
"""Static workbench plus isolated MJCF / USD import and model conversion."""
import argparse
import base64
from email import policy
from email.parser import BytesParser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import threading
from urllib.parse import quote, unquote, urlsplit
import zipfile

ROOT=Path(__file__).resolve().parent
MAX_BYTES=384*1024*1024
LIMIT=threading.BoundedSemaphore(2)


def portable_path(value):
    if not isinstance(value,str) or not value or '\\' in value or '\x00' in value:
        raise ValueError('Invalid model resource path')
    path=PurePosixPath(value)
    if path.is_absolute() or any(p in ['..','.'] or p.startswith('.') or ':' in p for p in path.parts):
        raise ValueError('Model resource paths must stay inside the uploaded folder')
    return path.as_posix()


def unpack_usdz(entry,root):
    if entry.suffix.lower()!='.usdz':return entry
    target=root/'__usdz_unpacked__';target.mkdir()
    with zipfile.ZipFile(entry) as archive:
        members=archive.infolist()
        if sum(m.file_size for m in members)>MAX_BYTES or len(members)>1500:raise ValueError('USDZ is too large')
        first=None
        for member in members:
            if member.is_dir():continue
            name=portable_path(member.filename)
            if member.external_attr>>16 & 0o170000==0o120000:raise ValueError('USDZ symbolic links are not supported')
            path=target/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(archive.read(member))
            if first is None and path.suffix.lower() in ['.usd','.usda','.usdc']:first=path
        if first is None:raise ValueError('USDZ contains no USD stage')
    return first


def run_worker(config_path):
    import model_formats as formats
    cfg=json.loads(Path(config_path).read_text())
    root=Path(cfg['input']);entry=unpack_usdz(root/cfg['entry'],root)
    model,source=formats.read_model(root,entry)
    out=Path(cfg['output'])
    target=cfg['target']
    source_info=cfg.get('sourceInfo') or {}
    for note in source_info.get('warnings',[]):model.note(note)
    path=formats.write_model(model,target,out)
    report={'sourceFormat':source_info.get('format',source),'workingFormat':source,'targetFormat':target,'entry':path.name,'model':model.name,
            'links':len(model.links),'joints':len(model.joints),'units':{'length':'m','mass':'kg','angle':'rad'},
            'upAxis':'Z','scope':'Rigid geometry, tree joints, mass, COM and inertia. Not a simulator-equivalence guarantee.',
            'warnings':model.notes}
    (out/'conversion-report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    (out/'README.txt').write_text('Open '+path.name+' with its meshes/ folder beside it.\nSee conversion-report.json for format differences.\n')
    (Path(cfg['result'])).write_text(json.dumps({'report':report,'entry':path.name}))


def dependencies(python):
    try:
        result=subprocess.run([python,'-c','import json,mujoco,pxr,trimesh,collada,numpy,scipy; from pxr import Usd; print(json.dumps({"mujoco":mujoco.__version__,"usd":".".join(map(str,Usd.GetVersion()))}))'],capture_output=True,text=True,timeout=30)
        if result.returncode==0:return {'available':True,'versions':json.loads(result.stdout.strip()),'problem':''}
        problem=result.stderr.strip().splitlines()[-1] if result.stderr.strip() else 'Missing model conversion dependencies'
    except (OSError,ValueError,subprocess.TimeoutExpired) as error:problem=str(error)
    return {'available':False,'versions':{},'problem':problem+'; install requirements-formats.txt and restart model_server.py'}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,python,status,**kwargs):
        self.python=python;self.status=status
        super().__init__(*args,directory=str(ROOT),**kwargs)

    def json_response(self,data,status=200):
        body=json.dumps(data,ensure_ascii=False,allow_nan=False).encode()
        self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)

    def do_GET(self):
        path=unquote(urlsplit(self.path).path)
        if path=='/api/formats/status':return self.json_response({'application':'urdf-viewer-formats',**self.status})
        if path.startswith('/api/'):return self.json_response({'error':'Unknown model format API'},404)
        if any(p.startswith('.') for p in PurePosixPath(path).parts if p!='/'):return self.send_error(404)
        return super().do_GET()

    def do_POST(self):
        path=urlsplit(self.path).path
        if path not in ['/api/formats/import','/api/formats/convert']:return self.json_response({'error':'Unknown model format API'},404)
        if self.headers.get('Origin') not in [None,'http://'+self.headers.get('Host','')]:return self.json_response({'error':'Cross-origin requests are not accepted'},403)
        if not self.status['available']:return self.json_response({'error':self.status['problem']},503)
        if self.headers.get_content_type()!='multipart/form-data':return self.json_response({'error':'Use multipart/form-data'},415)
        if not LIMIT.acquire(blocking=False):return self.json_response({'error':'Two conversions are already running; try again shortly'},429)
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=MAX_BYTES:raise ValueError('Model upload exceeds 384 MiB')
            raw=self.rfile.read(size)
            if len(raw)!=size:raise ValueError('Incomplete model upload')
            message=BytesParser(policy=policy.default).parsebytes(('Content-Type: '+self.headers['Content-Type']+'\r\nMIME-Version: 1.0\r\n\r\n').encode()+raw)
            fields={};files=[]
            if not message.is_multipart():raise ValueError('Invalid multipart model upload')
            for part in message.iter_parts():
                name=part.get_param('name',header='content-disposition')
                value=part.get_payload(decode=True) or b''
                if name=='files':files.append(value)
                elif name in ['entry','paths','target','sourceInfo']:
                    if name in fields:raise ValueError('Duplicate request field')
                    fields[name]=value.decode('utf-8')
            paths=json.loads(fields['paths']);entry=portable_path(fields['entry'])
            if not isinstance(paths,list) or not 0<len(paths)<=1500 or len(paths)!=len(files):raise ValueError('Invalid model resource list')
            paths=[portable_path(p) for p in paths]
            if len(set(paths))!=len(paths) or entry not in paths:raise ValueError('Duplicate paths or missing model entry')
            source_info=json.loads(fields.get('sourceInfo','null'))
            if source_info is not None:
                if not isinstance(source_info,dict) or source_info.get('format') not in ['mjcf','usd'] or not isinstance(source_info.get('warnings',[]),list) or len(source_info.get('warnings',[]))>50 or any(not isinstance(w,str) or len(w)>2000 for w in source_info.get('warnings',[])):raise ValueError('Invalid source format report')
            target='urdf' if path.endswith('/import') else fields.get('target')
            if target not in ['urdf','mjcf','usd','usda']:raise ValueError('Unknown target format')
            with tempfile.TemporaryDirectory(prefix='urdf-format-job-') as temp:
                folder=Path(temp);inputs=folder/'input';output=folder/'output';inputs.mkdir();output.mkdir()
                for filename,value in zip(paths,files):
                    dest=inputs/filename;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(value)
                config={'input':str(inputs),'entry':entry,'output':str(output),'target':target,'result':str(folder/'result.json'),'sourceInfo':source_info}
                settings=folder/'settings.json';settings.write_text(json.dumps(config))
                env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
                result=subprocess.run([self.python,str(ROOT/'model_server.py'),'--worker',str(settings)],cwd=folder,capture_output=True,text=True,timeout=180,env=env)
                if result.returncode!=0:
                    try:problem=json.loads(result.stdout)['error']
                    except (ValueError,KeyError):problem=result.stderr.strip()[-1500:] or 'Native reader failed while processing this model'
                    return self.json_response({'error':problem},422)
                info=json.loads((folder/'result.json').read_text())
                if path.endswith('/import'):
                    resources=[]
                    for asset in sorted((output/'meshes').rglob('*')) if (output/'meshes').exists() else []:
                        if asset.is_file():resources.append({'path':asset.relative_to(output).as_posix(),'data':base64.b64encode(asset.read_bytes()).decode()})
                    return self.json_response({'urdf':(output/info['entry']).read_text(),'assets':resources,'report':info['report']})
                memory=io.BytesIO()
                with zipfile.ZipFile(memory,'w',zipfile.ZIP_DEFLATED) as archive:
                    for asset in sorted(output.rglob('*')):
                        if asset.is_file():archive.write(asset,asset.relative_to(output).as_posix())
                body=memory.getvalue();name='robot-'+target+'.zip'
                self.send_response(200);self.send_header('Content-Type','application/zip');self.send_header('Content-Disposition','attachment; filename="'+name+'"')
                self.send_header('X-Conversion-Report',quote(json.dumps(info['report'],ensure_ascii=False)))
                self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
        except (ValueError,KeyError,TypeError,OSError,zipfile.BadZipFile) as error:self.json_response({'error':str(error)},400)
        except subprocess.TimeoutExpired:self.json_response({'error':'Model conversion exceeded 180 seconds; simplify or split the model'},504)
        finally:LIMIT.release()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=8080);parser.add_argument('--bind',default='127.0.0.1');parser.add_argument('--python',dest='python',default=str(ROOT/'.venv-formats/bin/python') if (ROOT/'.venv-formats/bin/python').is_file() else sys.executable);parser.add_argument('--worker',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:
        try:run_worker(args.worker)
        except Exception as error:print(json.dumps({'error':str(error)},ensure_ascii=False));sys.exit(1)
        return
    # Keep the venv executable path: resolving its symlink would bypass pyvenv.cfg.
    args.python=os.path.abspath(os.path.expanduser(args.python))
    status=dependencies(args.python)
    server=ThreadingHTTPServer((args.bind,args.port),partial(Handler,python=args.python,status=status))
    print('URDF-Viewer: http://'+args.bind+':'+str(server.server_port)+' | MJCF / USD: '+('ready' if status['available'] else status['problem']),flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
