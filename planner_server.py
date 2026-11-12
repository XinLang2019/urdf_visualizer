#!/usr/bin/env python3
"""Serve URDF-Viewer and isolated go2_flip_TO planning jobs on localhost."""
import argparse
import copy
from functools import partial
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import unquote,urlsplit
import uuid
from planner_protocol import DEFAULTS,validate_config,model_xml,description_hash

ROOT=Path(__file__).resolve().parent

class PlannerJobs:
    def __init__(self,solver_root,python):
        self.root=Path(solver_root).resolve();self.python=str(Path(python).resolve());self.jobs={};self.lock=threading.RLock();self.available=False;self.problem='Checking solver dependencies';self.model=None
        try:
            text=model_xml(self.root);self.model={'name':'Go2','descriptionHash':description_hash(text),'urdfUrl':'/api/planner/model','jointNames':[s+'_'+p+'_joint' for s in ['FL','FR','RL','RR'] for p in ['hip','thigh','calf']]}
            result=subprocess.run([self.python,'-c','import pinocchio,cyipopt; import importlib.util; assert importlib.util.find_spec("coal") or importlib.util.find_spec("hppfcl")'],capture_output=True,text=True,timeout=25)
            self.available=result.returncode==0;self.problem='' if self.available else 'Pinocchio / cyipopt / coal dependencies unavailable: '+result.stderr[-1500:]
        except Exception as e:self.problem=str(e)
    def capabilities(self):return {'available':self.available,'problem':self.problem,'model':self.model,'defaults':DEFAULTS,'motions':['backflip','frontflip','jump','stand'],'progress':'real IPOPT iterations; iteration budget is not convergence percentage'}
    def start(self,config,model_hash):
        cfg=validate_config(config)
        if not self.available:raise ValueError(self.problem)
        if model_hash!=self.model['descriptionHash']:raise ValueError('Load the exact planner Go2 model before planning')
        with self.lock:
            if any(j['state'] in ['queued','running','cancelling'] for j in self.jobs.values()):raise RuntimeError('Another plan is running; cancel it before starting a new plan')
            ident=uuid.uuid4().hex;directory=Path(tempfile.mkdtemp(prefix='urdf-whole-body-'))
            job={'id':ident,'state':'queued','stage':'queued','message':'Starting isolated solver','started':time.monotonic(),'config':cfg,'directory':directory,'logs':[],'attempt':1,'process':None,'cancel':False}
            self.jobs[ident]=job
            # Bound retained jobs/results; running jobs are never removed.
            for old in list(self.jobs)[:-12]:
                if self.jobs[old]['state'] not in ['queued','running','cancelling']:self.jobs.pop(old)
            threading.Thread(target=self._run,args=(job,),daemon=True).start();return self.status(ident)
    def status(self,ident):
        with self.lock:
            j=self.jobs[ident];fields=['id','state','stage','message','iteration','round','objective','primal','dual','attempt','logs']
            out={k:copy.deepcopy(j[k]) for k in fields if k in j};out['elapsed']=j.get('finished',time.monotonic())-j['started'];out['maxIterations']=j['config']['maxIterations'];out['maxRounds']=j['config']['rounds'];return out
    def cancel(self,ident):
        with self.lock:
            j=self.jobs[ident]
            if j['state'] in ['queued','running','cancelling']:
                j['cancel']=True;j['state']='cancelling';j['message']='Cancelling solver process'
                if j['process'] and j['process'].poll() is None:
                    try:os.killpg(j['process'].pid,signal.SIGTERM)
                    except ProcessLookupError:pass
            return self.status(ident)
    def result(self,ident):
        with self.lock:
            job=self.jobs[ident]
            if job['state']!='succeeded':raise ValueError('No converged feasible result is available')
            return json.loads((job['directory']/'result.json').read_text())
    def _run(self,j):
        config=j['directory']/'settings.json';output=j['directory']/'result.json';config.write_text(json.dumps(j['config']))
        try:
            for attempt in range(1,3):
                with self.lock:
                    if j['cancel']:break
                    j.update(state='running',attempt=attempt)
                    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
                    process=subprocess.Popen([self.python,'-u',str(ROOT/'planner_worker.py'),'--root',str(self.root),'--config',str(config),'--result',str(output)],cwd=j['directory'],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
                    j['process']=process
                # Killing the whole session also releases native solver threads.
                def deadline():
                    if process.poll() is None:
                        with self.lock:j['message']='Planning time budget exhausted'
                        try:os.killpg(process.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                timer=threading.Timer(2*j['config']['rounds']*j['config']['wallTime']+120,deadline);timer.daemon=True;timer.start()
                try:
                    with open(j['directory']/'solver.log','a') as log:
                        for line in process.stdout:
                            log.write(line);log.flush()
                            with self.lock:
                                if line.startswith('@planner '):
                                    try:
                                        update=json.loads(line[len('@planner '):]);j.update({k:v for k,v in update.items() if k in ['stage','message','iteration','round','objective','primal','dual']})
                                    except ValueError:pass
                                elif line.strip():j['logs']=(j['logs']+[line.rstrip()])[-30:]
                    code=process.wait()
                finally:timer.cancel();process.stdout.close()
                if j['cancel']:break
                if code==0 and output.exists():
                    # Parse before declaring success; partial files are not results.
                    data=json.loads(output.read_text())
                    if data.get('schema')!='urdf-viewer.whole-body.v1' or not data.get('report',{}).get('discreteDynamicsFeasible'):raise RuntimeError('Invalid solver result')
                    with self.lock:j.update(state='succeeded',stage='complete',message='Feasible discrete whole-body trajectory ready',finished=time.monotonic())
                    return
                if code not in [-signal.SIGSEGV,-signal.SIGABRT] or attempt==2:break
                with self.lock:j.update(stage='retrying',message='Native solver crashed; retrying in a fresh process')
            with self.lock:j.update(state='cancelled' if j['cancel'] else 'failed',finished=time.monotonic())
        except Exception as error:
            with self.lock:j.update(state='cancelled' if j['cancel'] else 'failed',message=str(error),finished=time.monotonic())
    def shutdown(self):
        for ident in list(self.jobs):self.cancel(ident)

class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,jobs,**kwargs):self.jobs=jobs;super().__init__(*args,directory=str(ROOT),**kwargs)
    def json_response(self,value,status=200):
        data=json.dumps(value,allow_nan=False).encode();self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
    def do_GET(self):
        path=unquote(urlsplit(self.path).path)
        try:
            if path=='/api/planner/capabilities':return self.json_response(self.jobs.capabilities())
            if path=='/api/planner/model':
                data=model_xml(self.jobs.root).encode();self.send_response(200);self.send_header('Content-Type','application/xml');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data);return
            if path.startswith('/api/planner/assets/'):
                asset_root=self.jobs.root/'src/robots/go2/go2';asset=(asset_root/path[len('/api/planner/assets/'):]).resolve()
                if asset_root.resolve() not in asset.parents or not asset.is_file() or asset.suffix.lower() not in ['.dae','.stl','.obj','.mtl','.png','.jpg','.jpeg']:return self.send_error(404)
                self.send_response(200);self.send_header('Content-Type',self.guess_type(str(asset)));self.send_header('Content-Length',str(asset.stat().st_size));self.end_headers()
                with asset.open('rb') as f:self.copyfile(f,self.wfile)
                return
            if path.startswith('/api/planner/jobs/'):
                parts=path.strip('/').split('/')
                if len(parts)<4:return self.send_error(404)
                ident=parts[3]
                if len(parts)==4:return self.json_response(self.jobs.status(ident))
                if len(parts)==5 and parts[4]=='result':return self.json_response(self.jobs.result(ident))
                return self.send_error(404)
            if path.startswith('/api/'):return self.json_response({'error':'Unknown API'},404)
            if any(p.startswith('.') for p in Path(path).parts if p not in ['/','.']):return self.send_error(404)
            return super().do_GET()
        except KeyError:return self.json_response({'error':'Job not found'},404)
        except (ValueError,OSError) as e:return self.json_response({'error':str(e)},400)
    def do_POST(self):
        origin=self.headers.get('Origin')
        if origin and origin!='http://'+self.headers.get('Host',''):return self.json_response({'error':'Cross-origin requests are not accepted'},403)
        if self.headers.get_content_type()!='application/json':return self.json_response({'error':'Use application/json'},415)
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=32768:raise ValueError('Invalid request size')
            body=json.loads(self.rfile.read(size));path=urlsplit(self.path).path
            if path=='/api/planner/jobs':return self.json_response(self.jobs.start(body.get('settings',{}),body.get('modelHash')),202)
            if path.startswith('/api/planner/jobs/') and path.endswith('/cancel'):return self.json_response(self.jobs.cancel(path.split('/')[4]))
            return self.json_response({'error':'Unknown API'},404)
        except KeyError:return self.json_response({'error':'Job not found'},404)
        except RuntimeError as e:return self.json_response({'error':str(e)},409)
        except (ValueError,TypeError,AttributeError) as e:return self.json_response({'error':str(e)},400)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=8080);parser.add_argument('--bind',default='127.0.0.1');parser.add_argument('--solver-root',default=str(ROOT.parent/'go2_flip_TO'));parser.add_argument('--solver-python',default=str(ROOT.parent/'se3_trajopt/.conda/bin/python'))
    args=parser.parse_args();jobs=PlannerJobs(args.solver_root,args.solver_python)
    server=ThreadingHTTPServer((args.bind,args.port),partial(Handler,jobs=jobs));print(f'URDF-Viewer: http://{args.bind}:{server.server_port} | planner: '+('ready' if jobs.available else jobs.problem),flush=True)
    def stop(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:jobs.shutdown();server.server_close()
if __name__=='__main__':main()
