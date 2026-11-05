"""Minimal Chrome DevTools client for the dependency-free browser smoke test."""
import base64, json, os, socket, struct, time, urllib.request
from urllib.parse import urlparse
class CDP:
 def __init__(self, port=9224):
  targets=json.load(urllib.request.build_opener(urllib.request.ProxyHandler({})).open(f'http://127.0.0.1:{port}/json'))
  target=next(t for t in targets if t['type']=='page')
  u=urlparse(target['webSocketDebuggerUrl']); self.s=socket.create_connection((u.hostname,u.port)); self.s.settimeout(30)
  key=base64.b64encode(os.urandom(16)).decode()
  self.s.sendall((f'GET {u.path} HTTP/1.1\r\nHost: {u.netloc}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\nOrigin: http://localhost:{port}\r\n\r\n').encode())
  data=b''
  while b'\r\n\r\n' not in data:data+=self.s.recv(1)
  assert b'101' in data,data
  self.i=0;self.events=[]
 def receive(self,n):
  data=b''
  while len(data)<n:data+=self.s.recv(n-len(data))
  return data
 def frame(self):
  a,b=self.receive(2); n=b&127
  if n==126:n=struct.unpack('!H',self.receive(2))[0]
  elif n==127:n=struct.unpack('!Q',self.receive(8))[0]
  mask=self.receive(4) if b&128 else None;data=self.receive(n)
  if mask:data=bytes(v^mask[i%4] for i,v in enumerate(data))
  return json.loads(data)
 def call(self,method,params=None):
  self.i+=1;ident=self.i;data=json.dumps({'id':ident,'method':method,'params':params or {}}).encode();mask=os.urandom(4);n=len(data)
  header=bytes([129,128|n]) if n<126 else bytes([129,254])+struct.pack('!H',n) if n<65536 else bytes([129,255])+struct.pack('!Q',n)
  self.s.sendall(header+mask+bytes(v^mask[i%4] for i,v in enumerate(data)))
  while True:
   result=self.frame()
   if result.get('id')==ident:
    if 'error' in result:raise RuntimeError(result['error'])
    return result.get('result',{})
   self.events.append(result)
 def js(self,expression):
  result=self.call('Runtime.evaluate',{'expression':expression,'returnByValue':True,'awaitPromise':True})
  if 'exceptionDetails' in result:raise RuntimeError(result['exceptionDetails'])
  return result.get('result',{}).get('value')
 def until(self,expression,timeout=30):
  deadline=time.time()+timeout
  while time.time()<deadline:
   value=self.js(expression)
   if value:return value
   time.sleep(.15)
  raise AssertionError('Timeout: '+expression)
 def screenshot(self,path):
  result=self.call('Page.captureScreenshot',{'format':'png'})
  open(path,'wb').write(base64.b64decode(result['data']))

 def close(self):
  self.s.close()
