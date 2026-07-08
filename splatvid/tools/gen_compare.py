import base64, os

ROOT = "/Users/rho/dev/jabberwocky/splatvid"
OUT = ROOT + "/compare.html"

# reconstruction ladder (same scene, increasing gaussian budget)
SCENES = [
    ("out_mesh_e2e/scene.splat", "Fast", "old SfM · 250 iters"),
    ("out_img6547/scene.splat", "Short", "old SfM · 800 iters"),
    ("out_neural_hq/scene.splat", "Neural-geom", "old SfM · 1000 iters"),
    ("out_img6547_hq/scene.splat", "High", "old SfM · 2000 iters"),
    ("out_sota/scene.splat", "SOTA ★", "DISK+LightGlue SfM · 1800 iters"),
]

data = []
for path, name, note in SCENES:
    b = open(os.path.join(ROOT, path), "rb").read()
    data.append({
        "name": name, "note": note,
        "count": len(b) // 32,
        "b64": base64.b64encode(b).decode(),
    })

scenes_js = ",\n".join(
    '{name:"%s",note:"%s",count:%d,b64:"%s"}' % (d["name"], d["note"], d["count"], d["b64"])
    for d in data
)

# ---- mesh iterations (TSDF surface reconstructions) + the depth study ----
import numpy as np, open3d as o3d

MESHES = [
    ("out_img6547_hq/mesh.ply", "TSDF-HQ (old)", "old SfM · mean depth", "old"),
    ("out_img6547_hq/mesh_mono.ply", "Depth-prior (old)", "old SfM · mono depth", "old"),
    ("out_sota/mesh_fine_smooth.ply", "SOTA-Fine ★", "new SfM · median · --mesh-fine", "new"),
    ("out_sota/mesh_poisson.ply", "SOTA-Poisson", "dense-cloud screened Poisson", "new"),
]
TARGET_FACES = 150_000


def pack_mesh(path):
    m = o3d.io.read_triangle_mesh(path)
    native = len(m.triangles)
    if native > TARGET_FACES:
        m = m.simplify_quadric_decimation(TARGET_FACES)
    m.remove_unreferenced_vertices()
    v = np.asarray(m.vertices, np.float32)
    f = np.asarray(m.triangles).astype(np.uint32)
    c = np.asarray(m.vertex_colors)
    if len(c) == 0:
        c = np.full((len(v), 3), 0.7)
    col = (np.clip(c, 0, 1) * 255).astype(np.uint8)
    center = np.median(v, axis=0)
    dist = np.linalg.norm(v - center, axis=1)
    radius = float(np.percentile(dist, 70) * 1.8)
    b64 = lambda a: base64.b64encode(a.tobytes()).decode()  # noqa: E731
    return {
        "native": native, "nV": len(v), "nF": len(f),
        "center": [float(x) for x in center], "radius": radius,
        "pos": b64(v), "col": b64(col), "idx": b64(f),
    }


mesh_entries = []
for path, name, note, depth in MESHES:
    d = pack_mesh(os.path.join(ROOT, path))
    d.update(name=name, note=note, depth=depth)
    mesh_entries.append(d)
    print(f"  mesh {name:12s} native={d['native']:>8,}f -> {d['nF']:>7,}f  V={d['nV']:,}")

meshes_js = ",\n".join(
    '{name:"%s",note:"%s",depth:"%s",native:%d,nF:%d,center:[%s],radius:%.6f,'
    'pos:"%s",col:"%s",idx:"%s"}' % (
        d["name"], d["note"], d["depth"], d["native"], d["nF"],
        ",".join("%.6f" % x for x in d["center"]), d["radius"],
        d["pos"], d["col"], d["idx"],
    )
    for d in mesh_entries
)

# ---- shaders (match splatvid/viewer.html: 2.5-sigma + fwidth AA) ----
VS = r"""#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
in vec2 corner; in float sortedIndex;
uniform sampler2D uData; uniform int uTexW;
uniform mat4 uView, uProj; uniform vec2 uFocal, uViewport;
out vec4 vColor; out vec2 vG;
vec4 texel(int i,int k){ int idx=i*4+k; return texelFetch(uData, ivec2(idx%uTexW, idx/uTexW),0); }
void main(){
  int i=int(sortedIndex+0.5);
  vec4 t0=texel(i,0),t1=texel(i,1),t2=texel(i,2),t3=texel(i,3);
  vec3 center=t0.xyz; vec4 cam=uView*vec4(center,1.0); float z=-cam.z;
  if(z<0.05){ gl_Position=vec4(0.,0.,2.,1.); return; }
  vec4 q=normalize(t2); float w=q.x,x=q.y,y=q.z,zz=q.w;
  mat3 R=mat3(1.-2.*(y*y+zz*zz),2.*(x*y+w*zz),2.*(x*zz-w*y),
             2.*(x*y-w*zz),1.-2.*(x*x+zz*zz),2.*(y*zz+w*x),
             2.*(x*zz+w*y),2.*(y*zz-w*x),1.-2.*(x*x+y*y));
  mat3 S=mat3(t1.x,0.,0., 0.,t1.y,0., 0.,0.,t1.z);
  mat3 M=R*S; mat3 cov3d=M*transpose(M);
  mat3 W=mat3(uView);
  W=mat3(vec3(W[0].xy,-W[0].z),vec3(W[1].xy,-W[1].z),vec3(W[2].xy,-W[2].z));
  mat3 J=mat3(uFocal.x/z,0.,0., 0.,uFocal.y/z,0., -uFocal.x*cam.x/(z*z),-uFocal.y*cam.y/(z*z),0.);
  mat3 T=J*W; mat3 cov2d=T*cov3d*transpose(T);
  float a=cov2d[0][0]+0.3,b=cov2d[1][0],c=cov2d[1][1]+0.3;
  float mid=0.5*(a+c); float disc=sqrt(max(0.01,mid*mid-(a*c-b*b)));
  float l1=mid+disc,l2=max(mid-disc,0.01);
  vec2 e1=normalize(vec2(b,l1-a));
  if(abs(b)<1e-9&&a>=c) e1=vec2(1.,0.);
  if(abs(b)<1e-9&&a<c) e1=vec2(0.,1.);
  vec2 e2=vec2(-e1.y,e1.x);
  vec2 axis1=e1*sqrt(l1)*2.5, axis2=e2*sqrt(l2)*2.5;
  vec4 clip=uProj*cam; vec2 ndc=clip.xy/clip.w;
  vec2 off=(corner.x*axis1+corner.y*axis2)*2.0/uViewport;
  gl_Position=vec4(ndc+off,0.,1.); vG=corner; vColor=vec4(t3.rgb,t0.w);
}"""

FS = r"""#version 300 es
precision highp float;
in vec4 vColor; in vec2 vG; out vec4 frag;
void main(){
  float d2=dot(vG,vG);
  float alpha=vColor.a*exp(-0.5*6.25*d2);
  alpha*=1.0-smoothstep(1.0-fwidth(d2),1.0,d2);
  if(alpha<0.004) discard;
  frag=vec4(vColor.rgb*alpha,alpha);
}"""

JS = r"""
const VS=`__VS__`, FS=`__FS__`;
const SCENES=[__SCENES__];
const sub3=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const dot3=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross3=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const norm3=a=>{const l=Math.hypot(a[0],a[1],a[2])||1;return[a[0]/l,a[1]/l,a[2]/l];};
function viewMatrix(cam){
  const ct=Math.cos(cam.theta),st=Math.sin(cam.theta),cp=Math.cos(cam.phi),sp=Math.sin(cam.phi);
  const eye=[cam.target[0]+cam.radius*sp*ct,cam.target[1]+cam.radius*cp,cam.target[2]+cam.radius*sp*st];
  const f=norm3(sub3(cam.target,eye)); const r=norm3(cross3(f,[0,1,0])); const u=cross3(r,f);
  const zax=[-f[0],-f[1],-f[2]];
  return[r[0],u[0],zax[0],0, r[1],u[1],zax[1],0, r[2],u[2],zax[2],0,
    -dot3(r,eye),-dot3(u,eye),-dot3(zax,eye),1];
}
function projMatrix(cam,aspect){
  const f=1/Math.tan(cam.fov/2),near=0.01,far=1000;
  return[f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)/(near-far),-1, 0,0,(2*far*near)/(near-far),0];
}
function b64buf(s){const bin=atob(s);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u.buffer;}

class SplatView{
  constructor(canvas,buffer){
    this.canvas=canvas;
    const gl=this.gl=canvas.getContext("webgl2",{antialias:false,alpha:false});
    if(!gl){this.dead=true;return;}
    const cm=(t,s)=>{const sh=gl.createShader(t);gl.shaderSource(sh,s);gl.compileShader(sh);
      if(!gl.getShaderParameter(sh,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(sh));return sh;};
    const prog=this.prog=gl.createProgram();
    gl.attachShader(prog,cm(gl.VERTEX_SHADER,VS));gl.attachShader(prog,cm(gl.FRAGMENT_SHADER,FS));
    gl.linkProgram(prog);gl.useProgram(prog);
    this.U=n=>gl.getUniformLocation(prog,n);
    const quad=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,quad);
    gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,1,1]),gl.STATIC_DRAW);
    const lc=gl.getAttribLocation(prog,"corner");gl.enableVertexAttribArray(lc);gl.vertexAttribPointer(lc,2,gl.FLOAT,false,0,0);
    this.idxBuf=gl.createBuffer();const li=gl.getAttribLocation(prog,"sortedIndex");
    gl.bindBuffer(gl.ARRAY_BUFFER,this.idxBuf);gl.enableVertexAttribArray(li);
    gl.vertexAttribPointer(li,1,gl.FLOAT,false,0,0);gl.vertexAttribDivisor(li,1);
    gl.disable(gl.DEPTH_TEST);gl.enable(gl.BLEND);gl.blendFunc(gl.ONE,gl.ONE_MINUS_SRC_ALPHA);
    this.N=0;this.lastSortZ=null;this.needSort=true;
    this.counts=new Uint32Array(65536);
    this.load(buffer);
  }
  load(buffer){
    const gl=this.gl,n=Math.floor(buffer.byteLength/32),f32=new Float32Array(buffer),u8=new Uint8Array(buffer);
    const med=a=>{const s=Float32Array.from(a).sort();return s.length?s[s.length>>1]:0;};
    const xs=new Float32Array(n),ys=new Float32Array(n),zs=new Float32Array(n);
    for(let i=0;i<n;i++){const o=i*8;xs[i]=f32[o];ys[i]=f32[o+1];zs[i]=f32[o+2];}
    const center=[med(xs),med(ys),med(zs)];
    const dist=new Float32Array(n);
    for(let i=0;i<n;i++){const dx=xs[i]-center[0],dy=ys[i]-center[1],dz=zs[i]-center[2];dist[i]=Math.hypot(dx,dy,dz);}
    const sorted=Float32Array.from(dist).sort();const pct=p=>sorted[Math.min(n-1,Math.floor(n*p))]||0;
    const cullR=2.0*pct(0.7)||(pct(1)||1);
    const keep=new Uint8Array(n);let m=0;for(let i=0;i<n;i++){if(dist[i]<=cullR){keep[i]=1;m++;}}
    this.N=m;this.positions=new Float32Array(m*3);
    const texW=1024,texH=Math.max(1,Math.ceil((m*4)/texW)),tex=new Float32Array(texW*texH*4);
    let j=0;
    for(let i=0;i<n;i++){if(!keep[i])continue;const o=i*8,bo=i*32,x=f32[o],y=f32[o+1],z=f32[o+2];
      this.positions[j*3]=x;this.positions[j*3+1]=y;this.positions[j*3+2]=z;const t=j*16;
      tex[t]=x;tex[t+1]=y;tex[t+2]=z;tex[t+3]=u8[bo+27]/255;
      tex[t+4]=f32[o+3];tex[t+5]=f32[o+4];tex[t+6]=f32[o+5];
      tex[t+8]=(u8[bo+28]-128)/128;tex[t+9]=(u8[bo+29]-128)/128;
      tex[t+10]=(u8[bo+30]-128)/128;tex[t+11]=(u8[bo+31]-128)/128;
      tex[t+12]=u8[bo+24]/255;tex[t+13]=u8[bo+25]/255;tex[t+14]=u8[bo+26]/255;j++;}
    this.center=center;this.radius=cullR*1.8;
    const dt=gl.createTexture();gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,dt);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA32F,texW,texH,0,gl.RGBA,gl.FLOAT,tex);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
    gl.uniform1i(this.U("uData"),0);gl.uniform1i(this.U("uTexW"),texW);
    this.order=new Float32Array(m);for(let i=0;i<m;i++)this.order[i]=i;this.needSort=true;
  }
  sort(viewZ){
    const gl=this.gl,N=this.N,P=this.positions,d=new Float32Array(N);let lo=1e30,hi=-1e30;
    for(let i=0;i<N;i++){const v=viewZ[0]*P[i*3]+viewZ[1]*P[i*3+1]+viewZ[2]*P[i*3+2];d[i]=v;if(v<lo)lo=v;if(v>hi)hi=v;}
    const s=65535/Math.max(hi-lo,1e-9);this.counts.fill(0);const bkt=new Uint32Array(N);
    for(let i=0;i<N;i++){const b=((d[i]-lo)*s)|0;bkt[i]=b;this.counts[b]++;}
    let acc=0;const st=new Uint32Array(65536);for(let b=0;b<65536;b++){st[b]=acc;acc+=this.counts[b];}
    for(let i=0;i<N;i++)this.order[st[bkt[i]]++]=i;
    gl.bindBuffer(gl.ARRAY_BUFFER,this.idxBuf);gl.bufferData(gl.ARRAY_BUFFER,this.order,gl.DYNAMIC_DRAW);
  }
  render(cam){
    if(this.dead)return;
    const gl=this.gl,cv=this.canvas,dpr=Math.min(devicePixelRatio||1,2);
    const w=Math.round(cv.clientWidth*dpr),h=Math.round(cv.clientHeight*dpr);
    if(cv.width!==w||cv.height!==h){cv.width=w;cv.height=h;}
    gl.useProgram(this.prog);gl.viewport(0,0,w,h);gl.clearColor(0.03,0.04,0.06,1);gl.clear(gl.COLOR_BUFFER_BIT);
    if(!this.N)return;
    const v=viewMatrix(cam),viewZ=[v[2],v[6],v[10]];
    if(this.needSort||!this.lastSortZ||dot3(viewZ,this.lastSortZ)<0.9997){this.sort(viewZ);this.lastSortZ=viewZ.slice();this.needSort=false;}
    const fpx=0.5*h/Math.tan(cam.fov/2);
    gl.uniformMatrix4fv(this.U("uView"),false,new Float32Array(v));
    gl.uniformMatrix4fv(this.U("uProj"),false,new Float32Array(projMatrix(cam,w/h)));
    gl.uniform2f(this.U("uFocal"),fpx,fpx);gl.uniform2f(this.U("uViewport"),w,h);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP,0,4,this.N);
  }
}

// ---- triangle-mesh viewer (flat-shaded, depth-tested) ----
const MESHES=[__MESHES__];
const MVS=`#version 300 es
in vec3 aPos; in vec3 aCol;
uniform mat4 uMVP, uView;
out vec3 vCol; out vec3 vPosV;
void main(){ vCol=aCol; vPosV=(uView*vec4(aPos,1.0)).xyz; gl_Position=uMVP*vec4(aPos,1.0); }`;
const MFS=`#version 300 es
precision highp float;
in vec3 vCol; in vec3 vPosV; out vec4 frag;
void main(){
  vec3 n=normalize(cross(dFdx(vPosV),dFdy(vPosV)));      // per-face normal, view space
  vec3 L=normalize(vec3(0.35,0.55,0.75));
  float dif=clamp(abs(dot(n,L)),0.0,1.0);                // two-sided
  vec3 c=vCol*(0.42+0.58*dif);
  frag=vec4(c,1.0);
}`;
function mul4(a,b){const o=new Array(16);
  for(let c=0;c<4;c++)for(let r=0;r<4;r++){let s=0;for(let k=0;k<4;k++)s+=a[k*4+r]*b[c*4+k];o[c*4+r]=s;}return o;}
function b64f32(s){return new Float32Array(b64buf(s));}
function b64u8(s){return new Uint8Array(b64buf(s));}
function b64u32(s){return new Uint32Array(b64buf(s));}

class MeshView{
  constructor(canvas,e){
    this.canvas=canvas;this.center=e.center;this.radius=e.radius;this.needSort=false;
    const gl=this.gl=canvas.getContext("webgl2",{antialias:true,alpha:false});
    if(!gl){this.dead=true;return;}
    const cm=(t,s)=>{const sh=gl.createShader(t);gl.shaderSource(sh,s);gl.compileShader(sh);
      if(!gl.getShaderParameter(sh,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(sh));return sh;};
    const p=this.prog=gl.createProgram();
    gl.attachShader(p,cm(gl.VERTEX_SHADER,MVS));gl.attachShader(p,cm(gl.FRAGMENT_SHADER,MFS));
    gl.linkProgram(p);gl.useProgram(p);this.U=n=>gl.getUniformLocation(p,n);
    const pos=b64f32(e.pos),col=b64u8(e.col),idx=b64u32(e.idx);this.nIdx=idx.length;
    const pb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,pb);gl.bufferData(gl.ARRAY_BUFFER,pos,gl.STATIC_DRAW);
    const la=gl.getAttribLocation(p,"aPos");gl.enableVertexAttribArray(la);gl.vertexAttribPointer(la,3,gl.FLOAT,false,0,0);
    const cb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,cb);gl.bufferData(gl.ARRAY_BUFFER,col,gl.STATIC_DRAW);
    const lc=gl.getAttribLocation(p,"aCol");gl.enableVertexAttribArray(lc);gl.vertexAttribPointer(lc,3,gl.UNSIGNED_BYTE,true,0,0);
    const ib=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,ib);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,idx,gl.STATIC_DRAW);
    gl.enable(gl.DEPTH_TEST);gl.disable(gl.CULL_FACE);gl.disable(gl.BLEND);
  }
  render(cam){
    if(this.dead)return;
    const gl=this.gl,cv=this.canvas,dpr=Math.min(devicePixelRatio||1,2);
    const w=Math.round(cv.clientWidth*dpr),h=Math.round(cv.clientHeight*dpr);
    if(cv.width!==w||cv.height!==h){cv.width=w;cv.height=h;}
    gl.useProgram(this.prog);gl.viewport(0,0,w,h);
    gl.clearColor(0.03,0.04,0.06,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
    const view=viewMatrix(cam),mvp=mul4(projMatrix(cam,w/h),view);
    gl.uniformMatrix4fv(this.U("uMVP"),false,new Float32Array(mvp));
    gl.uniformMatrix4fv(this.U("uView"),false,new Float32Array(view));
    gl.drawElements(gl.TRIANGLES,this.nIdx,gl.UNSIGNED_INT,0);
  }
}

// ---- build grids ----
let sync=true;
// Each reconstruction is an INDEPENDENT SfM solve with its own arbitrary scale
// and coordinate frame, so we can't share an absolute world camera. Instead we
// share the VIEWING STATE (orbit angle, zoom, pan) and let each view frame to
// its OWN center/radius. Sync = same angle; each cloud stays correctly framed.
const FOV=60*Math.PI/180;
const DEF=()=>({theta:0.4,phi:1.2,zoom:1,pu:0,pv:0});
let shared=DEF();
const views=[];
function basis(theta,phi){
  const ct=Math.cos(theta),st=Math.sin(theta),cp=Math.cos(phi),sp=Math.sin(phi);
  const f=[-sp*ct,-cp,-sp*st];               // forward = target - eye (unit)
  const r=norm3(cross3(f,[0,1,0])); const u=cross3(r,f);
  return {r,u};
}
function buildCam(v,S){                       // viewing-state -> absolute cam for THIS view
  const {r,u}=basis(S.theta,S.phi),R=v.radius;
  const t=[v.center[0]+(r[0]*S.pu+u[0]*S.pv)*R,
           v.center[1]+(r[1]*S.pu+u[1]*S.pv)*R,
           v.center[2]+(r[2]*S.pu+u[2]*S.pv)*R];
  return {theta:S.theta,phi:S.phi,radius:R*S.zoom,target:t,fov:FOV};
}
function makeCell(gridId,meta,factory){
  const cell=document.createElement("div");cell.className="cell";
  const cv=document.createElement("canvas");
  const bar=document.createElement("div");bar.className="bar";
  bar.innerHTML=`<span class="nm">${meta.nm}${meta.tag?` <i class="tag">${meta.tag}</i>`:""}</span><span class="ct">${meta.ct}</span>`;
  cell.appendChild(cv);cell.appendChild(bar);document.getElementById(gridId).appendChild(cell);
  let view;try{view=factory(cv);}catch(e){bar.innerHTML+=" · <span class=err>"+e.message+"</span>";return;}
  view.S=DEF();views.push(view);attach(cv,view);
}
SCENES.forEach(sc=>makeCell("grid-splat",
  {nm:sc.name,ct:`${sc.count.toLocaleString()} splats · ${sc.note}`},
  cv=>new SplatView(cv,b64buf(sc.b64))));
MESHES.forEach(m=>makeCell("grid-mesh",
  {nm:m.name,tag:m.depth,ct:`${(m.native/1000).toFixed(0)}k faces · ${m.note}`},
  cv=>new MeshView(cv,m)));

function state(view){return sync?shared:view.S;}
function markSort(view){if(sync)views.forEach(v=>v.needSort=true);else view.needSort=true;}
function attach(cv,view){
  let drag=0,lx=0,ly=0;
  cv.addEventListener("pointerdown",e=>{drag=(e.button===2||e.shiftKey)?2:1;lx=e.clientX;ly=e.clientY;cv.setPointerCapture(e.pointerId);});
  cv.addEventListener("pointerup",()=>drag=0);
  cv.addEventListener("pointerleave",()=>drag=0);
  cv.addEventListener("contextmenu",e=>e.preventDefault());
  cv.addEventListener("pointermove",e=>{
    if(!drag)return;const dx=e.clientX-lx,dy=e.clientY-ly;lx=e.clientX;ly=e.clientY;const S=state(view);
    if(drag===1){S.theta+=dx*0.005;S.phi=Math.min(Math.PI-0.05,Math.max(0.05,S.phi-dy*0.005));}
    else{S.pu-=dx*0.0015*S.zoom;S.pv+=dy*0.0015*S.zoom;}
    markSort(view);
  });
  cv.addEventListener("wheel",e=>{e.preventDefault();const S=state(view);S.zoom=Math.min(6,Math.max(0.1,S.zoom*Math.exp(e.deltaY*0.001)));markSort(view);},{passive:false});
}
function loop(){requestAnimationFrame(loop);for(const v of views)v.render(buildCam(v,state(v)));}
loop();

// controls
const syncBtn=document.getElementById("sync");
syncBtn.addEventListener("click",()=>{
  if(sync){views.forEach(v=>{v.S={...shared};v.needSort=true;});}  // fork shared -> per-view
  else{shared={...(views[0]?views[0].S:DEF())};views.forEach(v=>v.needSort=true);}  // adopt first view's angle
  sync=!sync;syncBtn.setAttribute("aria-pressed",sync);
  syncBtn.textContent=sync?"Synced cameras":"Independent cameras";
});
document.getElementById("reset").addEventListener("click",()=>{
  shared=DEF();views.forEach(v=>{v.S=DEF();v.needSort=true;});
});
"""

JS = (JS.replace("__VS__", VS).replace("__FS__", FS)
        .replace("__SCENES__", scenes_js).replace("__MESHES__", meshes_js))

STYLE = """<style>
:root{--bg:#0a0d13;--panel:#111722;--line:#222b3a;--ink:#e8edf5;--muted:#8b97a8;--lime:#c9f542;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
  background-image:radial-gradient(1100px 500px at 75% -8%,rgba(201,245,66,.05),transparent 60%);}
header{padding:34px 28px 20px;max-width:1360px;margin:0 auto}
.kick{font-family:var(--mono);font-size:12px;letter-spacing:.2em;text-transform:uppercase;color:#9dc233;margin:0 0 10px}
h1{font-size:clamp(26px,3.4vw,38px);letter-spacing:-.02em;font-weight:800;margin:0;text-wrap:balance}
h1 em{font-style:normal;color:var(--lime)}
.sub{color:var(--muted);margin:12px 0 0;max-width:70ch;font-size:15.5px}
.toolbar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-top:18px}
button{font-family:var(--mono);font-size:12.5px;letter-spacing:.03em;color:var(--ink);
  background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:9px 14px;cursor:pointer}
button:hover{border-color:#3a4658}
button:focus-visible{outline:2px solid var(--lime);outline-offset:2px}
#sync[aria-pressed="true"]{border-color:rgba(201,245,66,.5);color:var(--lime);background:rgba(201,245,66,.07)}
.hint{font-family:var(--mono);font-size:12px;color:var(--muted)}
section{max-width:1360px;margin:0 auto;padding:0 28px}
.sec-head{display:flex;align-items:baseline;gap:14px;margin:30px 0 4px;padding-top:18px;border-top:1px solid var(--line)}
.sec-head h2{font-size:19px;font-weight:800;letter-spacing:-.01em;margin:0}
.sec-head .n{font-family:var(--mono);font-size:12px;color:#9dc233;letter-spacing:.12em;text-transform:uppercase}
.lead{color:var(--muted);font-size:14px;margin:8px 0 14px;max-width:78ch}
.lead b{color:var(--ink);font-weight:600}
.grid{display:grid;gap:16px;margin:8px 0 8px}
#grid-splat{grid-template-columns:repeat(4,1fr)}
#grid-mesh{grid-template-columns:repeat(3,1fr)}
.cell{display:flex;flex-direction:column;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#05070b}
canvas{display:block;width:100%;aspect-ratio:9/15;touch-action:none;cursor:grab;background:#05070b}
.grid-mesh-wrap canvas,#grid-mesh canvas{aspect-ratio:4/5}
canvas:active{cursor:grabbing}
.bar{display:flex;justify-content:space-between;align-items:baseline;gap:10px;padding:10px 13px;
  border-top:1px solid var(--line);background:var(--panel)}
.nm{font-weight:700;font-size:14px}
.tag{font-style:normal;font-family:var(--mono);font-size:10px;letter-spacing:.04em;color:var(--muted);
  border:1px solid var(--line);border-radius:5px;padding:1px 5px;margin-left:5px;vertical-align:1px}
.ct{font-family:var(--mono);font-size:11.5px;color:var(--muted);text-align:right}
.err{color:#e77b6b}
.callout{border:1px solid rgba(201,245,66,.28);background:rgba(201,245,66,.05);border-radius:10px;
  padding:13px 16px;margin:6px 0 34px;font-size:13.5px;color:#cdd7e4}
.callout b{color:var(--lime);font-weight:700}
footer{max-width:1360px;margin:0 auto;padding:8px 28px 60px;color:var(--muted);font-size:12.5px;font-family:var(--mono)}
@media(max-width:1100px){#grid-splat{grid-template-columns:repeat(2,1fr)}#grid-mesh{grid-template-columns:repeat(3,1fr)}}
@media(max-width:760px){#grid-mesh{grid-template-columns:1fr}}
@media(max-width:560px){#grid-splat{grid-template-columns:1fr}}
</style>"""

HTML = STYLE + f"""
<header>
  <p class="kick">Live · WebGL · before &amp; after the SOTA overhaul</p>
  <h1>Explore each reconstruction <em>yourself</em>.</h1>
  <p class="sub">The same cooler scene, rebuilt every way we tried it — and now the <b style="color:var(--lime)">★ SOTA</b> rebuild on the rewritten SfM (DISK + LightGlue learned matching, 40/40 cameras at 0.61px) with median-depth meshing. Drag to orbit, shift-drag (or right-drag) to pan, scroll to zoom. Each panel frames its own reconstruction; the camera angle syncs across all so you compare like-for-like.</p>
  <div class="toolbar">
    <button id="sync" aria-pressed="true">Synced cameras</button>
    <button id="reset">Reset view</button>
    <span class="hint">drag = orbit · shift-drag = pan · scroll = zoom</span>
  </div>
</header>

<section>
  <div class="sec-head"><span class="n">01</span><h2>Gaussian splats — old SfM budget ladder vs the ★ SOTA rebuild</h2></div>
  <p class="lead">Point-based radiance, sorted &amp; alpha-blended live. The first four ran on the old SIFT + brute-force SfM; <b>SOTA ★</b> runs on the rewritten DISK + LightGlue matcher (denser, sub-pixel poses) with NaN-guarded training. Same 1800-iter budget as the old runs — the difference is the geometry underneath.</p>
  <div class="grid" id="grid-splat"></div>
</section>

<section>
  <div class="sec-head"><span class="n">02</span><h2>Surface meshes — old vs new-SfM median-depth &amp; Poisson</h2></div>
  <p class="lead">Flat-shaded so geometry, not texture, is what you judge. <b>TSDF-HQ</b> / <b>Depth-prior</b> are the old-SfM meshes. <b>SOTA-Fine ★</b> is the new pipeline's best dense mesh — new SfM, <em>median</em> depth (the true front surface, not the opacity-weighted mean), finer voxel + 450k faces via <code>--mesh-fine</code>; <b>SOTA-Poisson</b> runs screened Poisson on the dense median-depth surface cloud. Rendered at ~150k faces for the web (native counts labeled).</p>
  <div class="grid" id="grid-mesh"></div>
  <div class="callout"><b>The result →</b> <b>SOTA-Fine ★</b> is the clear winner — the cooler is a clean, recognizable solid (lid, body, strap) where the old meshes were fragmented noise. <b>SOTA-Poisson</b> fills more holes (92% vs 65% coverage from one view) but over-smooths and adds balloon artifacts, so median-depth TSDF keeps the crisp detail. The lift comes from the rewritten SfM (DISK+LightGlue, 40/40 cams, 0.61px) + median-depth fusion + the finer voxel.</div>
</section>

<footer>splatvid · IMG_6547 · 5 splat runs + 4 meshes · self-contained, no network</footer>
<script>{JS}</script>
"""

open(OUT, "w").write(HTML)
print("wrote", OUT, round(len(HTML) / 1e6, 2), "MB")
