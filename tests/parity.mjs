import * as tf from '@tensorflow/tfjs';
import fs from 'fs';
import { GraphModel } from '../engine/graph.js';
const dir=new URL('../model/', import.meta.url).pathname;
const json=JSON.parse(fs.readFileSync(dir+'model.json'));
const bin=fs.readFileSync(dir+'group1-shard1of1.bin');
const ab=bin.buffer.slice(bin.byteOffset,bin.byteOffset+bin.length);
const N=43844, x=new Float32Array(N);
for(let i=0;i<N;i++){const t=i/22050; x[i]=0.3*Math.sin(2*Math.PI*196*t)+0.2*Math.sin(2*Math.PI*329.63*t)*(t>0.5)+0.05*(Math.random()-0.5);}
const ws=json.weightsManifest[0].weights;
const ref=await tf.loadGraphModel(tf.io.fromMemory({ modelTopology: json.modelTopology, weightSpecs: ws, weightData: ab }));
let t0=Date.now();
const r=ref.execute(tf.tensor(x,[1,N,1]),['Identity','Identity_1','Identity_2']);
console.log('tfjs ms',Date.now()-t0);
const m=new GraphModel(json,ab);
t0=Date.now();
const o=m.execute(x,[1,N,1]);
console.log('mine ms',Date.now()-t0, Object.keys(o));
const names={Identity:'contour',Identity_1:'Identity_1:0',Identity_2:'Identity_2:0'};
r.forEach((t,i)=>{const k=['Identity','Identity_1','Identity_2'][i];const a=t.dataSync();const b=o[names[k]].data;
 let md=0,mx=0;for(let j=0;j<a.length;j++){md=Math.max(md,Math.abs(a[j]-b[j]));mx=Math.max(mx,a[j]);}
 console.log(k,t.shape,o[names[k]].shape,'maxdiff',md,'max',mx);});
