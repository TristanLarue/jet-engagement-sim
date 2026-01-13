import os, math, pickle
from pathlib import Path
import numpy as np
import tensorflow as tf
import physics

REWARD_PHASE=5
PHASE=5
TRAINING=os.environ.get('JET_TRAINING','1').strip().lower() not in ('0','false','no','n')
ACTION_SKIP=4

DT=1.0/60.0
SD=34
AD=4
CAP=250000
BATCH=256
START=5000
UPD_PER=4
GAMMA=0.99
TAU=0.005
LR=3e-4
LOG_EVERY=200
SAVE_EVERY=2000

_ROOT=Path(__file__).resolve().parents[2]
_DIR=_ROOT/'res'/'models'/'jet_ai'
_MODEL=_DIR/'jet_ai'
_LOG=_DIR/'train.log'
_MIRR=(Path(__file__).resolve().parent/'jet_ai', _ROOT/'jet_ai')

_actor=_q1=_q2=_q1t=_q2t=None
_ao=_q1o=_q2o=_alo=None
_log_alpha=None

_s=_a=_r=_s2=_d=None
_ptr=0
_sz=0

_started=False
_last_s=None
_last_a=None
_hold=0
_racc=0.0
_steps=0
_upd=0
_rhist=[]


def _missiles(entities, jet):
    jp=np.asarray(getattr(jet,'position',[0.0,0.0,0.0]),dtype=float).reshape(3)
    jv=np.asarray(getattr(jet,'velocity',[0.0,0.0,0.0]),dtype=float).reshape(3)
    out=[]
    for e in entities:
        if e is jet or not bool(getattr(e,'alive',True)):
            continue
        if e.__class__.__name__.lower()!='missile':
            continue
        mp=np.asarray(getattr(e,'position',[0.0,0.0,0.0]),dtype=float).reshape(3)
        mv=np.asarray(getattr(e,'velocity',[0.0,0.0,0.0]),dtype=float).reshape(3)
        rel=mp-jp
        d=float(np.linalg.norm(rel))
        if d>1e-6 and np.isfinite(d):
            out.append((d,rel,mv-jv,mv,mp))
    out.sort(key=lambda t:t[0])
    return out


def _state(entities, jet):
    p=np.asarray(getattr(jet,'position',[0.0,0.0,0.0]),dtype=float).reshape(3)
    v=np.asarray(getattr(jet,'velocity',[0.0,0.0,0.0]),dtype=float).reshape(3)
    R=np.asarray(getattr(jet,'orientation',np.eye(3)),dtype=float).reshape(3,3)
    w=np.asarray(getattr(jet,'omega',[0.0,0.0,0.0]),dtype=float).reshape(3)
    if not (np.isfinite(p).all() and np.isfinite(v).all() and np.isfinite(R).all() and np.isfinite(w).all()):
        return np.zeros(SD,dtype=np.float32)
    vb=R.T@v
    sp=float(np.linalg.norm(v))
    aoa=float(physics.get_angle_of_attack(v,R))
    slip=float(physics.get_sideslip(v,R))
    f=physics.get_forward_dir(R)
    u=physics.get_up_dir(R)
    ms=_missiles(entities,jet)
    m=[]
    for i in range(3):
        if i<len(ms):
            _,rel,dv,_,_=ms[i]
            rb=R.T@rel
            dvb=R.T@dv
            m.extend((rb/20000.0).tolist())
            m.extend((dvb/600.0).tolist())
        else:
            m.extend([0.0]*6)
    s=np.concatenate([
        np.array([p[1]/10000.0, sp/600.0],dtype=float),
        np.clip(vb/600.0,-5.0,5.0),
        np.clip(w/6.0,-5.0,5.0),
        np.array([aoa/45.0, slip/45.0],dtype=float),
        np.clip(np.concatenate([f,u]),-2.0,2.0),
        np.clip(np.array(m,dtype=float),-5.0,5.0)
    ],axis=0).astype(np.float32)
    if s.shape[0]!=SD:
        t=np.zeros(SD,dtype=np.float32)
        n=min(SD,int(s.shape[0]))
        t[:n]=s[:n]
        return t
    return s


def _reward(entities, jet):
    if not bool(getattr(jet,'alive',True)):
        return -300.0, True
    p=np.asarray(getattr(jet,'position',[0.0,0.0,0.0]),dtype=float).reshape(3)
    v=np.asarray(getattr(jet,'velocity',[0.0,0.0,0.0]),dtype=float).reshape(3)
    R=np.asarray(getattr(jet,'orientation',np.eye(3)),dtype=float).reshape(3,3)
    if not (np.isfinite(p).all() and np.isfinite(v).all() and np.isfinite(R).all()):
        setattr(jet,'alive',False)
        return -300.0, True
    alt=float(p[1])
    sp=float(np.linalg.norm(v))
    aoa=abs(float(physics.get_angle_of_attack(v,R)))
    r=0.02*DT
    r+=0.30*np.clip((sp-120.0)/300.0,0.0,1.0)*DT
    r-=0.90*np.clip((120.0-sp)/120.0,0.0,1.0)*DT
    r-=1.60*np.clip((3000.0-alt)/3000.0,0.0,3.0)**2*DT
    r-=0.90*np.clip((alt-7000.0)/3000.0,0.0,3.0)**2*DT
    if aoa>18.0:
        r-=2.20*np.clip((aoa-18.0)/18.0,0.0,3.0)**2*DT
    if int(REWARD_PHASE)>=3:
        ms=_missiles(entities,jet)[:3]
        jp=p
        for d,_,_,mv,mp in ms:
            rel=jp-mp
            dn=float(np.linalg.norm(rel))
            mvn=float(np.linalg.norm(mv))
            if dn<1e-6 or mvn<1e-6:
                continue
            c=float(np.clip(np.dot(mv/mvn, rel/dn),-1.0,1.0))
            close=float(np.clip(1.0-d/8000.0,0.0,1.0))
            r+=0.70*close*(1.0-c)*DT
            r-=1.60*close*(max(0.0,c)**2)*DT
            r-=2.20*np.clip((2000.0-d)/2000.0,0.0,2.0)**2*DT
    return float(r), False


def _to_ctrl(a):
    a=np.clip(np.asarray(a,dtype=float).reshape(AD),-1.0,1.0)
    return float(a[0]),float(a[1]),float(a[2]),float((a[3]+1.0)*0.5)


def _build_actor():
    s=tf.keras.Input(shape=(SD,),dtype=tf.float32)
    x=tf.keras.layers.Dense(256,activation='relu')(s)
    x=tf.keras.layers.Dense(256,activation='relu')(x)
    mu=tf.keras.layers.Dense(AD)(x)
    ls=tf.keras.layers.Dense(AD)(x)
    return tf.keras.Model(s,[mu,ls])


def _build_q():
    s=tf.keras.Input(shape=(SD,),dtype=tf.float32)
    a=tf.keras.Input(shape=(AD,),dtype=tf.float32)
    x=tf.keras.layers.Concatenate()([s,a])
    x=tf.keras.layers.Dense(256,activation='relu')(x)
    x=tf.keras.layers.Dense(256,activation='relu')(x)
    q=tf.keras.layers.Dense(1)(x)
    return tf.keras.Model([s,a],q)


def _pi(mu,ls,det):
    ls=tf.clip_by_value(ls,-5.0,2.0)
    if det:
        a=tf.tanh(mu)
        lp=tf.zeros((tf.shape(mu)[0],),tf.float32)
        return a,lp
    std=tf.exp(ls)
    eps=tf.random.normal(tf.shape(mu))
    u=mu+std*eps
    a=tf.tanh(u)
    lp=-0.5*tf.reduce_sum(((u-mu)/(std+1e-6))**2+2.0*ls+math.log(2.0*math.pi),axis=1)
    lp-=tf.reduce_sum(tf.math.log(1.0-a*a+1e-6),axis=1)
    return a,lp


def _train_step(s,a,r,s2,d):
    alpha=tf.exp(_log_alpha)
    with tf.GradientTape(persistent=True) as t:
        mu2,ls2=_actor(s2,training=True)
        a2,lp2=_pi(mu2,ls2,False)
        q1t=_q1t([s2,a2],training=True)
        q2t=_q2t([s2,a2],training=True)
        y=r[:,None]+GAMMA*(1.0-d[:,None])*(tf.minimum(q1t,q2t)-alpha*lp2[:,None])
        q1=_q1([s,a],training=True)
        q2=_q2([s,a],training=True)
        lq1=tf.reduce_mean((q1-y)**2)
        lq2=tf.reduce_mean((q2-y)**2)
        mu,ls=_actor(s,training=True)
        ap,lp=_pi(mu,ls,False)
        qpi=tf.minimum(_q1([s,ap],training=True),_q2([s,ap],training=True))
        la=tf.reduce_mean(alpha*lp-tf.squeeze(qpi,1))
        lt=-tf.reduce_mean(_log_alpha*(lp-float(AD)))
    _q1o.apply_gradients(zip(t.gradient(lq1,_q1.trainable_variables),_q1.trainable_variables))
    _q2o.apply_gradients(zip(t.gradient(lq2,_q2.trainable_variables),_q2.trainable_variables))
    _ao.apply_gradients(zip(t.gradient(la,_actor.trainable_variables),_actor.trainable_variables))
    _alo.apply_gradients([(t.gradient(lt,[_log_alpha])[0],_log_alpha)])
    _log_alpha.assign(tf.clip_by_value(_log_alpha,-10.0,2.0))
    for v,tv in zip(_q1.variables,_q1t.variables):
        tv.assign((1.0-TAU)*tv+TAU*v)
    for v,tv in zip(_q2.variables,_q2t.variables):
        tv.assign((1.0-TAU)*tv+TAU*v)
    return lq1,lq2,la,alpha


_train_step=tf.function(_train_step,reduce_retracing=True)


def _add(s,a,r,s2,d):
    global _ptr,_sz
    _s[_ptr]=s
    _a[_ptr]=a
    _r[_ptr]=r
    _s2[_ptr]=s2
    _d[_ptr]=d
    _ptr=(_ptr+1)%CAP
    _sz=_sz+1 if _sz<CAP else CAP


def _batch():
    idx=np.random.randint(0,_sz,size=BATCH)
    return (
        tf.convert_to_tensor(_s[idx],dtype=tf.float32),
        tf.convert_to_tensor(_a[idx],dtype=tf.float32),
        tf.convert_to_tensor(_r[idx],dtype=tf.float32),
        tf.convert_to_tensor(_s2[idx],dtype=tf.float32),
        tf.convert_to_tensor(_d[idx],dtype=tf.float32),
    )


def _log_line(s):
    try:
        _DIR.mkdir(parents=True,exist_ok=True)
        with open(_LOG,'a',encoding='utf-8') as f:
            f.write(s+'\n')
    except Exception:
        pass
    for m in _MIRR:
        try:
            m.mkdir(parents=True,exist_ok=True)
            with open(m/'train.log','a',encoding='utf-8') as f:
                f.write(s+'\n')
        except Exception:
            pass


def _save():
    try:
        _DIR.mkdir(parents=True,exist_ok=True)
        d={
            'w_actor':_actor.get_weights(),
            'w_q1':_q1.get_weights(),
            'w_q2':_q2.get_weights(),
            'w_q1t':_q1t.get_weights(),
            'w_q2t':_q2t.get_weights(),
            'log_alpha':float(_log_alpha.numpy()),
            'buf':{'s':_s,'a':_a,'r':_r,'s2':_s2,'d':_d,'ptr':_ptr,'sz':_sz},
            'steps':_steps,
            'upd':_upd,
        }
        with open(_MODEL,'wb') as f:
            pickle.dump(d,f,protocol=4)
    except Exception:
        pass


def _load():
    global _ptr,_sz,_steps,_upd
    try:
        if not _MODEL.exists():
            return
        with open(_MODEL,'rb') as f:
            d=pickle.load(f)
        if isinstance(d,dict):
            if 'w_actor' in d: _actor.set_weights(d['w_actor'])
            if 'w_q1' in d: _q1.set_weights(d['w_q1'])
            if 'w_q2' in d: _q2.set_weights(d['w_q2'])
            if 'w_q1t' in d: _q1t.set_weights(d['w_q1t'])
            if 'w_q2t' in d: _q2t.set_weights(d['w_q2t'])
            if 'log_alpha' in d: _log_alpha.assign(float(d['log_alpha']))
            b=d.get('buf',{})
            if isinstance(b,dict) and all(k in b for k in ('s','a','r','s2','d','ptr','sz')):
                global _s,_a,_r,_s2,_d
                if b['s'].shape==_s.shape: _s[:]=b['s']
                if b['a'].shape==_a.shape: _a[:]=b['a']
                if b['r'].shape==_r.shape: _r[:]=b['r']
                if b['s2'].shape==_s2.shape: _s2[:]=b['s2']
                if b['d'].shape==_d.shape: _d[:]=b['d']
                _ptr=int(b.get('ptr',0))%CAP
                _sz=int(b.get('sz',0))
            _steps=int(d.get('steps',_steps))
            _upd=int(d.get('upd',_upd))
    except Exception:
        pass


def initialize_deeplearning():
    global PHASE,_actor,_q1,_q2,_q1t,_q2t,_ao,_q1o,_q2o,_alo,_log_alpha,_s,_a,_r,_s2,_d
    PHASE=int(REWARD_PHASE)
    if _actor is not None:
        return
    _DIR.mkdir(parents=True,exist_ok=True)
    _actor=_build_actor()
    _q1=_build_q(); _q2=_build_q()
    _q1t=_build_q(); _q2t=_build_q()
    _log_alpha=tf.Variable(0.0,dtype=tf.float32)
    _ao=tf.keras.optimizers.Adam(LR)
    _q1o=tf.keras.optimizers.Adam(LR)
    _q2o=tf.keras.optimizers.Adam(LR)
    _alo=tf.keras.optimizers.Adam(LR)
    ds=tf.zeros((1,SD),tf.float32)
    da=tf.zeros((1,AD),tf.float32)
    _actor(ds); _q1([ds,da]); _q2([ds,da]); _q1t([ds,da]); _q2t([ds,da])
    _q1t.set_weights(_q1.get_weights()); _q2t.set_weights(_q2.get_weights())
    _s=np.zeros((CAP,SD),dtype=np.float32)
    _a=np.zeros((CAP,AD),dtype=np.float32)
    _r=np.zeros((CAP,),dtype=np.float32)
    _s2=np.zeros((CAP,SD),dtype=np.float32)
    _d=np.zeros((CAP,),dtype=np.float32)
    _load()


def cleanup_deeplearning():
    global _started,_last_s,_last_a,_hold,_racc
    if TRAINING and _started and _last_s is not None and _last_a is not None:
        _add(_last_s,_last_a,_racc,_last_s,1.0)
    _save()
    _started=False
    _last_s=None
    _last_a=None
    _hold=0
    _racc=0.0


def jet_ai_step(entities, jet):
    global PHASE,_started,_last_s,_last_a,_hold,_racc,_steps,_upd,_rhist
    PHASE=int(REWARD_PHASE)
    initialize_deeplearning()
    s=_state(entities,jet)
    if not _started:
        _started=True
        _last_s=s
        st=tf.convert_to_tensor(s[None,:],tf.float32)
        mu,ls=_actor(st,training=False)
        a,_=_pi(mu,ls,not TRAINING)
        _last_a=a.numpy()[0].astype(np.float32)
        _hold=0
        _racc=0.0
        return _to_ctrl(_last_a)
    r,done=_reward(entities,jet)
    setattr(jet,'current_reward',float(r))
    _racc+=float(r)
    _hold+=1
    if done or _hold>=int(ACTION_SKIP):
        if TRAINING and _last_s is not None and _last_a is not None:
            _add(_last_s,_last_a,_racc,s,1.0 if done else 0.0)
            _rhist.append(float(_racc))
            if len(_rhist)>800:
                _rhist=_rhist[-800:]
            _steps+=1
            if _sz>=START:
                for _ in range(int(UPD_PER)):
                    bs,ba,br,bs2,bd=_batch()
                    _train_step(bs,ba,br,bs2,bd)
                    _upd+=1
                    if _upd%LOG_EVERY==0:
                        av=float(np.mean(_rhist[-200:])) if _rhist else 0.0
                        _log_line(f"[SAC update #{_upd//LOG_EVERY}] steps={_steps} avg_r/transition={av:.6f} phase={PHASE}")
                    if _upd%SAVE_EVERY==0:
                        _save()
        _last_s=s
        _racc=0.0
        _hold=0
        if done:
            _last_a=np.zeros((AD,),dtype=np.float32)
            return _to_ctrl(_last_a)
        st=tf.convert_to_tensor(s[None,:],tf.float32)
        mu,ls=_actor(st,training=False)
        a,_=_pi(mu,ls,not TRAINING)
        _last_a=a.numpy()[0].astype(np.float32)
    return _to_ctrl(_last_a)
