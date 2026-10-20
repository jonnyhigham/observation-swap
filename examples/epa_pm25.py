#!/usr/bin/env python3
"""Second-domain demonstration following the separately saved EPA protocol.

Primary: site-level recurrence across FRM/FEM and non-FRM/FEM reporting classes.
Secondary: original versus EPA-corrected Teledyne processing of the same stream.
No measurement is treated as an error-free reference or as independent by fiat.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import urllib.request
import zipfile

import numpy as np
import pandas as pd

BASE=Path.cwd()
from observation_swap import swap_test
from observation_swap.core import _estimates

DATA=BASE/'epa_airdata'
OUT=BASE/'epa_swap_output'
COLS=['State Code','County Code','Site Num','Parameter Code','POC','Method Code',
      'Sample Duration','Date Local','Units of Measure','Event Type',
      'Observation Percent','Arithmetic Mean','Latitude','Longitude']
ID=['site','Parameter Code','POC','Method Code','Sample Duration']


def load():
    DATA.mkdir(parents=True,exist_ok=True)
    pieces=[];manifest=[]
    for code,year in itertools.product([88101,88502],[2021,2022]):
        path=DATA/f'daily_{code}_{year}.zip'
        url='https://aqs.epa.gov/aqsweb/airdata/'+path.name
        if not path.exists():
            with urllib.request.urlopen(url,timeout=120) as r:path.write_bytes(r.read())
        raw=path.read_bytes()
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            with z.open(z.namelist()[0]) as f:
                x=pd.read_csv(f,usecols=COLS,low_memory=False)
        manifest.append(dict(file=path.name,url=url,bytes=len(raw),rows=len(x),
                             sha256=hashlib.sha256(raw).hexdigest()))
        pieces.append(x)
    return pd.concat(pieces,ignore_index=True),manifest


def prepare(raw,exclude_events=False):
    flow={'input_rows':len(raw)}
    x=raw[(raw['State Code']<=56)
          &raw['Sample Duration'].isin(['24 HOUR','24-HR BLK AVG'])
          &raw['Units of Measure'].eq('Micrograms/cubic meter (LC)')
          &raw['Arithmetic Mean'].notna()
          &raw['Observation Percent'].ge(75)
          &raw['Event Type'].ne('Excluded')].copy()
    if exclude_events:x=x[x['Event Type'].isna()].copy()
    flow['after_measurement_filters']=len(x)
    x['site']=(x['State Code'].astype(int).astype(str).str.zfill(2)+'-'
               +x['County Code'].astype(int).astype(str).str.zfill(3)+'-'
               +x['Site Num'].astype(int).astype(str).str.zfill(4))
    # Copies under different regulatory standards must agree; never average
    # disagreeing daily summaries or accidentally blend streams.
    keys=ID+['Date Local']
    counts=x.groupby(keys,observed=True)['Arithmetic Mean'].agg(['min','max'])
    bad=counts[counts['max']-counts['min']>1e-10].index
    ix=pd.MultiIndex.from_frame(x[keys])
    flow['ambiguous_monitor_dates_excluded']=len(bad)
    x=x[~ix.isin(bad)].drop_duplicates(keys).copy()
    flow['deduplicated_monitor_dates']=len(x)
    x['date']=pd.to_datetime(x['Date Local'])
    x['window']=x.date.dt.to_period('Q').astype(str)
    return x,flow


def select_streams(x):
    counts=x.groupby(ID,observed=True).size().rename('n_days').reset_index()
    counts=counts.sort_values(['site','Parameter Code','n_days','POC','Method Code','Sample Duration'],
                             ascending=[True,True,False,True,True,True])
    chosen=counts.drop_duplicates(['site','Parameter Code'])
    return x.merge(chosen[ID],on=ID,validate='many_to_one'),chosen


def panel_from_pair(paired):
    g=paired.groupby(['site','window'],observed=True)
    agg=g.agg(a=('a','mean'),b=('b','mean'),n=('date','size')).reset_index()
    long=agg.melt(id_vars=['site','window','n'],value_vars=['a','b'],
                  var_name='process',value_name='value')
    return long


def state_bootstrap(panel,replicates,seed=20261019):
    wide=panel[panel.n>=10].pivot(index='site',columns=['window','process'],values='value')
    windows=sorted(panel.window.unique())
    arrays=[]
    for w1,w2 in itertools.combinations(windows,2):
        sub=wide[[(w1,'a'),(w1,'b'),(w2,'a'),(w2,'b')]].dropna()
        if len(sub)>=30:arrays.append((sub.index,sub.to_numpy()))
    states=sorted({site[:2] for ids,_ in arrays for site in ids})
    rng=np.random.default_rng(seed);draws={k:[] for k in ['gap','covariance_gap']}
    for _ in range(replicates):
        chosen=rng.choice(states,len(states),replace=True)
        counts={s:int((chosen==s).sum()) for s in states}
        estimates=[]
        for ids,m in arrays:
            repeats=np.array([counts[s[:2]] for s in ids])
            sample=np.repeat(m,repeats,axis=0)
            if len(sample)<3:break
            estimates.append(_estimates(sample))
        if len(estimates)!=len(arrays):continue
        for k in draws:
            v=np.array([e[k] for e in estimates])
            if np.isfinite(v).all():draws[k].append(float(v.mean()))
    return {'n_states':len(states),'basis':'whole state clusters retaining site histories',
            'intervals':{k:{'lo':float(np.percentile(v,2.5)),
                            'hi':float(np.percentile(v,97.5)),
                            'n_draws':len(v)} for k,v in draws.items()}}


def run(panel,n_boot,name):
    panel.to_csv(OUT/f'{name}_panel.csv',index=False)
    result=swap_test(panel,unit='site',window='window',process='process',value='value',
                     weight='n',min_weight=10,n_boot=n_boot,min_units=30,
                     sparsity_split=False,pooled_bootstrap=True,seed=20261019)
    obj=result.to_dict()
    print(name,json.dumps(obj['summary']),flush=True)
    return obj


def main():
    global DATA, OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-boot',type=int,default=1000)
    parser.add_argument('--data-dir',type=Path,default=DATA)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args();DATA=args.data_dir;OUT=args.output_dir
    OUT.mkdir(parents=True,exist_ok=True)
    raw,manifest=load();x,flow=prepare(raw)
    # Keep updated calibration as instructed by EPA in the primary comparison.
    primary=x[~((x['Parameter Code']==88101)&x['Method Code'].isin([236,238]))]
    selected,streams=select_streams(primary)
    streams.to_csv(OUT/'selected_monitor_streams.csv',index=False)
    a=selected[selected['Parameter Code']==88101]
    b=selected[selected['Parameter Code']==88502]
    streams_both=streams[streams['Parameter Code']==88101].merge(
        streams[streams['Parameter Code']==88502],on='site',suffixes=('_a','_b'))
    physical_poc=streams_both.POC_a.where(
        ~streams_both['Method Code_a'].isin([736,738]),streams_both.POC_a-20)
    family=streams_both['Method Code_a'].replace({736:236,738:238,636:236,638:238})
    same_stream=(physical_poc==streams_both.POC_b)&(family==streams_both['Method Code_b'])
    excluded_sites=streams_both.loc[same_stream,'site']
    a=a[~a.site.isin(excluded_sites)];b=b[~b.site.isin(excluded_sites)]
    flow['potential_same_stream_sites_excluded']=excluded_sites.tolist()
    paired=a[['site','date','window','Arithmetic Mean']].rename(columns={'Arithmetic Mean':'a'}).merge(
        b[['site','date','Arithmetic Mean']].rename(columns={'Arithmetic Mean':'b'}),
        on=['site','date'],validate='one_to_one')
    panel=panel_from_pair(paired)
    flow.update(primary_candidate_sites=int(primary.site.nunique()),
                selected_streams=len(streams),sites_with_both_classes=int(paired.site.nunique()),
                matched_site_dates=len(paired),eligible_site_quarters=int((panel[panel.process=='a'].n>=10).sum()))
    results={'manifest':manifest,'flow':flow,
             'design':{'years':[2021,2022],'windows':'eight calendar quarters',
                       'unit':'AQS site','primary_processes':['88101','88502'],
                       'primary_outcome':'mean daily PM2.5 on common dates, micrograms/cubic metre LC',
                       'min_paired_dates':10,'bootstrap_replicates':args.n_boot},
             'primary':run(panel,args.n_boot,'primary')}
    results['primary_state_bootstrap']=state_bootstrap(panel,args.n_boot)
    good=panel[panel.n>=10].groupby('site').size()
    fixed_ids=good[good==16].index
    results['fixed_all_quarters']=run(panel[panel.site.isin(fixed_ids)],max(200,args.n_boot//2),'fixed_all_quarters')
    # Event exclusion is a support sensitivity on the same pre-selected streams.
    noevent,_=prepare(raw,exclude_events=True)
    noevent=noevent.merge(streams[ID],on=ID,validate='many_to_one')
    na=noevent[noevent['Parameter Code']==88101]
    nb=noevent[noevent['Parameter Code']==88502]
    npair=na[['site','date','window','Arithmetic Mean']].rename(columns={'Arithmetic Mean':'a'}).merge(
        nb[['site','date','Arithmetic Mean']].rename(columns={'Arithmetic Mean':'b'}),on=['site','date'],validate='one_to_one')
    results['no_event_dates']=run(panel_from_pair(npair),max(200,args.n_boot//2),'no_event_dates')
    # Pair original and corrected processing of the same physical monitor.
    cal=x[(x['Parameter Code']==88101)&x['Method Code'].isin([236,238,736,738])].copy()
    cal['family']=cal['Method Code'].replace({736:236,738:238})
    ck=['site','POC','Sample Duration','family']
    orig=cal[cal['Method Code']<700]
    corr=cal[cal['Method Code']>700].copy()
    # In these exports all 736/738 streams occupy POC 21..29; original
    # 236/238 streams occupy 1..9. Match the observed +20 coding convention,
    # together with site, method family, duration and exact date. This is a
    # stream linkage, not an independent adjudication of instrument serials.
    assert corr.POC.between(21,29).all()
    assert orig.POC.between(1,9).all()
    corr['POC']=corr.POC-20
    cp=orig[ck+['date','window','Arithmetic Mean']].rename(columns={'Arithmetic Mean':'a'}).merge(
        corr[ck+['date','Arithmetic Mean']].rename(columns={'Arithmetic Mean':'b'}),
        on=ck+['date'],validate='one_to_one')
    cc=cp.groupby(ck).size().rename('n_days').reset_index().sort_values(
        ['site','n_days','POC','family','Sample Duration'],ascending=[True,False,True,True,True]).drop_duplicates('site')
    cp=cp.merge(cc[ck],on=ck,validate='many_to_one')
    cpanel=panel_from_pair(cp)
    results['calibration_change']=run(cpanel,args.n_boot,'calibration_change')
    results['calibration_change_state_bootstrap']=state_bootstrap(cpanel,args.n_boot)
    cd=cp.assign(change=cp.a-cp.b).groupby('site').change.mean()
    results['calibration_change_support']={'sites':int(cp.site.nunique()),'matched_dates':len(cp),
        'site_mean_change_median':float(cd.median()),
        'site_mean_change_q25':float(cd.quantile(.25)),'site_mean_change_q75':float(cd.quantile(.75))}
    results['limits']=[
        'Secondary calibration comparison reprocesses the same observations; it is not independent instrumentation.',
        'Primary classes may share agency calibration, environmental exposure and reporting practice.',
        'Site identity is not independently adjudicated instrument continuity; replacement can retain a POC.',
        'Pair-specific eligibility selects a population; bootstrap intervals condition on that selection.',
        'State-cluster sensitivity does not capture all cross-state spatial dependence.',
        'No latent true concentration or causal attributable fraction is identified.',
    ]
    (OUT/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    print('Wrote',OUT/'results.json',flush=True)


if __name__=='__main__':main()
