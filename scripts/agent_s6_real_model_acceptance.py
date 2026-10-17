"""S6固定模拟订单评测：仅请求用户授权的本机项目B服务，默认预览不发请求。"""

import argparse
import asyncio
from contextlib import closing
from datetime import datetime, timezone
from getpass import getpass
from html.parser import HTMLParser
import json
from pathlib import Path
import sqlite3
import ssl
import sys
import time
from urllib.parse import urljoin,urlsplit
from uuid import uuid4

import httpx

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from evaluation.agent_runtime.s6_cases import templates,judge,percentile


def simulated_cases():
    """执行前只读核对S6生成器的固定样本，拒绝原账号的其他订单或真实业务数据。"""
    folder=ROOT/'.skillflow/s6-fixtures'
    data=json.loads((folder/'s6-cases.json').read_text(encoding='utf-8'))
    fixture=json.loads((folder/'orders.json').read_text(encoding='utf-8'))
    if data.get('version')!='s6-cases-v1' or len(data['cases'])!=16:
        raise ValueError('S6固定模拟清单不符合约定')
    database=ROOT/'.skillflow/skillflow.sqlite3'
    # 只结束事务不会释放Windows文件句柄；核对样本后必须关闭连接，避免干扰容器WAL。
    with closing(sqlite3.connect(database.as_uri()+'?mode=ro',uri=True)) as db:
        for case in data['cases']:
            row=fixture[case['order_id']]
            # 商品、金额、归属必须是S6生成器的固定模拟值；不批量读取买家的原有订单。
            if row['sku'] not in {'S6-PB','S6-EAR-BLACK','S6-EAR-WHITE'} or row['paid_cents'] not in {9900,18900}:
                raise ValueError('拒绝非S6固定模拟商品')
            actual=db.execute('SELECT buyer_id,paid_cents FROM business_orders WHERE order_id=?',(case['order_id'],)).fetchone()
            if actual!=(case['buyer_id'],row['paid_cents']): raise ValueError('模拟记录归属或金额不匹配')
        if db.execute("SELECT 1 FROM business_orders WHERE order_id='A69999999'").fetchone():
            raise ValueError('预设不存在单号已被占用')
    return data


class Forms(HTMLParser):
    """只解析已授权本机Keycloak表单；不把密码、授权码、Cookie写入报告。"""
    def __init__(self): super().__init__();self.forms=[];self.current=None
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='form': self.current={'action':a.get('action',''),'fields':{}};self.forms.append(self.current)
        if tag=='input' and self.current is not None and a.get('name'):
            if a.get('type') not in {'submit','password'}: self.current['fields'][a['name']]=a.get('value','')
    def handle_endtag(self,tag):
        if tag=='form': self.current=None


def local_url(base,action):
    """禁止把现有登录凭据发到外部网站；只允许原本机IdP与8001回调。"""
    value=urljoin(base,action);u=urlsplit(value)
    if u.scheme!='https' or u.hostname!='localhost' or u.port not in {None,443,8001} or u.username or u.password:
        raise RuntimeError('拒绝非本机授权登录地址')
    return value


async def login(client,username,password):
    """正常OIDC授权码登录；不增权限、不生成旁路凭证、不修改现有账号。"""
    r=await local_request(client,'GET','https://localhost:8001/auth/login')
    credentials_sent=False
    for _ in range(4):
        page=Forms();page.feed(r.text)
        if not page.forms: break
        form=next((f for f in page.forms if 'username' in f['fields']),page.forms[0])
        fields=form['fields']
        if 'username' in fields:
            # 错误密码不自动重试，避免锁定现有账号；额外认证要求交由用户处理。
            if credentials_sent: raise RuntimeError('原账号登录未通过，需要核对凭据')
            credentials_sent=True
            fields={**fields,'username':username,'password':password}
        target=local_url(str(r.url),form['action'])
        # 密码只交给原IdP，授权码只交给固定回调，地址不进入验收报告。
        destination=urlsplit(target)
        if 'password' in fields and (destination.port not in {None,443} or not destination.path.startswith('/realms/myrealm/')):
            raise RuntimeError('拒绝非原IdP密码表单')
        if 'code' in fields and (destination.port!=8001 or destination.path!='/auth/callback'):
            raise RuntimeError('拒绝非原应用授权回调')
        r=await local_request(client,'POST',target,data=fields)
        if urlsplit(str(r.url)).port==8001 and urlsplit(str(r.url)).path=='/': break
    me=await client.get('/auth/me');me.raise_for_status()
    if not me.json().get('runtime',{}).get('agent_s6_enabled'): raise RuntimeError('目标不是S6')
    return me.json()['buyer_id']


async def local_request(client,method,url,data=None):
    """逐跳验证后才发送请求，避免自动重定向在地址校验前带走表单内容。"""
    for _ in range(10):
        url=local_url('https://localhost:8001',url)
        response=await client.request(method,url,data=data,follow_redirects=False)
        if response.status_code not in {301,302,303,307,308}: return response
        target=local_url(str(response.url),response.headers['location'])
        if data and response.status_code in {307,308} and urlsplit(target).netloc!=urlsplit(url).netloc:
            raise RuntimeError('拒绝跨源重定向表单内容')
        if response.status_code==303 or (response.status_code in {301,302} and method=='POST'):
            method,data='GET',None
        url=target
    raise RuntimeError('本机登录重定向次数超限')


def save_report(report,path):
    """原始JSON与可读摘要一起保存；缺样本、异常轮次均不能判为通过。"""
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    # Windows索引/杀毒程序可能短暂占用旧文件；只重试本地报告替换，不重试业务请求。
    for attempt in range(4):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt==3: raise
            time.sleep(.01*(attempt+1))
    lines=['# S6 真实模型容量记录','',f"状态：{report['state']}；正式达标：{report.get('passed',False)}。",
        f"已发容量业务请求：{report['requests_sent']}/120；容量内观察到模型调用：{report['model_requests_observed']}。",'',
        '数据仅限S6固定模拟订单；两名买家，最多四会话。延迟含排队；缺样本不能判为通过。','',
        '| 会话 | 类型 | 已完成样本 | 正确 | 全请求P95/秒 | 正确请求P95/秒 | 达标 |',
        '|---|---|---:|---:|---:|---:|---|']
    for level in report['levels']:
        for category,summary in level.get('summary',{}).items():
            lines.append(f"| {level['level']} | {category} | {summary['count']}/20 | {summary['correct']} | "
                f"{summary['all_p95']} | {summary['correct_p95']} | {summary['passed']} |")
    if report.get('error_type'): lines.extend(['',f"中断原因类别：{report['error_type']}；不保存登录凭据或异常中的敏感地址。"])
    if 'joint_check' in report:
        joint=report['joint_check']
        lines.extend(['',f"联合功能：{joint.get('state')}；通过：{joint.get('passed',False)}；"
            f"只读业务样本：{len(joint.get('queries',[]))}/6；跨买家拒绝：{len(joint.get('denials',[]))}/12。",
            f"联合功能已发业务请求：{joint.get('requests_sent',0)}；观察到模型调用：{joint.get('model_requests_observed',0)}。",
            '联合功能请求单独记录，不计入正式120次容量或其延迟。'])
    path.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def headers(client):
    """CSRF值只发给原本机服务，不导出认证Cookie，不进入任务清单或报告。"""
    return {'X-CSRF-Token':client.cookies.get('__Host-skillflow-csrf','')}


async def run(args,passwords):
    """固定120个只读模拟任务；模型目的地仍由B已授权的服务配置决定，脚本不接收外部URL。"""
    manifest=simulated_cases()
    ca=ssl.create_default_context(cafile=str(args.ca_file))
    clients=[httpx.AsyncClient(base_url='https://localhost:8001',verify=ca,trust_env=False,timeout=150) for _ in range(2)]
    report={'schema':'s6-real-model-v2','recorded_at':datetime.now(timezone.utc).isoformat(),
        'model_called':False,'requests_sent':0,'model_requests_observed':0,
        'data':'S6_FIXED_SIMULATED_ONLY','authentication':'real_oidc_authorization_code',
        'templates':templates(),'levels':[],'state':'authenticating','passed':False}
    out=ROOT/'evaluation/agent_runtime/s6';out.mkdir(parents=True,exist_ok=True)
    path=out/('real-model-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid4().hex[:6]+'.json')
    def save():
        """联合阶段也如实更新模型是否已调用，容量调用数量继续单列，不混入120次。"""
        joint=report.get('joint_check',{})
        if joint.get('model_called',False) is not False:
            report['model_called']=joint['model_called']
        save_report(report,path)
    save()
    try:
        buyers=[await login(c,u,p) for c,u,p in zip(clients,args.users,passwords)]
        if len(set(buyers))!=2 or set(buyers)!=set(manifest['buyers']): raise RuntimeError('登录买家不是S6样本买家')
        if getattr(args,'joint_preflight',False) or getattr(args,'joint_only',False):
            from evaluation.agent_runtime.s6_joint import check_joint
            # 按原清单买家顺序执行Alice四轮与Bob查询，不依赖用户传入账号的排列顺序。
            ordered=[clients[buyers.index(buyer)] for buyer in manifest['buyers']]
            report.update(state='joint_check',joint_check={})
            save()
            passed=await check_joint(ordered,manifest['buyers'],manifest,report['joint_check'],save)
            if not passed:
                report['state']='joint_check_failed';save();return 1
            if getattr(args,'joint_only',False):
                report['state']='joint_check_complete';save();return 0
        lookup={(r['buyer_id'],r['kind']):r['order_id'] for r in manifest['cases']}
        report.update(buyers=buyers,state='running')
        save()
        for level in [1,2,4]:
            workers=[]
            for i in range(level):
                person=i%2;client=clients[person]
                response=await client.post('/sessions',headers=headers(client));response.raise_for_status()
                workers.append((client,buyers[person],response.json()['session_id']))
            rows=[];unsafe=asyncio.Event();started=time.perf_counter()
            # 公共任务队列避免静态切片把全部简单任务分给Alice、多工具任务分给Bob。
            pending=asyncio.Queue()
            for case in templates(): pending.put_nowait(case)
            async def worker(index,item):
                client,buyer,session=item
                while not pending.empty():
                    if unsafe.is_set(): break
                    case=pending.get_nowait()
                    oid='A69999999' if case['kind']=='missing' else lookup[(buyer,case['kind'])]
                    row={'case_id':case['id'],'category':case['category'],'buyer_id':buyer,'session_id':session,
                         'order_id':oid,'message':case['prompt'].format(oid=oid)}
                    t=time.perf_counter()
                    try:
                        report['requests_sent']+=1
                        # 发出后超时也可能已调用模型；未知不能登记成“没有调用”。
                        if report['model_called'] is False: report['model_called']=None
                        save()
                        # 文件持久化耗时不计入HTTP响应；服务端排队仍在HTTP等待范围内。
                        t=time.perf_counter()
                        response=await client.post('/chat',json={'session_id':session,'message':row['message']},headers=headers(client))
                        row['http_status']=response.status_code;row['elapsed_seconds']=time.perf_counter()-t
                        if response.status_code==200:
                            body=response.json();ok,issues,security=judge(case,body,buyer,oid)
                            consumed=(body.get('result',{}).get('budget') or {}).get('model_requests',0)
                            if consumed>0:
                                report['model_called']=True;report['model_requests_observed']+=consumed
                            row.update(correct=ok,issues=issues,security=security,response=body)
                            if security: unsafe.set()
                        else: row.update(correct=False,issues=['HTTP_ERROR'],security=[])
                    except (httpx.RequestError,ValueError,KeyError,TypeError) as e:
                        row.update(elapsed_seconds=time.perf_counter()-t,correct=False,issues=[type(e).__name__],security=[])
                    rows.append(row);save()
            data={'level':level,'rows':rows};report['levels'].append(data);save()
            jobs=[asyncio.create_task(worker(i,w)) for i,w in enumerate(workers)]
            try:
                await asyncio.gather(*jobs)
            except BaseException:
                # 意外故障或取消后先收束其他会话，防止关闭客户端后仍有后台请求和报告修改。
                for job in jobs: job.cancel()
                await asyncio.gather(*jobs,return_exceptions=True)
                raise
            data['wall_seconds']=time.perf_counter()-started;data['summary']={}
            for category,limit in [('simple',15),('multi',30)]:
                subset=[r for r in rows if r['category']==category];correct=[r for r in subset if r['correct']]
                times=[r['elapsed_seconds'] for r in subset];p95=percentile(times,95)
                data['summary'][category]={'count':len(subset),'correct':len(correct),
                    'correct_rate':len(correct)/len(subset) if subset else 0,
                    'all_p50':percentile(times,50),'all_p95':p95,'all_p99':percentile(times,99),
                    'correct_p95':percentile([r['elapsed_seconds'] for r in correct],95),
                    'passed':len(subset)==20 and len(correct)>=19 and p95<=limit}
            data['throughput_rps']=len(rows)/data['wall_seconds'];save()
            print('LEVEL='+str(level)+' '+json.dumps(data['summary'],ensure_ascii=False),flush=True)
            if unsafe.is_set(): report['state']='stopped_security_failure';break
        else: report['state']='complete'
        report['passed']=report['state']=='complete' and all(s['passed'] for l in report['levels'] for s in l['summary'].values())
        save()
        return 0 if report['passed'] else 1
    except BaseException as exc:
        # 不记录异常文本中的授权地址、Cookie等内容；取消与故障均留未完成报告。
        report.update(state='incomplete',passed=False,error_type=type(exc).__name__)
        raise
    finally:
        save();print('S6_REAL_MODEL_REPORT='+str(path),flush=True)
        for client in clients: await client.aclose()


def main():
    """默认预览无网络；执行时在终端输入原账号密码，不保存在磁盘或命令行参数中。"""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true');p.add_argument('--users',nargs=2,default=['b6-alice','b6-bob'])
    p.add_argument('--ca-file',type=Path,default=Path.home()/'AppData/Local/mkcert/rootCA.pem')
    p.add_argument('--joint-preflight',action='store_true',help='正常登录后先做联合功能与双向越权检查，通过后运行固定120次容量')
    p.add_argument('--joint-only',action='store_true',help='只做联合功能与双向越权检查，不发起正式120次容量')
    args=p.parse_args()
    if not args.execute:
        print(json.dumps({'levels':[] if args.joint_only else [1,2,4],'tasks_per_level':40,
                          'total':0 if args.joint_only else 120,'data':'S6_FIXED_SIMULATED_ONLY',
                          'joint_queries':6 if args.joint_preflight or args.joint_only else 0,
                          'templates':templates(),'joint_preflight':args.joint_preflight or args.joint_only,
                          'joint_only':args.joint_only},ensure_ascii=False,indent=2));return 0
    simulated_cases()
    passwords=[getpass(f'{user} 原账号密码（不回显）: ') for user in args.users]
    try: return asyncio.run(run(args,passwords))
    except Exception as e:
        print('验收未完成：'+type(e).__name__);return 2


if __name__=='__main__': raise SystemExit(main())
