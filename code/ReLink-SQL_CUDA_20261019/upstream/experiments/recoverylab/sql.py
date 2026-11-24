from pathlib import Path
from collections import Counter
import sqlite3,time,re,itertools

def connect(db):
 con=sqlite3.connect(Path(db).resolve().as_uri()+'?mode=ro',uri=True,timeout=2)
 con.execute('PRAGMA query_only=ON')
 return con

def extract_sql(text):
 blocks=re.findall(r'```(?:sql|sqlite)?\s*(.*?)```',text,re.S|re.I)
 text=blocks[0] if blocks else text
 text=text.strip()
 m=re.search(r'\b(SELECT|WITH)\b',text,re.I)
 return text[m.start():].strip() if m else text

def execute(db,sql,timeout=3,max_rows=50000):
 started=time.perf_counter()
 try:
  sql=extract_sql(sql)
  if not re.match(r'^(SELECT|WITH)\b',sql,re.I):raise ValueError('Only read-only SELECT/WITH is allowed')
  with connect(db) as con:
   allowed={sqlite3.SQLITE_SELECT,sqlite3.SQLITE_READ,sqlite3.SQLITE_FUNCTION,sqlite3.SQLITE_RECURSIVE}
   con.set_authorizer(lambda action,*args: sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY)
   con.set_progress_handler(lambda: int(time.perf_counter()-started>timeout),10000)
   cur=con.execute(sql);rows=cur.fetchmany(max_rows+1)
   if len(rows)>max_rows:raise ValueError('Result exceeds evaluation row limit')
   return {'status':'ok','columns':[x[0] for x in cur.description or []],'rows':rows,'seconds':time.perf_counter()-started,'sql':sql}
 except Exception as e:
  return {'status':'error','error':str(e),'seconds':time.perf_counter()-started,'sql':sql}

def normalize_value(x):
 if isinstance(x,float):return round(x,6)
 if isinstance(x,bytes):return x.hex()
 return x

def result_equal(pred,gold,order_matters=False):
 """Single-database result equality, preserving duplicates; not Spider TSA."""
 if pred['status']!='ok' or gold['status']!='ok':return False
 a=[tuple(map(normalize_value,r)) for r in pred['rows']]
 b=[tuple(map(normalize_value,r)) for r in gold['rows']]
 if len(a)!=len(b) or len(pred['columns'])!=len(gold['columns']):return False
 if not a:return True
 n=len(a[0])
 # Allow projection column permutations as common SQL execution evaluators do.
 permutations=itertools.permutations(range(n)) if n<=5 else [tuple(range(n))]
 for order in permutations:
  transformed=[tuple(r[i] for i in order) for r in a]
  if (transformed==b if order_matters else Counter(transformed)==Counter(b)):return True
 return False

def schema(db):
 lines=[]
 with connect(db) as con:
  names=[x[0] for x in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
  for name in names:
   quoted='"'+name.replace('"','""')+'"'
   columns=list(con.execute(f'PRAGMA table_info({quoted})'))
   lines.append(name+'('+', '.join(x[1]+':'+(x[2] or 'TEXT') for x in columns)+')')
   for fk in con.execute(f'PRAGMA foreign_key_list({quoted})'):lines.append(f'  FK {name}.{fk[3]} -> {fk[2]}.{fk[4]}')
 return '\n'.join(lines)

def probe(db,question,prediction,max_chars=1800):
 """Read a bounded sample of real values. Never consult gold SQL or labels."""
 lines=[]
 with connect(db) as con:
  names=[x[0] for x in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
  text=(question+' '+prediction).lower()
  ranked=sorted(names,key=lambda n:(-(5*(n.lower() in text)+sum(w in text for w in n.lower().split('_'))),n))
  for name in ranked[:3]:
   quoted='"'+name.replace('"','""')+'"'
   cur=con.execute(f'SELECT * FROM {quoted} LIMIT 2')
   columns=[x[0] for x in cur.description]
   rows=[[str(v)[:90] for v in r] for r in cur.fetchall()]
   lines.append(f'{name} columns={columns}; sample_rows={rows}')
 return '\n'.join(lines)[:max_chars]
