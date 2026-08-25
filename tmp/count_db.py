import sqlite3
p=r"D:\我的项目\揭榜挂帅\ai_agent-trainer\data\sqlite_data\project.db"
c=sqlite3.connect(p)
tables=[r[0] for r in c.execute("select name from sqlite_master where type='table' order by name")]
print('TABLES',len(tables))
for table in tables:
    try:
        count=c.execute('select count(*) from "'+table.replace('"','""')+'"').fetchone()[0]
        print(f'{table}: {count}')
    except Exception as exc:
        print(f'{table}: ERROR {exc}')
