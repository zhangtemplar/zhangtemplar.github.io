---
layout: post
title: MongoDB Tips
---
 
# add user
using python

```
client.admin.add_user('zhangtemplar', 'tomato7G2262', roles=[ { 'role': "userAdminAnyDatabase", 'db': "admin" } ])
```

or using mongo shell
```db.createUser({user: 'zhangtemplar', pwd: 'tomato7G2262', roles: ['root']})
```

# copy database

```
client2.admin.command('copydb', fromdb='test', todb='test', fromhost='url:port')
```
Or you can use `mongodump` and `mongorestore` as described below.

# merge collection/database

There is not command letting your merge collection or database, thus you need to iteration all the collections and all their documents to merge:

```
src = client['test']
dst = client['test2']
for name in src.collection_names():
    for doc in src[name].find():
        result = dst[name].replace_one({'_id' : doc['_id']}, doc, upsert=True)
        if result.modified_count > 0:
            print 'updated', doc['_id']
        else:
            print 'inserted', doc['_id']
    print 'finish', name
```

# Authentication

In python with `pymongo`:
```
MongoClient('mongodb://user:' + password + '@127.0.0.1')
```

# Search with regular expression

In python with `pymongo`:
```
import re
regx = re.compile("^foo", re.IGNORECASE)
db.users.find_one({"files": regx})
```

# Start a mongodb in docker

```
docker run -p 27017:27017 --name mongo -v /home/ubuntu/mongo/data/db/:/data/db -d mongo:latest --auth
```
More specially it map ports `27017` to `27017`, name the container as `mongo`, map the local volume `/home/ubuntu/mongo/data/db/` to `/data/db` and enable authorization.

# Restore
```
mongorestore -u $username -p $password --authenticationDatabase=admin --gzip --archive=$backup
```
It will restore the database from a gzip archive called `$backup`. If you don't have authentication, you can skip `-u $username -p $password --authenticationDatabase=admin`. It will restore all the database but will not overwrite any existing documents. To replace the whole database, please add argument `-d`. To restore a specific database, please use `-database $database`.

# Backup

To back up all the database:
```
mongodump --gzip --archive=mongodb_08012017192000.gz
```

# Use Python Eve to Provide Rest Interface for MongoDB

call the api as:
```
curl -i -g http://url:port/user?where={"id":"some-id"}
```
Note `url` and `port` are for eve.

# start python-eve
```
docker run --detach --volume /home/ubuntu/eve/src:/src --name python-eve -p 5000:5000 zhangtemplar/docker-python-eve:3.5
```

# pymongo join
```
pipeline = [{"$lookup": {"from":"user_keyword", "localField":"id","foreignField":"userId","as":"keyword" }}, {'$match': {'id' : 'https://angel.co/paul-judge-1'}}]
result = client['angellist']['user'].aggregate(pipeline)
```
