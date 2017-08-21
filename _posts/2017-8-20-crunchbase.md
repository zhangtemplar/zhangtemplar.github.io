---
layout: post
title: Accessing Crunchbase Dataset
---

[Crunchbase](https://data.crunchbase.com/docs) has provdes datasets/api for user to utilize the database in Crunchbase. The data are provided in the following four ways. Assume `$user_key` is the the user key you obtained from Crunchbase.

# Open Data Map

[Open Data Map](https://data.crunchbase.com/docs/open-data-map) is freely available for all registered user, which only 
contains the basic profile information of organizations and persons.

The data is in csv format and tar and compressed. You can access the data to the following ways:

```
wget https://api.crunchbase.com/v/3/odm/odm.csv.tar.gz?user_key=%user_key --no-check-certificate
tar -xzvf odm.csv.tar.gz
```

You can read the database via pandas. There are two files included `organizations.csv` for profile information of the organization, e.g., companies, and `people.csv`.

```
import pandas as pd
df = pd.read_csv("organizations.csv")
```

# 2013 Snapshot

[2013 Snapshot](https://data.crunchbase.com/docs/2013-snapshot) contains a snapshot of full database of crunchbase at 
the time of 2013. It is also free of use.

Similarly you can obtain the data as:

```
wget https://api.crunchbase.com/v/3/snapshot/crunchbase_2013_snapshot_mysql.tar.gz?user_key=e1946f8ab01472d6b7166091e791be02 --no-check-certificate
tar -xzvf crunchbase_2013_snapshot_mysql.tar.gz
```

It contains multiple files dumped by `mysql`. You could either import those dataset into `mysql` or other dataset. For example, you can use the following code to for `sqlite3`:

```
wget https://gist.githubusercontent.com/zhangtemplar/128028ad62eca273cb2b098748169d83/raw/7155da4476002c9907a0af159bda3f54cdc9c5db/mysql2sqlite.sh
chmod +x mysql2sqlite.sh
for entry in `ls crunchbase_2013_snapshot_20131212/*.sql`; do 
  echo $entry; 
  ./mysql2sqlite.sh $entry | sqlite3 crunchbase.sqlite; 
done
```

Then you can also access it via `Pandas`:

```
import pandas as pd
import sqlite3
con = sqlite3.connect("crunchbase.sqlite")
con.text_factory = bytes
df = pd.read_sql("SELECT * FROM cb_objects", con)
```

# Daily CSV Export

[Daily CSV Export](https://data.crunchbase.com/docs/daily-csv-export) contains the daily updated information, which is 
only available to enterprise user.

# Rest Api

[Rest Api](https://data.crunchbase.com/docs/using-the-api) which is a sets of rest apis for accessing the crunchbase, 
which is only available to enterprise or application user.
