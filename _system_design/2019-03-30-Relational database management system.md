---
layout: post
title: Relational database management system
---

A relational database like SQL is a collection of data items organized in tables.

![Source: Scaling up to your first 10 million users](https://camo.githubusercontent.com/15a7553727e6da98d0de5e9ca3792f6d2b5e92d4/687474703a2f2f692e696d6775722e636f6d2f586b6d3543587a2e706e67)

# ACID

ACID is a set of properties of relational database transactions.

- Atomicity - Each transaction is all or nothing
- Consistency - Any transaction will bring the database from one valid state to another
- Isolation - Executing transactions concurrently has the same results as if the transactions were executed serially
- Durability - Once a transaction has been committed, it will remain so

# Scale the RDBMS

There are many techniques to scale a relational database.

## master-slave replication

The master serves reads and writes, replicating writes to one or more slaves, which serve only reads. Disadvatanges:
- Additional logic is needed to promote a slave to a master.
- Common to replication
  - There is a potential for loss of data if the master fails before any newly written data can be replicated to other nodes.
  - Writes are replayed to the read replicas. It can be a big load when there are many slaves

## master-master replication
Both masters serve reads and writes and coordinate with each other on writes. Disadvantanges:
- You'll need a load balancer or you'll need to make changes to your application logic to determine where to write.
- Most master-master systems are either loosely consistent (violating ACID) or have increased write latency due to synchronization.
- Conflict resolution comes more into play as more write nodes are added and as latency increases.


## federation

Federation (or functional partitioning) splits up databases by function. 

![Source: Scaling up to your first 10 million users](https://camo.githubusercontent.com/6eb6570a8b6b4e1d52e3d7cc07e7959ea5dac75f/687474703a2f2f692e696d6775722e636f6d2f553371563333652e706e67)

Advatanges:
- less read and write traffic to each database and therefore less replication lag. 
- Smaller databases result in more data that can fit in memory, which in turn results in more cache hits due to improved cache locality. 
- With no single central master serializing writes you can write in parallel, increasing throughput.

Disadvantage:
- Federation is not effective if your schema requires huge functions or tables.
- You'll need to update your application logic to determine which database to read and write.
- Joining data from two databases is more complex with a server link.

## sharding

Sharding distributes data across different databases such that each database can only manage a subset of the data. It has similar advantanges as federation.

![Source: Scalability, availability, stability, patterns](https://camo.githubusercontent.com/1df78be67b749171569a0e11a51aa76b3b678d4f/687474703a2f2f692e696d6775722e636f6d2f775538783549642e706e67)

Disadvantage
- You'll need to update your application logic to work with shards, which could result in complex SQL queries.
- Data distribution can become lopsided in a shard. For example, a set of power users on a shard could result in increased load to that shard compared to others.
- Rebalancing adds additional complexity. A sharding function based on consistent hashing can reduce the amount of transferred data.
- Joining data from multiple shards is more complex.

## Denormalization

Denormalization attempts to improve read performance at the expense of some write performance. Redundant copies of the data are written in multiple tables to avoid expensive joins. Some RDBMS can automatically handels the work of storing redundant information and keeping redundant copies consistent.

In most systems, reads can heavily outnumber writes 100:1 or even 1000:1. A read resulting in a complex database join can be very expensive, spending a significant amount of time on disk operations.

Disadvantages:
- Data is duplicated.
- Constraints can help redundant copies of information stay in sync, which increases complexity of the database design.
- A denormalized database under heavy write load might perform worse than its normalized counterpart.


## SQL tuning

It's important to benchmark and profile to simulate and uncover bottlenecks.

- Benchmark - Simulate high-load situations with tools such as ab.
- Profile - Enable tools such as the slow query log to help track performance issues.

Common solutions:
* MySQL dumps to disk in contiguous blocks for fast access.
- Use CHAR instead of VARCHAR for fixed-length fields.
CHAR effectively allows for fast, random access, whereas with VARCHAR, you must find the end of a string before moving onto the next one.
- Use TEXT for large blocks of text such as blog posts. TEXT also allows for boolean searches. Using a TEXT field results in storing a pointer on disk that is used to locate the text block.
- Use INT for larger numbers up to 2^32 or 4 billion.
Use DECIMAL for currency to avoid floating point representation errors.
- Avoid storing large BLOBS, store the location of where to get the object instead.
- VARCHAR(255) is the largest number of characters that can be counted in an 8 bit number, often maximizing the use of a byte in some RDBMS.
- Set the NOT NULL constraint where applicable to improve search performance.

Good indices:
- Columns that you are querying (SELECT, GROUP BY, ORDER BY, JOIN) could be faster with indices.
- Indices are usually represented as self-balancing B-tree that keeps data sorted and allows searches, sequential access, insertions, and deletions in logarithmic time.
- Placing an index can keep the data in memory, requiring more space.
- Writes could also be slower since the index also needs to be updated.
- When loading large amounts of data, it might be faster to disable indices, load the data, then rebuild the indices.