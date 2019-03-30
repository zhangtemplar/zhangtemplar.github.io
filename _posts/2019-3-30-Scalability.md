---
layout: post
title: Scalability
---

# Introduction

This note is based on understanding from [system-design-primer](https://github.com/donnemartin/system-design-primer#step-2-review-the-scalability-article) and [www.lecloud.net](http://www.lecloud.net/tagged/scalability/chrono)

# Clones
The first golden rule for scalability: every server contains exactly the same codebase and does not store any user-related data. 

you can now create an image file from one of these servers.  Use this as a “super-clone” that all your new instances are based upon. Whenever you start a new instance/clone, just do an initial deployment of your latest code and you are ready!

# Database
## SQL Path

- do master-slave replication (read from slaves, write to master)
- upgrade your master server by adding RAM, RAM and more RAM. 
- "sharding”, “denormalization” and “SQL tuning”

## NoSQL Path

- denormalize right from the beginning and include no more Joins in any database query
- stay with MySQL, and use it like a NoSQL database, or you can switch to a better and easier to scale NoSQL database like MongoDB or CouchDB
- Joins will now need to be done in your application code

# Cache

Always mean in-memory caches like Memcached or Redis. Please never do file-based caching

A cache is a simple key-value store and it should reside as a buffering layer between your application and your data storage.

- Whenever you do a query to your database, you store the result dataset in cache. A hashed version of your query is the cache key.
  - The main issue is the expiration. It is hard to delete a cached result when you cache a complex query
  - When one piece of data changes (for example a table cell) you need to delete all cached queries who may include that table cell.
- [Recommended] Let your class assemble a dataset from your database and then store the complete instance of the class or the assembed dataset in the cache. Some ideas of objects to cache:
  - user sessions (never use the database!)
  - fully rendered blog articles
  - activity streams
  - user<->friend relationships
  
# Asynchronism

If you do something time-consuming, try to do it always asynchronously. There are two ways:
- doing the time-consuming work in advance (e.g., training/updating a model) and serving the finished work with a low request time.
- for some time consuming work which does depends on user's input, the frontend of your website sends a job onto a job queue and immediately signals back to the user: your job is in work, please continue to the browse the page.
  - The job queue is constantly checked by a bunch of workers for new jobs.
  - The frontend, which constantly checks for new “job is done” - signals, sees that the job was done and informs the user about it.
  - RabbitMQ is one of many systems which help to implement async processing.
  
  
# Two Types of Scale

- Horizontal scale: Scaling out using more commodity machines, which is more cost efficient and results in higher availability - Vertical scale: scaling up a single server on more expensive hardware

Disavatanges of horizontal scale:

- Scaling horizontally introduces complexity and involves cloning servers
- Downstream servers such as caches and databases need to handle more simultaneous connections as upstream servers scale out
