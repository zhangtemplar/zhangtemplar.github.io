---
layout: post
title: Tradeoff
---

There are three high level trade-offs:

- Performance vs scalability
- Latency vs throughput
- Availability vs consistency

# Performance vs scalability

A way to look at performance vs scalability:


- If you have a performance problem, your system is slow for a single user.
- If you have a scalability problem, your system is fast for a single user but slow under heavy load.

Difficulties to make system scalable:

- scalability cannot be an after-thought. It requires applications and platforms to be designed with scaling in mind
- Heterogeneity means that some nodes will be able to process faster or store more data than other nodes in a system and algorithms that rely on uniformity either break down under these conditions or underutilize the newer resources.

# Latency vs throughput

Generally, you should aim for maximal throughput with acceptable latency:
- **Latency** is the time to perform some action or to produce some result.
- **Throughput** is the number of such actions or results per unit of time.

# Availability vs consistency

In a distributed computer system, you can only support two of the following guarantees:

- **Consistency** - Every read receives the most recent write or an error
- **Availability** - Every request receives a response, without guarantee that it contains the most recent version of the information
- **Partition Tolerance** - The system continues to operate despite arbitrary partitioning due to network failures

**Networks aren't reliable, so you'll need to support partition tolerance. You'll need to make a software tradeoff between consistency and availability.**

## CP - consistency and partition tolerance

Waiting for a response from the partitioned node might result in a timeout error. CP is a good choice if your business needs require atomic reads and writes.

## AP - availability and partition tolerance

Responses return the most recent version of the data available on a node, which might not be the latest. Writes might take some time to propagate when the partition is resolved.

AP is a good choice if the business needs allow for eventual consistency or when the system needs to continue working despite external errors.

# Consistency patterns

## Weak consistency

After a write, reads may or may not see it. A best effort approach is taken.

This approach is seen in systems such as memcached. Weak consistency works well in real time use cases such as VoIP, video chat, and realtime multiplayer games. 

## Eventual consistency

After a write, reads will eventually see it (typically within milliseconds). Data is replicated asynchronously.

This approach is seen in systems such as DNS and email. Eventual consistency works well in highly available systems.

## Strong consistency

After a write, reads will see it. Data is replicated synchronously.

This approach is seen in file systems and RDBMSes. Strong consistency works well in systems that need transactions.

# Availability patterns

There are two main patterns to support high availability: fail-over and replication.

## Fail-over

### Active-passive

With active-passive fail-over (or master-slave failover), heartbeats are sent between the active and the passive server on standby. If the heartbeat is interrupted, the passive server takes over the active's IP address and resumes service.

### Active-active

In active-active (master-master failover), both servers are managing traffic, spreading the load between them.

DNS (for public) or load-balancer can be used to control the traffic to which active.


### Disadvantage

- Fail-over adds more hardware and additional complexity.
- There is a potential for loss of data if the active system fails before any newly written data can be replicated to the passive.

## Replication

- Master-slave
- master-master

