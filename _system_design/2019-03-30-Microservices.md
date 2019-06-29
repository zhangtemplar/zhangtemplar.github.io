---
layout: post
title: Microservices
---

Micro service architecture is an evolution of the SOA (Service Oriented Architecture) architecture, whereas SOA was focused on integration of various applications, Micro services architecture (MSA) aims to create small modular services which belong to a single application.

This article is based on reading of [msa-getting-started](https://cloudncode.blog/2016/07/22/msa-getting-started/)


# Benefits

- high speed computing (lowered latencies) and the continuously growing need of continuous deployment and integration.
- now independent services can be written and re-written in different languages which is best suited for the service.

# Issues and Resolutions

- Nano-services: services are so small that the overhead and complexities of so many services begin to outweigh the benefits
- Its hard to keep track of the service URL’s
  - "Service Discovery” is the soltuion. In this concept services which start up register themselves with a central authority with some basic parameters such as environment & application name
- discovering services, the available API’s & writing & maintaining clients: a lot of services. If we don’t document these services well and let’s not kid ourselves we won’t, numerous issues arise over time
  - "Open API" would be the solution
- Too many deployment jobs: use the CI/CD tools, e.g., Jenkins
- One service connection fails, everything fails. Soltutions would be:
  - retry
  - Circuit-breaker pattern: the code keeps a counter of what is happening to any particular service and if a threshold number of errors have been continuously returned by the service, it fails fast all the subsequent requests to the same service for a predetermined amount of time 
- Database isolation: this implies that their storage technologies are self contained as well. There are three approaches to tackle this problem:
  - Database per service
  - Schema per service
  - Tables per service
- Every service has to support all the transport protocols & authentication
- Too many servers, too much cost. Solution:
  - employ small machines (smaller the better) and just add more hardware as traffic starts rolling in and remove as it goes out, staying as close to the actual requirement as possible.
  - Docker
  - K8S