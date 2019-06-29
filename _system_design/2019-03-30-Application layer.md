---
layout: post
title: Application layer
---

Separating out the web layer from the application layer (also known as platform layer) allows you to scale and configure both layers independently. The single responsibility principle advocates for small and autonomous services that work together. Small teams with small services can plan more aggressively for rapid growth.

![Source: Intro to architecting systems for scale](https://camo.githubusercontent.com/feeb549c5b6e94f65c613635f7166dc26e0c7de7/687474703a2f2f692e696d6775722e636f6d2f7942355359776d2e706e67)

# Microservices

A suite of independently deployable, small, modular services. Each service runs a unique process and communicates through a well-defined, lightweight mechanism to serve a business goal. 

# Service Discovery

- Systems such as Consul, Etcd, and Zookeeper can help services find each other by keeping track of registered names, addresses, and ports. 
- Health checks help verify service integrity and are often done using an HTTP endpoint. 
- Both Consul and Etcd have a built in key-value store that can be useful for storing config values and other shared data.

# Disadvantage

- Adding an application layer with loosely coupled services requires a different approach from an architectural, operations, and process viewpoint (vs a monolithic system).
- Microservices can add complexity in terms of deployments and operations.
