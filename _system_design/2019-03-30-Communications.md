---
layout: post
title: Communications
---

Different components needs to communicate with each other. Here is a simple comparison of those protocols.

- UDP and TCP are both transport layer protocols. TCP is reliable and connection-based. UDP is connectionless and unreliable.
- HTTP is in the application layer and normally TCP based, since HTTP assumes a reliable transport.
- RPC, an application layer protocol, is an inter-process communication that allows a computer program to cause a subroutine or procedure to execute in another address space (commonly on another computer on a shared network), without the programmer explicitly coding the details for this remote interaction. That is, the programmer writes essentially the same code whether the subroutine is local to the executing program, or remote. In an Object-Oriented Programming context, RPC is also called remote invocation or remote method invocation (RMI).

![Source: OSI 7 layer model](https://camo.githubusercontent.com/1d761d5688d28ce1fb12a0f1c8191bca96eece4c/687474703a2f2f692e696d6775722e636f6d2f354b656f6351732e6a7067)

# Remote procedure call (RPC)

![Source: Crack the system design interview](https://camo.githubusercontent.com/1a3d7771c0b0a7816d0533fffeb6eeeb442d9945/687474703a2f2f692e696d6775722e636f6d2f6946344d6b62352e706e67)

A local procedure that marshals the procedure identifier and the arguments into a request message, and then to send via its communication module to the server. When the reply message arrives, it unmarshals the results.

We do not have to implement our own RPC protocols. There are off-the-shelf frameworks.

- Google Protobuf: an open source RPC with only APIs but no RPC implementations. Smaller serialized data and slightly faster. Better documentations and cleaner APIs.
- Facebook Thrift: supports more languages, richer data structures: list, set, map, etc. that Protobuf does not support) Incomplete documentation and hard to find good examples.
User case: Hbase/Cassandra/Hypertable/Scrib/…
- Apache Avro: Avro is heavily used in the hadoop ecosystem and based on dynamic schemas in Json. It features dynamic typing, untagged data, and no manually-assigned field IDs.

# REST API

Generally speaking, RPC is internally used by many tech companies for performance issues, but it is rather hard to debug and not flexible. So for public APIs, we tend to use HTTP APIs, and are usually following the RESTful (Representational state transfer of resources) style.

Best practice of HTTP API to interact with resources.
- URL only decides the location. Headers (Accept and Content-Type, etc.) decide the representation. HTTP methods(GET/POST/PUT/DELETE) decide the state transfer.
- minimize the coupling between client and server (a huge number of HTTP infras on various clients, data-marshalling).
- stateless (which is very difficult) and scaling out.
- service partitioning feasible.
- used for public API.
