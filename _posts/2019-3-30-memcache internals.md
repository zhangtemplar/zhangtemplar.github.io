---
layout: post
title: memcache
---

# Introduction

Memcache provides O(1) read and write operations. It has three core concepts.

Source: [memcache internals
](https://www.adayinthelifeof.nl/2011/02/06/memcache-internals/)

# LRU algorithm

Memcache uses **Least Recently Used** to decide which data to remove when out of space: it deletes the item that isn’t used for the longest period of time.

Internally, all objects have a “counter”. This counter holds a timestamp, which is updated whenever the object is updated or accessed.

# Memory allocation/Slab allocation

Memory gets fragmented easily after allocate space for object and free space. Memcache uses it own memory manager which will allocate the maximum amount of memory from the operating system that you have set (for instance, 64Mb, but probably more) through one malloc() call. From that point on, it will use its own memory manager system called the slab allocator.

When memcache starts, 
- it partitions its allocated memory into smaller parts called **pages**.
  - Each page is 1Mb large (coincidentally, the maximum size that an object can have you can store in memcache). 
- Each of those pages can be assigned to a **slab-class**, or can be unassigned (being a free page). 
  - Memcache will initially create 1 page per slab-class
- Each page that is designated to a particular slab-class will be divided into smaller parts called **chunks**.
  - The chunks in each slab have the same size
  - The smallest chunk-size starts at 80 bytes and increases with a factor of 1.25
- as soon as a complete page if full, it will fetch a new free page, assign it to the specified slab-class, partition it into chunks and gets the first available chunk to store the data
- But as soon as there are no more pages left that we can designate to our slab-class, it will use the LRU-algorithm to evict one of the existing chunks to make room

# Consistent hashing

Your web application can talk to multiple memcache-servers at the same time. As soon we add an object to the memcache, it will automatically choose a server where it can store the data. for each key that gets stored or fetched, it will create a hash. With consistent hashing, we do not have to worry (much) about keys changing servers when your server count goes up or down.