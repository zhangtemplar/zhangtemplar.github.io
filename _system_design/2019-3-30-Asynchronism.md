---
layout: post
title: Asynchronism
---

Asynchronous workflows help reduce request times for expensive operations. They can also help by doing time-consuming work in advance, such as periodic aggregation of data.

# Message queues

If an operation is too slow to perform inline, you can use a message queue:
- An application publishes a job to the queue, then notifies the user of job status
- A worker picks up the job from the queue, processes it, then signals the job is complete

The user is not blocked and the job is processed in the background. During this time, the client might optionally do a small amount of processing to make it seem like the task has completed.

Example:
- [redis](https://redis.io/)
- [RabbitMQ](https://www.rabbitmq.com/)
- [Amazon SQS](https://aws.amazon.com/sqs/)

# Task queues

Tasks queues receive tasks and their related data, runs them, then delivers their results. They can support scheduling and can be used to run computationally-intensive jobs in the background. Example: [Celery](http://www.celeryproject.org/)

# Back pressure

If queues start to grow significantly, the queue size can become larger than memory, resulting in cache misses, disk reads, and even slower performance. Back pressure can help by limiting the queue size, thereby maintaining a high throughput rate and good response times for jobs already in the queue. Once the queue fills up, clients get a server busy or HTTP 503 status code to try again later.
