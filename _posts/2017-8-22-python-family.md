---
layout: post
title: How big is the Python Family
---

This article is original from [How big is the Python Family](https://py.checkio.org/blog/how-big-is-the-python-family/)

![python](https://static.checkio.org/media/blog/share/python_family.jpg)

As we all know Python is a high-level programming language widely used for general-purpose programming. But talking about Python we not only mean the language, but also its implementations, because Python is actually a specification for a language that can be implemented in many different ways. Python's interpreters are available for many operating systems so that its code could run on a vast variety of them.

Here we want to go through some of the Python implementations which are the true members of a big Python family.

# CPython

CPython is considered to be the 'default' implementation of Python which is written in **C**. This reference implementation compiles Python code to intermediate bytecode and the last one is then being interpreted by a virtual machine. CPython makes it very easy to write C-extensions because in the end Python code is executed by a C interpreter.

The highest level of compatibility with Python packages and C extension modules is being provided by CPython.

CPython is the best for the job is you're writing an open-source Python code and your goal is to reach the audience on a large scale. And it's also the only option to be able to use packages functioning on C extensions.

# Jython

Jython is a Python implementation that makes it very easy to work with **Java** programs. Python code is being compiled by it to Java bytecode which is then executed by the Java Virtual Machine, a.k.a. JVM.

In addition, you can import any Java classes with no further effort.

Jython is definitely the prime choice you're seeking to interface with an existing Java codebase or for any other reason you'll need to write Python code for the JVM.

# IronPython

IronPython is also one of the popular Python implementations. It targets the .NET framework libraries and is written completely in **C#**.

It runs on the .NET Virtual Machine comparable to the JVM and can expose Python code to other languages in the .NET framework. IronPython supports Python 2.7.

It's an ideal choice for Windows developers due to the integration of IronPython directly into the Visual Studio development environment by Python Tools for Visual Studio.

# PyPy

PyPy is an implementation that brings JIT (just-in-time compiler) to Python. It has a lot of confusion around it and this is why.

Pypy is actually two things. On one hand, it's a Python interpreter written in **RPython** which is a subset of Python with static typing. But, on the other hand, it's also a compiler that compiles RPython code for various aims and adds in JIT, and its default platform is C, but it's also possible to target the JVM and others. This way, by being two-in-one, PyPy can dynamically add JIT to an interpreter, generating its own compiler. So the result is a standalone executable that interprets Python source code and exploits JIT optimizations.

If the increase performance is what you are looking for your Python code, Pypy is worth a shot. It's currently over 5 times faster than CPython and supports Python 2.7, while released in beta PyPy3 targets Python 3.

# PythonNet

Python for **.NET** is a package that provides near seamless integration with the .NET Common Language Runtime (CLR). This package allows by using .NET services and components written in any language that targets the CLR (Managed C++, C#, VB, JScript) to build entire applications in Python or script .NET applications. It gives a powerful application scripting tool for .NET developers. It allows to use CLR services and to continue using the existing Python code and C- based extensions at the same time maintaining native execution speeds for Python code.

Python for .NET uses a standard CPython runtime and is good if you want to integrate one or two components from .NET into a standard python application. But it doesn't produce managed code (IL) from Python code.

Python for .NET is released under the open source MIT License and supports from Python 2.6 up to Python 3.5 and 3.6. It can be run in addition to IronPython without conflict.

# Cython

Cython is a static compiler that includes bindings to call C functions. It's used for both the Python programming language and the extended Cython programming language (based on Pyrex).

Cyton allows to easily **write C extensions** for the Python code. It's also possible to add static typing to your existing Python code, but here you'll have to enforce typing in the user's code before passing it to a compiler (similar to Pypy, but not quite).

Cython is the ideal language that could be used for fast C modules that speed up the execution of Python code, to wrap external C libraries and embed CPython into existing applications.

# Grumpy

Grumpy is an experimental Python runtime and a source code transcompiler. It translates Python code into **Go** programs, after what they run within the Go runtime, which means that Go source code gets compiled into native code rather than bytecode and this way it doesn't require a VM.

The idea behind this was to find a way to make concurrent workloads perform well and it was necessary to support a large existing Python codebase. A great degree of compatibility with CPython was required. Grumpy intended to be a near drop-in replacement for CPython 2.7.

The abundance of existing Python C extensions can't be leveraged by Grumpy, but it has no global interpreter lock and can use Go's garbage collection for object lifetime management. And while Grumpy is not an interpreter, it can create optimization opportunities at compile time by means of static program analysis and can import Go packages just like Python modules.

In spite of many great possibilities that Grumpy provide, there still are a lot of things that should be improved.

# MicroPython

MicroPython is an efficient implementation of the Python 3 programming language, a full Python compiler and runtime written in C99. It runs on **microcontrollers** in constrained environments and incorporates a small subset of the Python standard library. It's an interactive prompt (the REPL, a.k.a. Read-eval-print loop) able to execute commands immediately.

MicroPython is being run on a bare metal by the MicroPython pyboard which is a compact electronic circuit board. As a result you get a low-level Python operating system that gives the convenience to control all kinds of electronic projects in addition to the ability to run and import scripts from the built-in filesystem.

This implementation is packed full of advanced features and yet it's compact due to the employment of many advanced coding techniques. Its goal is to be as compatible as possible with normal Python.

The entire MicroPython core is available for general use under the MIT license.

# Conclusion

As you can see, there're quite a number of Python implementations which made Python Family so vast and diverse. And there're even a bigger number of commonly-named and commonly-references "Python" tools that serve completely different purposes. Whether you decide to get more closely familiar with them or you'll stick to some version you've already worked or are working with now, there's definitely a bright side, at least you have a choice.
