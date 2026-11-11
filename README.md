# inde [![License](https://img.shields.io/github/license/bit-shift-right/inde-lang)](https://github.com/bit-shift-right/inde-lang/blob/main/LICENSE) [![CI](https://github.com/bit-shift-right/inde-lang/actions/workflows/ci.yml/badge.svg)](https://github.com/bit-shift-right/inde-lang/actions/workflows/ci.yml)

<a id="banner-logo" href="https://github.com/bit-shift-right/inde-lang">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="./assets/svg/banner/inde-text-logo-dark.svg?raw=true" type="image/svg+xml">
        <source media="(prefers-color-scheme: light)" srcset="./assets/svg/banner/inde-text-logo-light.svg?raw=true" type="image/svg+xml">
        <img alt="Inde Logo" src="./assets/png/banner/inde-text-logo-dark.png?raw=true">
    </picture>
</a>

A modern, [stack-oriented](https://en.wikipedia.org/wiki/Stack-oriented_programming), 
[concatenative](https://en.wikipedia.org/wiki/Concatenative_programming_language) programming language 
designed with stronger static typing, better error handling and more convenient stack manipulation in mind.

***

## Introduction

The intention of inde is to serve as something of a spiritual successor to some of the most well known 
stack-oriented languages such as [Forth](https://www.forth.com/forth/). Rather than being an outright replacement, 
it is designed to offer familiar syntax and a small learning curve for those already familiar with this style of 
programming while bringing plenty of modern features to ease development. For those new to this paradigm it is also 
designed to be a more intuitive introduction by simplifying many of the most painful aspects of this style of 
programming.

> [!CAUTION]
> inde is a heavily experimental project. The language specification and even implmentation is subject to change 
> between commits. It is not currently suitable for any kind of critical project that relies on the long-term 
> stability of the language. Changes will be thoroughly documented and major revisions that may affect previous 
> code will be clearly highlighted. Use inde at your own discretion.

## Why inde?

Why not? Seriously though, in its current state (and for the forseeable future) inde is little more than a passion 
project of mine. I have always had a fascination with programming languages and their architecture; stack-oriented 
languages in particular, and inde is a way for me to experiment with these concepts in a productive way by 
challenging myself to not only implement a laguage from the ground up, but also tackle elements of their fundamental 
design to solve personal sticking points I have with certain languages and their features (or lack thereof).

As it is probably a question that you thought looking at the project, this is also the reason the initial 
implementation is written in [Python](https://www.python.org/). Python doesn't seem like the most obvious choice 
for writing a compiler, much less an interpreter, however it is the natural option for being able to implement and 
play with features quickly as they come to mind. If and when the language is ever mature enough to warrant it, 
I will inevitably port the implementation to a more sensible language and possibly even 
[bootstrap](https://en.wikipedia.org/wiki/Bootstrapping_(compilers)) the language but for the purpose of 
development and prototyping Python is the most practical choice for me to be able to create freely without being
burdened by the boilerplate of other languages.

## Quick Start

### Installation

With respect to the warning in the introduction, the process for installing inde is also subject to change along 
with the rest of the project. Eventually full release builds of the Python implementation will be available as 
wheels as well as on [PyPI](https://pypi.org/). Currently the only method of installing inde is to build it from 
source. inde itself does not rely on any external packages, however [hatch](https://hatch.pypa.io/latest/), 
[mypy](https://mypy.readthedocs.io/en/stable/getting_started.html), [Ruff](https://docs.astral.sh/ruff/) and 
[pytest](https://docs.pytest.org/en/stable/) are required for building and testing. These can be installed using 
the instuctions in the links to their pages or on most systems with the command below (or using your python package 
manager of choice),

```
python -m pip install hatch mypy ruff pytest
```

You can then clone the [repo](https://github.com/bit-shift-right/inde-lang/tree/main) using,

```
git clone https://github.com/bit-shift-right/inde-lang.git
cd inde-lang
```

Finally, you can either build and install the wheel using,

```
hatch build
python -m pip install dist/inde_lang-0.1.0-py3-none-any.whl
```

Or use the defined hatch environments to run the project from source,

```
hatch shell
inde
```

Or alternatively,

```
hatch run main
```

There are also hatch commands for type-checking, linting and testing,

```
hatch run cleanup
hatch run test:check
```

## Usage

TBD

## Documentation

TBD



