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

> _“inde [ˈɪn.dɛ] (thence, from there), represents the core philosophy of a concatenative language”_

The intention of inde is to serve as something of a spiritual successor to some of the most well known 
[stack-oriented](https://en.wikipedia.org/wiki/Stack-oriented_programming) languages such as 
[Forth](https://www.forth.com/forth/). Rather than being an outright replacement, it is designed to offer 
familiar syntax and a small learning curve for those already familiar with this style of programming while 
bringing plenty of modern features to ease development. For those new to this paradigm it is also designed to be a 
more intuitive introduction by simplifying many of the most painful aspects of this style of programming.

> [!CAUTION]
> inde is a heavily experimental project. The language specification and even implmentation is subject to change
> between commits. It is not currently suitable for any kind of critical project that relies on the long-term
> stability of the language. Changes will be thoroughly documented and major revisions that may affect previous
> code will be clearly highlighted. Use inde at your own discretion.