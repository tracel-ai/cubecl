# CubeCL Server

This crate is the toolkit for implementing a CubeCL runtime.

A runtime depends on this crate and gets the memory pools, the stream scheduler, the driver helpers and the compilation pipeline.
User code never depends on it.
Everything a kernel author needs lives in `cubecl-runtime`, which this crate re-exports so an implementation sees one module tree.
