# CubeCL Intermediate Representation

The intermediate representation produced by the CubeCL frontend, and consumed by the backends.

## Reflection

Operations must implement `OperationReflect` to allow for reflecting the `OpCode` and argument list.
In most cases this can be derived, but for operations that take arguments other than `Variable`,
the `OperationCode` derive is available. This derive generates only the opcode and a corresponding
match, while leaving the remaining reflection up to manual implementation.
To make a parameter struct reflectable, use the `OperationArgs` derive.

## Mathematical properties

Mathematical properties like `pure` and `commutative` are marked directly on each operation using
the `#[operation] helper attribute.
