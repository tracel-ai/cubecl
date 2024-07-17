# cargo xtask CLI for CubeCL

The `xtask` command line interface for Cargo is a handy tool designed to help you execute common tasks with ease when developing for CubeCL.
Whether you'running checks, tests or preparing for a pull request, `xtask` has got you covered.

This guide will walk you through each command, explaining when and why to use them.

## Getting Started

To get started with `xtask`, simply run:

```sh
cargo xtask --help
```

This will display the main help screen with a list of available commands.

When a command accepts the `--target <TARGET>` argument it means that the command can be applied
on different components of the repository:
- `crates`: the Rust crates in the cargo workspace, they are usually in `crates` directory
- `examples`: the example crates in the cargo workspace, they are in `examples` directory

For instance using the `test` command:

```sh
# run all the tests
cargo xtask test --target crates all
# run the unit tests only
cargo xtask test --target crates unit
# run the integration tests only
cargo xtask test --target crates integration
# run the documentation tests only
cargo xtask test --target crates documentation
```

To run everything:

```sh
cargo xtask test --target all all
```

**Pro-tip:** create an alias in your shell of choice to map `cargo xtask` to something easy to type like `cx`.

For bash:

```bash
nano ~/.bashrc

# add this to the file
alias cx='cargo xtask'

# save and source the file or restart the shell session
source ~/.bashrc
```

For fish:

```fish
nano ~/.config/fish/config.fish


# add this to the file
alias cx='cargo xtask'

# save and source the file or restart the shell session
source ~/.config/fish/config.fish
```

For powershell:

```powershell
notepad $PROFILE

# add this at the end of file
function cx {
    cargo xtask $args
}

# save and quit then open a new powershell terminal
```

## Commands

### Pre-Pull Request Checks

This is one of the most important command for contributors.

Before opening a pull request, it's essential to ensure that all checks and tests pass.
The `pull-request-checks` command consolidates this process, running all necessary tests and checks that should pass before a pull request is submitted.
This helps maintain code quality and ensures that new contributions don’t introduce any issues.

Usage:
```sh
cargo xtask pull-request-checks
```

### Running Checks

`check` is another important command for contributors. It is recommended to explore and learn about its various sub-commands.

The `check` command is designed to help you maintain code quality during development.
It runs various checks and fixes issues, ensuring that your code is clean and follows best practices.
You can target specific aspects such as auditing dependencies, formatting code, or running linter checks.

There’s also an option to run all checks at once, providing a comprehensive validation of your codebase.

The main difference between the `check` command and the `ci` command is that the `check` command will attempt to fix the encountered issues for you.
You'll get a prompt at the beginning to confirm that the `check` can modify you code. For this reason, tt is recommended to commit your changes before
executing this command.

Usage:
```sh
cargo xtask check --target all lint
```

### Running Tests

Testing is a crucial part of development, and the `test` command is designed to make this process easy.

It allows you to run various types of tests, including unit tests, integration tests, and documentation tests.
You can choose to run all tests at once, ensuring comprehensive coverage and catching any issues early.

Usage:
```sh
cargo xtask test --target <TARGET> <COMMAND>
```

### Continuous Integration

The `ci` command is tailored for Continuous Integration (CI) workflows.

It performs a series of checks and tests to ensure that your code is ready for integration.
This includes auditing dependencies, formatting, linting, and running different types of tests (unit, integration, and documentation).

Contrary to the `check` command, this command won't attempt to auto-fix the errors, in other words this command only reports errors.

It is useful to execute it to reproduce errors you can encounter in our pull requests.

Usage:
```sh
cargo xtask ci --target <TARGET> <COMMAND>
```

### Bumping Versions

This is a command reserved for CubeCL maintainers.

The `bump` command is used to update the version numbers of all crates in the repository.
This is particularly useful when you're preparing for a new release and need to ensure that all crates have the correct version.

You can bump the version by major, minor, or patch levels, depending on the changes made.
For example, if you’ve made breaking changes, you should bump the major version.
For new features that are backwards compatible, bump the minor version.
For bug fixes, bump the patch version.

Usage:
```sh
cargo xtask bump <COMMAND>
```


### Publishing Crates

This is a command reserved for CubeCL maintainers and is mainly used by the `publish` workflow.

This command automates the process of publishing CubeCL crates to `crates.io`, the Rust package registry.
By specifying the name of the crate, `xtask` handles the publication process, ensuring that the crate is available for others to use.

Usage:
```sh
cargo xtask publish <NAME>
```

