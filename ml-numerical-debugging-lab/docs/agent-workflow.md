# Working with a coding agent

This project was built with the help of an AI coding assistant. This note records how its output was controlled, because generated code is only useful if it is verified.

1. **Specify objective checks first.** Every behaviour has a test that can fail: each case defines the property that must hold and the verdict function fails unless the buggy version violates it and the fixed version satisfies it.
2. **Never accept code on inspection alone.** Each generated component was run, and the tests were run before and after changes.
3. **Review for the failure modes agents are prone to:** plausible but wrong formulas, silent shape or dtype changes, tests that pass for the wrong reason, and invented results. For that reason no result is written into the documentation unless it was produced by running the code.
4. **Keep changes small and reproducible.** Seeds are fixed, there are no hidden dependencies, and the whole project runs from the commands in the README.
