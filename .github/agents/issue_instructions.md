# Instructions for AI agents that file issues

Read this before you write an issue. A maintainer reads your issue between other tasks. If it is
long, they will skim it, miss the point, and close it. Write for a person who has two minutes and
wants to check your claim themselves.

## Length

- Every sentence must help the reader reproduce or understand the bug. Remove the others.
- Cut each section to the least text that still works. Then check each section again.
- Do not repeat in one section what another section already says.
- If the whole issue is more than a screen or two (excluding collapsed blocks), it is too long.

## Language

- Write short, plain sentences in the style of ASD-STE100 (Simplified Technical English):
  one idea per sentence, active voice, common words, no filler.
- Use the domain term (for example "tensor core", "shared memory", "kernel") when no short plain
  phrase says the same thing. Do not avoid it, and do not explain it in a separate section. An
  issue never has a glossary or "terms" section.
- Link a source for a term only if a reader who knows Rust and GPUs would not know it. Link it
  once, in the sentence where the term first appears.
- The first time you name a struct, function, or file, link it with a permalink to the commit you
  tested. Use the plain name after that.

## Sections, in this order

1. **Summary.** One to three sentences: what is wrong, and where.
2. **Observed and expected.** What happened (include the exact error or output, trimmed to the
   relevant lines) and what you expected.
3. **Steps to reproduce.** Numbered commands that someone can run. State the exact commit or
   version that you tested.
4. **Repro program.** Inside a collapsed `<details>` block. The smallest program that shows the
   bug. Do not include code that is not needed.
5. **Environment.** Inside a collapsed `<details>` block: OS, GPU and driver, Rust version, backend
   and features, and the commit that you tested. If you did not test on `main`, say which commit
   you tested.
6. **Suspected cause and proposed change.** Optional. Include it only if you can point to specific
   code. Link files, structs, and methods. Keep it short.

## Observed facts and guesses

- Only report as observed what you ran and saw. Never say you ran something that you did not run.
- Mark every guess about the cause as "Suspected". Mark every fix that you did not test as
  "Untested". Use plain hedges such as "probably" or "I did not verify".
- Do not claim a root cause, a severity, or an impact that you did not show.
- If the cause is unknown, say so. A good report without a cause is better than one with a wrong
  cause.
- Do not propose a fix unless the cause is clear to you. If you propose one, say if you tested it.

## Before you submit

- Did you run the steps on the commit that you named?
- Is each claim in "Observed" something that you saw, not something you inferred?
- Can you remove one more sentence or paragraph?
