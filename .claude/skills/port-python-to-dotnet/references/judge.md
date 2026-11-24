# The judge

The judge is the set of checks that decides whether the .NET build behaves like the Python build. Build it first. A pytest suite cannot run against a .NET program, and if the porting agent also ports the tests, it controls its own exam. So the judge has to work from outside both programs.

## Layers

Build the layers the repository needs, in this order.

### 1. Recorded contracts

Run the Python program and save what it produces as golden files. For a CLI, save standard output, standard error and the exit code for a list of commands. For an HTTP server, save the status, headers that matter and body for a list of requests. For server-sent events, save the ordered list of event types and payloads.

Write a runner that takes a base command or base URL, replays every case and compares the result with the golden file. The same runner then runs against either build. Put it in `port/judge/` with the golden files next to it.

Generate golden files from the Python program at a pinned commit. Never write expected output by hand.

### 2. Pure functions

Some logic has no network or model in it: loading YAML into validated config, producing JSON Schema, building prompts, rendering templates. Give Python a small command that reads a case from standard input and prints normalised JSON. Give the .NET build the same command. The judge feeds both the same cases and compares.

Run Python as a separate process. Embedding Python inside the .NET process adds a second runtime to debug and is not needed for a comparison.

### 3. The model boundary

Model replies change from run to run, so compare one level below them. Point both builds at a stub provider and compare the request each one sends: the messages, the tool schemas and the settings. Then give both builds the same scripted fake model replies and compare the events they emit.

### 4. Properties for helpers

The `PyCompat` helpers (floor division, Python modulo, code point slicing) get property-based tests with CsCheck or FsCheck. Each test generates many inputs and compares the helper with values computed by Python and stored in a file.

## Normalising output

Decide once what counts as noise, then remove it in the runner and write the list in `port/judge/README.md`. Typical noise is timestamps, random identifiers, absolute paths, port numbers and durations. Key order in JSON objects is noise only if no consumer depends on it. Do not add to the list later to make a failing case pass without the user's agreement.

## Guards

Add cases by hand for empty input, zero, negative numbers, very large numbers, missing optional fields, null values, text outside ASCII (include an emoji and a combining mark) and malformed input that should produce an error. Random and recorded inputs tend to miss these.

Prove the judge catches breakage. Change the Python code so it is wrong in a small way, run the judge, and confirm a failure. Do this for each layer. A judge that stays green on broken code is not finished.

Make the judge read-only once the user approves it. Implementer agents must not be given write access to `port/judge/`.

Print the number of cases run at the end of each run, and fail the run if that number is lower than the number stored in `port/judge/count.txt`.

## Tracking progress

Keep `port/judge/expected-failures.txt` with one case name per line. At the start every case is in it. A run passes when the set of failing cases equals the file. Remove a line when its case passes, so the file shrinks to empty by the end of phase 7. A case that passes while still listed fails the run, which keeps the file accurate.

## What the judge does not prove

Passing means the two builds agree on the cases you have. Teams that ported large codebases with a full pass of their existing tests still found regressions afterwards. Plan for that: keep the Python build runnable after cutover and add a case to the judge for every difference found later.
