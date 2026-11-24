# Starting rulebook: Python to C#

Copy this file to `port/RULES.md` and extend it for the repository. Implementers follow it as written. Change a rule when it proves wrong, then redo the units it affected.

## Layout and names

| Python | C# |
| --- | --- |
| One module `pkg/sub/thing.py` | One file `Pkg/Sub/Thing.cs` in namespace `Pkg.Sub` |
| Module-level functions | Static methods on a static class named after the module |
| `snake_case` functions and variables | `PascalCase` methods and properties, `camelCase` locals and parameters |
| `_private` names | `private` or `internal` members |
| Module docstring and function docstrings | XML doc comments with the same text |
| Order of definitions in the file | Keep the same order |

Put the Python path in a comment on the first line of each C# file, for example `// Ported from src/pkg/sub/thing.py`. Reviewers use it to find the original.

## Behaviour that differs without a compile error

These are the risky ones. Wrong code still builds and still passes simple tests.

| Python | C# | Rule |
| --- | --- | --- |
| `int` has no size limit | `int` and `long` wrap on overflow | Use `long` inside a `checked` context. Use `BigInteger` where the code relies on large values |
| `-7 // 2` is `-4`, `-7 % 2` is `1` | `-7 / 2` is `-3`, `-7 % 2` is `-1` | Call `PyCompat.FloorDiv` and `PyCompat.Mod` unless both operands are known to be non-negative |
| `/` on two ints gives a float | `/` on two ints gives an int | Cast to `double` first |
| `round(2.5)` is `2` (rounds half to even) | `Math.Round(2.5)` is `2`, but `(int)2.5` cuts off and `2.5.ToString("F0")` gives `3` | Call `Math.Round` with `MidpointRounding.ToEven`. Do not round by casting or formatting |
| `len(s)`, `s[i]` and slices count code points | `Length` and indexers count UTF-16 units | Use `PyCompat` rune helpers for any length, index, slice or truncation on user text |
| String formatting ignores locale | `ToString()` and `string.Format` use the current culture | Pass `CultureInfo.InvariantCulture`. Compare with `StringComparison.Ordinal` |
| `if x:` is false for `None`, `0`, `""` and empty containers | A condition must be `bool` | Write the exact test meant: `is not null`, `!= 0`, `Count > 0`, `!string.IsNullOrEmpty` |
| `x or default` | `x ?? default` only covers null | Use `??` only when `x` cannot be `0`, `""` or empty. Otherwise write the full condition |
| `dict` keeps insertion order | `Dictionary` has no defined order | Use `OrderedDictionary<TKey, TValue>` when order reaches output. Sort before output where Python sorted |
| `==` on lists, dicts and dataclasses compares contents | `==` on classes compares references | Use records for value types. Use `SequenceEqual` for collections |
| Default argument values are created once | Not allowed for mutable values | Use `null` as the default and create the value inside |
| Mutable default shared between calls on purpose | No equivalent | Make it a field and note it in `GAPS.md` |
| `str` vs `bytes` | `string` vs `byte[]` or `ReadOnlySpan<byte>` | Encode and decode with `Encoding.UTF8` named at each boundary |
| `datetime.now()` is naive local time | `DateTime.Now` is local with a kind flag | Use `DateTimeOffset` and take the clock from `TimeProvider` |

## Types

| Python | C# |
| --- | --- |
| `Optional[T]`, `T \| None` | `T?` with nullable reference types on |
| `X \| Y` of different shapes | A sealed abstract record with one derived record per case |
| `Any` | Do not use `object` or `dynamic` by default. Find the real type. If the data is truly open, use `JsonElement` |
| `dict[str, Any]` from JSON | `JsonElement` or a typed record |
| `list[T]`, `tuple[T, ...]` | `List<T>` for values that change, `IReadOnlyList<T>` in signatures |
| `tuple[A, B]` | A named record. Use a value tuple only for private helpers |
| `Protocol`, `ABC` | `interface`, or `abstract class` when it carries shared code |
| `Enum`, `StrEnum` | `enum` with `[JsonConverter(typeof(JsonStringEnumConverter))]`, with explicit names where the wire value differs |
| `TypedDict`, `dataclass`, Pydantic `BaseModel` | `record` with `required` and `init` properties |
| `Callable[[A], B]` | `Func<A, B>`, or a named delegate when it is part of a public surface |

## Pydantic models and JSON

System.Text.Json does not validate, and its defaults accept more than Pydantic does. Create one shared `JsonSerializerOptions` and use it everywhere.

| Pydantic | C# |
| --- | --- |
| Required field | `required` property |
| Field that may be missing versus field that may be null | `RespectNullableAnnotations = true` and `RespectRequiredConstructorParameters = true` on the shared options |
| `extra="forbid"` | `UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow` |
| `Field(alias=...)`, snake_case wire names | `[JsonPropertyName]`, or `JsonNamingPolicy.SnakeCaseLower` on the shared options |
| Discriminated union on a `type` field | `[JsonPolymorphic(TypeDiscriminatorPropertyName = "type")]` with `[JsonDerivedType]`, and `AllowOutOfOrderMetadataProperties = true` |
| `field_validator`, `model_validator` | A `Validate()` method called right after deserialising, or FluentValidation. Keep the same error text where the judge compares it |
| `Field(default_factory=list)` | Property initialiser `= []` |
| `model_dump(exclude_none=True)` | `DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull` |
| `model_json_schema()` | `JsonSchemaExporter`. Compare output with the Python schema in the judge |

Type coercion is a gap. Pydantic in its default mode turns `"3"` into `3`. System.Text.Json refuses. Check what the Python code accepts and write the answer in `GAPS.md`.

## Async

asyncio runs on one thread, so code between two `await` points never interleaves with other tasks. .NET continues on thread-pool threads, so it can.

| Python | C# |
| --- | --- |
| `async def f()` | `async Task<T> FAsync(..., CancellationToken cancellationToken)` |
| Shared dict or list touched by several tasks | `ConcurrentDictionary`, a `lock`, a `SemaphoreSlim`, or an immutable value |
| `asyncio.gather(*tasks)` | `Task.WhenAll` |
| `asyncio.wait_for(x, timeout)` | A linked `CancellationTokenSource` with `CancelAfter` |
| `asyncio.Queue` | `Channel<T>` |
| `async for` and async generators | `IAsyncEnumerable<T>` with `[EnumeratorCancellation]` |
| `asyncio.CancelledError` arriving at any await | `OperationCanceledException` only where a token is observed. Pass the token to every call that takes one |
| `asyncio.run(main())` | `async Task<int> Main` |
| `asyncio.to_thread(f)` | `Task.Run(f)` |

Never block on a task with `.Result` or `.Wait()`.

## Control flow and resources

| Python | C# |
| --- | --- |
| `with open(...) as f` | `using var f = ...` |
| `async with` | `await using` |
| A context manager whose `__exit__` suppresses exceptions | `try`, `catch`, `finally` written out. `Dispose` cannot see the exception |
| `try` / `except A as e` / `else` / `finally` | `try` / `catch (A e)` / code after the try guarded by a flag / `finally` |
| `raise X from e` | `throw new X(message, e)` |
| Bare `raise` | `throw;` |
| Generators | `IEnumerable<T>` with `yield return`. Generators that use `send()` are a gap |
| List, dict and set comprehensions | LINQ, ended with `ToList()`, `ToDictionary()` or `ToHashSet()` so they run once |
| `for i, x in enumerate(xs)` | `foreach (var (i, x) in xs.Index())` |
| `zip(a, b)` | `a.Zip(b)`. Both stop at the shorter input |
| `sorted(xs, key=f)` | `xs.OrderBy(f)`. Both sorts are stable |

Map each Python exception class to one C# exception class in `port/RULES.md` and keep the hierarchy. Do not replace exceptions with result types before phase 9.

## Decorators

Map each decorator by what it does.

| Purpose | C# |
| --- | --- |
| Registering a function in a table | An attribute plus a registration call at startup |
| Wrapping calls (logging, timing, auth) | Middleware, a `DelegatingHandler`, or an explicit wrapper method |
| Retries | Polly through `Microsoft.Extensions.Resilience` |
| Caching | `IMemoryCache` or `HybridCache` |
| `@property` | A property |
| `@staticmethod`, `@classmethod` | A static method. A classmethod used as an alternate constructor becomes a static factory |
| `@dataclass` | A record |

## Tests

| pytest | xUnit |
| --- | --- |
| A test function | A `[Fact]` method |
| `@pytest.mark.parametrize` | `[Theory]` with `[InlineData]` or `[MemberData]` |
| A fixture | Constructor setup, `IAsyncLifetime`, or a class or collection fixture |
| `tmp_path` | A temp folder created in the constructor and removed in `Dispose` |
| `monkeypatch`, `mock.patch` | An injected interface with a fake. Use `TimeProvider` for clocks and a stub `HttpMessageHandler` for HTTP |
| `pytest.raises(X, match=...)` | `Assert.Throws<X>` then assert on the message |
| `caplog` | A fake `ILogger` that collects entries |

Port each test with the module it covers and keep the test names. Do not drop a test because it is hard to port. Mark it `// TODO(port)` and set the unit to `blocked`.

## Module state and startup

Module-level singletons, registries and config objects become services registered in the dependency injection container. Code that runs on import becomes an explicit startup call. Write each one in `GAPS.md` first, because import order in Python can hide a dependency that C# needs you to state.

## Not allowed

`dynamic`, the null-forgiving `!` operator, `#pragma warning disable`, `[SuppressMessage]`, empty `catch` blocks, `async void`, `.Result` and `.Wait()`. If a rule seems to need one of these, stop and raise it. The scanner fails the unit on the first four.
