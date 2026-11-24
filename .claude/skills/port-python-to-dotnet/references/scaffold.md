# .NET scaffold

Set the solution up so the compiler and analyzers catch mistakes before a reviewer has to. Put these files at the root of the .NET tree.

## `global.json`

Pins the SDK and selects the test runner. Replace the version with the installed .NET 10 SDK (`dotnet --version`).

```json
{
  "sdk": { "version": "10.0.100", "rollForward": "latestFeature" },
  "test": { "runner": "Microsoft.Testing.Platform" }
}
```

## `Directory.Build.props`

Applies to every project under it.

```xml
<Project>
  <PropertyGroup>
    <TargetFramework>net10.0</TargetFramework>
    <LangVersion>14.0</LangVersion>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
    <TreatWarningsAsErrors>true</TreatWarningsAsErrors>
    <AnalysisLevel>latest-recommended</AnalysisLevel>
    <EnforceCodeStyleInBuild>true</EnforceCodeStyleInBuild>
    <GenerateDocumentationFile>true</GenerateDocumentationFile>
    <InvariantGlobalization>true</InvariantGlobalization>
  </PropertyGroup>
</Project>
```

`InvariantGlobalization` makes culture-sensitive calls behave the same on every machine. Leave it out if the product needs real culture data, and rely on the analyzer rules below.

## `Directory.Packages.props`

Holds every package version in one place. Project files then reference packages without a version.

```xml
<Project>
  <PropertyGroup>
    <ManagePackageVersionsCentrally>true</ManagePackageVersionsCentrally>
  </PropertyGroup>
  <ItemGroup>
    <!-- <PackageVersion Include="YamlDotNet" Version="..." /> -->
  </ItemGroup>
</Project>
```

## `.editorconfig`

Raise the rules that catch the Python-to-C# mistakes from the rulebook to errors.

```ini
root = true

[*.cs]
dotnet_diagnostic.CA1305.severity = error   # pass an IFormatProvider
dotnet_diagnostic.CA1307.severity = error   # pass a StringComparison
dotnet_diagnostic.CA1310.severity = error   # pass a StringComparison for correctness
dotnet_diagnostic.CA2016.severity = error   # forward the CancellationToken
dotnet_diagnostic.CA2007.severity = none    # ConfigureAwait is not needed in app code
dotnet_diagnostic.CS8600.severity = error   # null assigned to non-nullable
dotnet_diagnostic.CS8602.severity = error   # possible null dereference
dotnet_diagnostic.CS8604.severity = error   # possible null argument
```

## Projects

Keep the layout close to the Python package so reviewers can find things.

| Project | Holds |
| --- | --- |
| `src/<Name>` | The ported package, one folder per Python subpackage |
| `src/<Name>.PyCompat` | Helpers for Python behaviour that C# lacks |
| `tests/<Name>.Tests` | Ported unit tests, same names as the pytest tests |
| `tests/<Name>.PyCompat.Tests` | Property tests for the helpers |

## The `PyCompat` library

Build and test this before any other code. Implementers call it. They should not work out these semantics again at each call site.

| Helper | Matches Python |
| --- | --- |
| `FloorDiv(long a, long b)` | `a // b`, rounds toward negative infinity |
| `Mod(long a, long b)` | `a % b`, result takes the sign of `b` |
| `Len(string s)` | `len(s)` in code points |
| `Slice(string s, int? start, int? stop)` | `s[start:stop]` in code points, with negative indexes |
| `At(string s, int index)` | `s[index]` in code points, with negative indexes |
| `Truthy(...)` overloads | `bool(x)` for strings, numbers and collections, for the few places where the type is truly open |
| `Repr(...)` overloads | `repr(x)` and `str(x)` for floats and bools, where their text reaches output |

Generate a file of inputs and expected outputs from Python, including negative numbers, zero divisors, empty strings, emoji and combining marks, and check every helper against it.

## The agent's loop

Run these after every unit, in this order.

```bash
dotnet build -warnaserror
```

```bash
dotnet format --verify-no-changes
```

```bash
dotnet test
```

A single project builds in seconds, so the compiler belongs inside the per-unit loop.
