---
paths:
  - "**/*.{py,pyi}"
  - "**/*.{cc,cpp,cxx,h,hpp,hxx}"
---

# Google Coding Convention Guide

## Python specific
- Module names: lowercase_with_underscores
- Class names: CapitalizedWords
- Function names: lowercase_with_underscores
- Constants: ALL_CAPS

## C++ specific
- Class names: PascalCase
- Method names: PascalCase
- Variables: snake_case
  - Private members: snake_case_ (trailing underscore)
- Constants: kConstantName (leading k prefix)
- Namespaces: lowercase

## General Principles
- Maximum line length: 100 characters (120 for Google)
- Indentation: 4 spaces (no tabs)
- Comments: Explain WHY, not WHAT
- Import/include order: grouped and alphabetically sorted
