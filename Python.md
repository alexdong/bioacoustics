# Python Philosophy, Conventions and Tools

## Development Environment

- Python: 3.12
- Tools: `ruff` `mypy`, `pyright` and `py.test`
- Dev Env: osx 15.1 with brew and common utils installed. 
- AI Availability: both Claude and OpenAI API KEYs available through environment variables

## Requirement

We are an one man's army. Readability after a long hiatus is crucial. 

This core requirement sets the tone for all the code practices.

In order to understand what's wrong with the code after having touching 
a piece of code for up to a couple of years require simple, shorter code 
with a data flow with a state transitions that is easy to trace.

## Conventions

To meet the above requirements, here are some practices I get religious about.
Make sure you follow them when you work with existing code base or create new
code.

- DO produce code that maximise readability
  - DO put the top level function on top of a file. Remember that we read from top to bottom.
  - DO produce a `mermeid` diagram for the architectural decision when applicable.
  - DO produce compact code. Less lines of code means less things to read and load into brain.
  - DO choose names wisely to minimise the effort required to understand the intent.

- DO fail early and fast.  
  - DO NOT use `try` and `except`. Let the code fail.
  - DO use a generous pinch of `assert` to check the validity of the code and fail fast.

- DO create as few mental constructs as possible to minimise cognitive load
  - DO follow `pylint` requirement as defined under `pyproject.toml` 
  - DO NOT use `class` unless the situation calls for it.
  - DO use one-liner, like lambda or list comprehension, to reduce number of lines.
  - DO NOT write docstrings for the functions unless the intention is unclear or
    the implementation is complex.
  - DO prefer calling out to unix CLI tools instead of relying on Python libraries. 
    e.g. (`pbcopy` instead of `pyperclip`, or `magick` instead of `Pillow`)

- DO make it easy to trace the logic flow and data transformation of the code
  - DO use `print(f"[LEVEL] {message}")` to document key actions and state transitions.
  - DO make printed messages easy to search for in log files.
  - DO use one empty line between key concept blocks within a function for visual scans.
  - DO use two empty line breaks between functions and classes. 
  - DO use the /tmp directory directory when the code requires generating temporary files.
  - DO NOT clean up temporary files to help examine intermediate changes.

- DO use types extensively for all functions and classes. So that it's easier to 
  track the purpose of the functions by looking at the signatures. (Equally
  importantly for AI to understand the code).
  - DO NOT include type information in variable name.
  - DO make sure the code pass both `mypy` and `pyright` for type checking. 
  - DO create `.stubs/*.pyi` files if there is no existing type hint for 3rd party libraries 

- DO make the code more fun to follow through
  - DO use emoji and lighthearted messages in tools
  - DO use more jeeves from woodhouse novels in server functions.
  - DO remember that humor is best served in moderation

- DO include a `if __name__ == "__main__"` block in every script
  - DO make sure the __main__ function is available for every script
  - DO use the __main__ function to both demonstrate the usage and test the code

