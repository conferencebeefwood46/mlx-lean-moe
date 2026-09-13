# Conventions

## Code comments

Comments are the exception, not the default. Write code without comments
unless one of the cases below applies.

Budget: at most 1 comment line per 10 lines of code in a file. A single
comment is at most 2 lines. If it needs more, it is not a comment —
it belongs in the commit message

A comment is allowed only when:

- an external constraint is invisible from the code (library bug, API limit,
  protocol requirement) — include a link or issue reference;
- the code deliberately departs from the obvious implementation and would
  otherwise be "fixed" back — state it in one sentence, without
  justification;
- a non-trivial formula or algorithm needs attribution — link the source, do
  not restate it.

Never:

- restate what the next line does;
- explain why an alternative approach was NOT chosen;
- describe consequences or failure scenarios that are readable from the code;
- add JSDoc/docstrings to private functions with descriptive names and
  obvious parameters, or `@param`/`@returns` that repeat the signature and
  types;
- record the state of external data as of a date (policy versions, feed
  contents, "these are the only three today") — it silently goes stale;
  encode that in a test or a runtime check instead;
- add section-header comments inside a function.

## Commit messages

Structure, always:

    <type>(<scope>): <short summary>

    <body — optional>

No footer, ever — no `BREAKING CHANGE:`, no `Closes #123`, nothing after the
body.

Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`,
`build`, `ci`, `chore`, `revert`.

A commit message must:

- use the imperative mood in the summary ("add", not "added" or "adds");
- keep the summary under 72 characters, no trailing period;
- leave exactly one blank line between summary and body;
- use the body to explain why the change was made, not what changed — the
  diff already shows what;
- cover exactly one logical change.

Never:

- add a footer of any kind;
- combine an unrelated fix, refactor, or feature in one commit;
- restate the diff in the body;
- describe a breaking change anywhere but the summary line.

A breaking change is marked with `!` right after the type/scope, in the
summary itself, not in a footer:

    feat(api)!: change response shape of /users

Examples:

    feat(auth): add Google sign-in

    fix(api): handle empty server response

    docs: update README with install instructions

    refactor(user-service): extract validation logic into its own module

    fix(payment): prevent double charge on repeated click

    Button stayed enabled after the first click, letting a user on a slow
    connection fire multiple requests before the first one completed.
