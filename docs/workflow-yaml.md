# Native YAML workflows

Install `power-loop[workflow]` to compile YAML. JSON/IR workflows remain usable
without PyYAML. The compiler does not start tasks or access a filesystem/database;
skill and workspace readers are supplied by the host.

```python
from power_loop.workflow import WorkflowSpec, create_workflow

spec = WorkflowSpec.from_yaml("""
name: draft-and-review
jobs:
  draft:
    prompt: Write a two-sentence explanation of a rainbow.
    tools: []
  review:
    needs: [draft]
    prompt: Check the explanation in your supplied context and return the result.
    tools: []
""")
handle = await create_workflow(
    spec.to_dict(), parent_loop=loop, parent_session_id=session_id,
).start(detached=True)
# handle identifies accepted work. Await handle.task in host code, or let the
# configured completion callback notify the parent. Acceptance is not completion.
```

`WorkflowSpec.from_yaml` defaults to `output_mode="text"`. It sends normal task
inputs and results, without requiring tools to write report files. Explicit
`output-file` and file-output references require `output_mode="file"`.
`compile_workflow_yaml`, `compile_workflow_document` and
`compile_workflow_template` in `power_loop.workflow.yaml` retain the original
file-mode default for existing hosts. They accept an explicit `output_mode` too.
File mode adds append-to-report instructions and `@@FILEREF:...@@` placeholders;
the host supplies file tools and the engine's file-IO adapter.

The compiler supports `jobs/needs/uses/with`, independent-job parallelism,
matrix and aggregate loops, conditions, retry, continuation, fallback, action IO,
structured outputs, parameterized templates, charter frontmatter and file-reference
line slicing. Duplicate YAML keys, unsafe tags, invalid fields, missing required
inputs, invalid references and dependency cycles are compile errors. Errors are
aggregated as `WorkflowYamlError.problems` where validation can continue.

Use `args={...}` with `from_yaml` for declared workflow inputs. A null input
default means required; it is not a placeholder to send to the model. Readers can
be passed as `load_file(skill_name, relative_path)` and `read_workspace(path)`;
they return text or `None`. The host must enforce publication, access and workspace
boundaries. Compilation does not make a supplied reader safe by itself.

Inline `prompt:` jobs are directly executable. Skill charters compile to retained
`charter:skill:...` references; a host must resolve and stamp these through its
spec transformation before execution. Tool declarations become real `spec.tools`
limits, including `[]`; unspecified tools inherit the host's allowed set. Library
control tools such as `pass_turn` may still be advertised. Host policy may further
restrict capabilities. DeepTalk keeps its charter resolver, sandbox/context policy,
todo/note ledger policy and final tool/concurrency clamp.

The former DeepTalk `app.workflow_yaml` module now only re-exports this compiler.
There is one parser/compiler implementation. This does not move DeepTalk's
business tool registry or general jobs/workflow tool facade into the library.
