Traceback (most recent call last):
  File "<frozen runpy>", line 203, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Users\Souq AL_Bostan\Projects\my-first-project\portfolio-audit\repos\ML-Engineering-Portfolio_proj\ml-inference-benchmark\infbench\run.py", line 21, in <module>
    raise SystemExit(main())
                     ~~~~^^
  File "C:\Users\Souq AL_Bostan\Projects\my-first-project\portfolio-audit\repos\ML-Engineering-Portfolio_proj\ml-inference-benchmark\infbench\run.py", line 15, in main
    (out / "benchmark.json").write_text(json.dumps(res, indent=2)); (out / "benchmark.md").write_text(to_markdown(res))
                                                                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
  File "C:\Python314\Lib\pathlib\__init__.py", line 809, in write_text
    with self.open(mode='w', encoding=encoding, errors=errors, newline=newline) as f:
         ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Python314\Lib\pathlib\__init__.py", line 771, in open
    return io.open(self, mode, buffering, encoding, errors, newline)
           ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PermissionError: [Errno 13] Permission denied: 'reports\\benchmark.md'
