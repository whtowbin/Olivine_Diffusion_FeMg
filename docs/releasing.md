# Releasing FORSTER to PyPI

The package is `forster` (free on PyPI and TestPyPI as of 2026-10-17). Uploads go through
`.github/workflows/publish.yml` with PyPI's trusted publishing: no API tokens are stored.
Every upload is confirmed with Henry first.

## One-time setup (Henry, on the websites)

1. **PyPI** (pypi.org, logged in as the owner): Account → Publishing → "Add a new pending
   publisher": project name `forster`, owner `whtowbin`, repository
   `Olivine_Diffusion_FeMg`, workflow `publish.yml`, environment `pypi`.
2. **TestPyPI** (test.pypi.org; a separate account): the same, with environment
   `testpypi`.
3. **GitHub** (repository Settings → Environments): create `pypi` and `testpypi`.
   - On `pypi`: add Henry as a **required reviewer**, and limit deployments to tags `v*`.
     The publish token exists only inside this environment, so nothing can publish
     without Henry's click. That covers a pushed tag from an agent session or a leaked
     GitHub credential.
   - On `testpypi`: limit deployments to `master`.
   - Settings → Rules → Rulesets: a tag ruleset on `v*` that blocks updates and
     deletions, so a released tag cannot be moved to other code.
   - After any workflow change, run `uvx zizmor .github/workflows/`: it should report no
     findings.
4. **Read the Docs** (readthedocs.org): import the repository; it builds from
   `.readthedocs.yaml`. The project slug `forster` gives https://forster.readthedocs.io
   (the URL in pyproject.toml's `Documentation`).

## Each release

1. Clear the checklist in CHANGELOG.md "Before release" (citations, release notes on
   the cross-term fix and the uncertainty versions, round robin results out of the
   public tree).
2. Set the version in `pyproject.toml` (e.g. `0.1.0`), move the CHANGELOG's
   "Unreleased" entries under that version and date, commit.
3. **TestPyPI first:** run the "Publish to PyPI" workflow by hand (Actions → Publish to
   PyPI → Run workflow). It builds, checks the metadata, installs the wheel in a clean
   environment and runs the worked example, then uploads to TestPyPI. Check the page and
   a clean install:
   `uv venv t && VIRTUAL_ENV=t uv pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ forster`.
4. **PyPI:** tag and push the tag: `git tag v0.1.0 && git push origin v0.1.0`. The
   workflow publishes to PyPI (after the reviewer approves, if set).
5. Check https://pypi.org/project/forster/ and `pip install forster` in a clean
   environment; build the app installers if wanted (`gui-installers.yml` also runs on
   `v*` tags).

## Local check (no upload)

    uv build && uvx twine check dist/*

The sdist holds only `src/forster`, the README, the license and pyproject.toml
(`[tool.hatch.build.targets.sdist]`): the docs, analyses and plans stay in the
repository. The README is the PyPI page, so its images use absolute URLs.
