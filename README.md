# Spring 2026 CS336 lectures

This repository contains the lecture materials for Stanford's Language Modeling from Scratch (CS336).

## Executable lectures

These are named `lecture_XX.py`.

Compiled traces are already checked in under `var/traces/`. If you only want to
read the lectures, do the setup below and skip to "Viewing".

### Setup

Install the Python dependencies (this installs the `edtrace` package):

        uv sync

Clone the frontend and install its dependencies:

        git clone https://github.com/percyliang/edtrace
        npm --prefix=edtrace/frontend install

### Compiling a lecture

`execute` is a module inside the installed `edtrace` package, not a script in
this repository:

        uv run python -m edtrace.execute -m lecture_01

This generates `var/traces/lecture_01.json` and caches any images as appropriate.

### Viewing

Start the dev server:

        npm --prefix=edtrace/frontend run dev

Then open `http://localhost:5173/?trace=lecture_01`.

The `trace` parameter accepts a bare lecture name, which expands to
`var/traces/lecture_01.json`. A full path also works:
`?trace=var/traces/lecture_01.json`.

The dev server reads the repository root through two symlinks in
`edtrace/frontend/public/`: `var` and `images`.

### Deploying to the main website

The build reads its output directory and its public base path from environment
variables (see `edtrace/README.md`):

        mkdir -p dist
        (cd dist && ln -s ../var && ln -s ../images)

        export VITE_EDTRACE_BASE_DIR=/lectures   # public base path of the site
        export VITE_EDTRACE_DIST_DIR=$PWD        # absolute path, writes index.html and assets/ here

        npm --prefix=edtrace/frontend run build
        git add index.html assets
        git commit -am "<some message>"
        git push

## Non-executable lectures

These are named `lecture_XX.pdf`.
