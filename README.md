# Spring 2025 CS336 lectures

This repo contains the lecture materials for "Stanford CS336: Language modeling from scratch".

Percy's lectures are stored in executable form as `lecture_*.py` and Tatsu's lectures are in PDF form under `nonexecutable/`. Lectures will be uploaded as the class progresses.

## Non-executable (ppt/pdf) lectures

Located in `nonexecutable/`as PDFs

## Executable lectures

Located as `lecture_*.py` in the root directory

### Install

        npm create vite@latest trace-viewer -- --template react
        cd trace-viewer
        npm install
        npm run dev

### Compile

        python execute.py -m lecture_??.py

### Deploy

        cd trace-viewer
        npm run build
        git add dist/assets
