# Notebook Sync Tool Documentation

## Overview
This tool synchronizes Jupyter notebooks to various formats including Markdown, Python, and HTML.

## Technical Decisions

### Markdown2 Package
Selected for:
- Robust HTML conversion capabilities
- Built-in support for code syntax highlighting
- Extended Markdown feature support including tables and footnotes
- quick solution for rendering Markdown


### Architecture Decisions
- Uses Jupytext for reliable notebook conversion
- Implements file hash comparison for change detection
- Supports selective file updates with force update option
- Preserves essential metadata while cleaning notebooks

## Configuration
See `config_sync.yml` for available options including:
- Sync options (replace_all, force_update)
- Format selection
- Specific notebook processing