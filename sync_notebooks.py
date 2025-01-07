#!/usr/bin/env python3
import os
import json
import shutil
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import subprocess
import nbformat
from nbconvert.preprocessors import ClearMetadataPreprocessor, ClearOutputPreprocessor
import hashlib
import yaml
import nbconvert
import markdown2
import pygments
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

class NotebookFooterChecker:
    """Class to check and enforce standard footer in Jupyter notebooks."""
    
    def __init__(self, footer_template_path: Path):
        """
        Initialize with path to the footer.html template.
        
        Args:
            footer_template_path: Path to the html file containing standard footer
        """
        self.footer_template = self._load_footer_template(footer_template_path)
        self.logger = logging.getLogger(__name__)

    def _load_footer_template(self, template_path: Path) -> str:
        """Load the standard footer from footer.html."""
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
            
        except Exception as e:
            self.logger.error(f"Error reading footer template: {e}")
            raise

    def standardize_notebook_footer(self, notebook_path: Path) -> tuple[bool, str]:
        """
        Replace or add standard footer to a notebook.
        
        Args:
            notebook_path: Path to notebook to update
            
        Returns:
            tuple[bool, str]: (success, message)
        """
        try:
            has_standard, message = self.check_notebook_footer(notebook_path)
            if has_standard:
                return True, "Already has standard footer"

            # Read notebook
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)

            # Find any existing footer
            footer_index = None
            for idx, cell in enumerate(reversed(nb.cells)):
                if cell.cell_type == 'markdown' and '**End of notebook.**' in cell.source:
                    footer_index = len(nb.cells) - idx - 1
                    break

            # Create new footer cell with standard footer
            new_footer_cell = nbformat.v4.new_markdown_cell(self.footer_template)

            # Replace or append footer
            if footer_index is not None:
                nb.cells[footer_index] = new_footer_cell
                action = "replaced"
            else:
                nb.cells.append(new_footer_cell)
                action = "added"

            # Write updated notebook
            with open(notebook_path, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)

            return True, f"Standard footer {action}"

        except Exception as e:
            return False, f"Error standardizing footer: {e}"

    def _load_footer_template(self, template_path: Path) -> str:
        """Load the standard footer from footer.html."""
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
            
        except Exception as e:
            self.logger.error(f"Error reading footer template: {e}")
            raise

    def check_notebook_footer(self, notebook_path: Path) -> tuple[bool, str]:
        """
        Check if a notebook has the standard footer.
        
        Args:
            notebook_path: Path to notebook to check
            
        Returns:
            tuple[bool, str]: (has_standard_footer, message)
        """
        try:
            # Read notebook
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
                
            # Look for footer in last few cells
            for cell in reversed(nb.cells):
                if cell.cell_type == 'markdown':
                    if '**End of notebook.**' in cell.source:
                        # Found a footer, check if it matches template
                        if self.footer_template in cell.source:
                            return True, "Has standard footer"
                        else:
                            return False, "Has non-standard footer"
                            
            return False, "No footer found"
            
        except Exception as e:
            return False, f"Error checking footer: {e}"

    def process_directory(self, content_dir: Path, fix_footers: bool = False) -> dict:
        """
        Process all notebooks in a directory.
        
        Args:
            content_dir: Directory containing notebooks
            fix_footers: If True, replace non-standard footers
            
        Returns:
            dict: Summary of processed notebooks
        """
        results = {
            'total': 0,
            'standard': 0,
            'non_standard': 0,
            'errors': 0,
            'fixed': 0,
            'details': []
        }
        
        try:
            for notebook_path in content_dir.rglob('*.ipynb'):
                results['total'] += 1
                
                # Skip template files
                if notebook_path.name == 'header_footer.ipynb':
                    continue
                    
                has_standard, message = self.check_notebook_footer(notebook_path)
                
                if has_standard:
                    results['standard'] += 1
                    status = 'OK'
                else:
                    results['non_standard'] += 1
                    status = 'NON-STANDARD'
                    
                    if fix_footers:
                        success, fix_message = self.standardize_notebook_footer(notebook_path)
                        if success:
                            results['fixed'] += 1
                            status = 'FIXED'
                        else:
                            results['errors'] += 1
                            status = 'ERROR'
                            message = fix_message
                
                results['details'].append({
                    'path': str(notebook_path.relative_to(content_dir)),
                    'status': status,
                    'message': message
                })
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing directory: {e}")
            raise

def get_file_hash(filepath: Path) -> str:
    """Calculate MD5 hash of a file."""
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logging.error(f"Error calculating hash for {filepath}: {e}")
        return ""  # Return empty string instead of None for better comparison



class NotebookSyncer:
    def __init__(self, config_file: Optional[str] = None):
        """Initialize the NotebookSyncer with optional config file."""
        # Default configuration
        self.config = {
            'content_dir': 'content',
            'output_dir': 'synced_files',
            'specific_notebooks': None,
            'formats': ['markdown', 'python', 'clean_python', 'html'],
            'html_options': {
                'template': 'basic',
                'include_code': True,
                'include_outputs': True
            }
        }
        
        # Load config from YAML if provided
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    self.config.update(yaml_config)

        self.content_dir = Path(self.config['content_dir']).absolute()
        self.output_dir = Path(self.config['output_dir']).absolute()
        self.error_log = "conversion_errors.log"
        self.start_time = None
        self.hash_cache_file = Path("notebook_hashes.json")
        self.hash_cache = self._load_hash_cache()
        
        # Clear existing log file
        if os.path.exists(self.error_log):
            os.remove(self.error_log)
        
        # Setup logging with compact format
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(self.error_log, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    def _process_image_paths(self, source_path: Path, output_path: Path, content: str, fmt: str) -> str:
        """
        Process image paths in the content for any format.
        
        Args:
            source_path: Path to source notebook
            output_path: Path to output file
            content: Content to process
            fmt: Format type ('markdown', 'python', 'html', etc.)
        """
        import re
        import base64
        import mimetypes
        import shutil
        from pathlib import Path

        def handle_image_reference(match):
            img_path_str = match.group(1)
            if not img_path_str.startswith(('./','../')):
                return match.group(0)
                
            # Convert to absolute path relative to source notebook
            img_path = (source_path.parent / img_path_str).resolve()
            
            if not img_path.exists():
                self.logger.warning(f"Image not found: {img_path}")
                return match.group(0)
                
            if fmt == 'html':
                # For HTML, embed image as base64
                try:
                    with open(img_path, 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode()
                        mime_type = mimetypes.guess_type(str(img_path))[0] or 'image/png'
                        return f'src="data:{mime_type};base64,{img_data}"'
                except Exception as e:
                    self.logger.error(f"Error embedding image {img_path}: {e}")
                    return match.group(0)
            else:
                # For other formats, copy image to output directory and update path
                try:
                    # Create images directory next to output file
                    images_dir = output_path.parent / 'images'
                    images_dir.mkdir(exist_ok=True)
                    
                    # Copy image to images directory
                    new_img_path = images_dir / img_path.name
                    shutil.copy2(img_path, new_img_path)
                    
                    # Return relative path from output file to new image location
                    rel_path = Path('images') / img_path.name
                    return f'![](./{rel_path})'
                except Exception as e:
                    self.logger.error(f"Error copying image {img_path}: {e}")
                    return match.group(0)

        # Find all image references in the content
        if fmt == 'html':
            pattern = r'src="([^"]+)"'
        else:
            pattern = r'!\[[^\]]*\]\(([^)]+)\)'
            
        return re.sub(pattern, handle_image_reference, content)

    def check_notebook_footers(self):
        """Run footer checks on all notebooks in content directory."""
        try:
            # Look for footer.html in templates directory
            template_path = Path(self.content_dir).parent / 'templates' / 'footer.html'
            if not template_path.exists():
                self.logger.error(f"Footer template not found at: {template_path}")
                return False
                
            # Initialize footer checker
            checker = NotebookFooterChecker(template_path)
            
            # Get footer check config from the yml file
            footer_config = self.config.get('footer_options', {})
            fix_footers = footer_config.get('fix_non_standard', False)
            
            # Process all notebooks
            results = checker.process_directory(self.content_dir, fix_footers=fix_footers)
            
            # Log summary
            self.logger.info("=== Footer Check Summary ===")
            self.logger.info(f"Total notebooks: {results['total']}")
            self.logger.info(f"Standard footers: {results['standard']}")
            self.logger.info(f"Non-standard footers: {results['non_standard']}")
            if fix_footers:
                self.logger.info(f"Fixed footers: {results['fixed']}")
            self.logger.info(f"Errors: {results['errors']}")
            
            # Log details
            self.logger.info("\n=== Detailed Results ===")
            for detail in results['details']:
                self.logger.info(f"{detail['status']}: {detail['path']} - {detail['message']}")
                
            return results['errors'] == 0
            
        except Exception as e:
            self.logger.error(f"Error in footer check: {e}")
            return False

    def read_notebook_safely(self, filepath: Path):
        """Safely read notebook with proper encoding handling."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return nbformat.read(f, as_version=4)
        except UnicodeDecodeError:
            with open(filepath, 'rb') as f:
                content = f.read()
                for encoding in ['utf-8-sig', 'utf-16', 'utf-32', 'cp949', 'euc-kr']:
                    try:
                        text = content.decode(encoding)
                        return nbformat.reads(text, as_version=4)
                    except:
                        continue
                try:
                    return nbformat.reads(content.decode('utf-8', errors='ignore'), as_version=4)
                except:
                    raise ValueError(f"Failed to decode {filepath} with any known encoding")

    def get_notebook_stats(self, notebook):
        """Get statistics about notebook changes."""
        stats = {
            'cells': len(notebook.cells),
            'code_cells': len([c for c in notebook.cells if c.cell_type == 'code']),
            'markdown_cells': len([c for c in notebook.cells if c.cell_type == 'markdown']),
            'total_lines': sum(len(c.source.split('\n')) for c in notebook.cells),
            'code_lines': sum(len(c.source.split('\n')) for c in notebook.cells if c.cell_type == 'code'),
        }
        return stats

    def _load_hash_cache(self) -> Dict[str, str]:
        """Load the hash cache from file."""
        if self.hash_cache_file.exists():
            try:
                with open(self.hash_cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_hash_cache(self):
        """Save the hash cache to file."""
        with open(self.hash_cache_file, 'w') as f:
            json.dump(self.hash_cache, f, indent=2)

    def should_process_file(self, source_path: Path, output_path: Path, force: bool = False, fmt: str = None) -> bool:
        """Determine if a file needs processing based on hash comparison and config."""
        # Check replace_all option from config
        replace_all = self.config.get('sync_options', {}).get('replace_all', True)
        
        # If replace_all is True, always process
        if replace_all:
            return True
            
        # Otherwise, check other conditions
        if force:
            self.logger.info(f"Force update enabled for {source_path}")
            return True
        
        if not output_path.exists():
            self.logger.info(f"Output file does not exist: {output_path}")
            return True
        
        # Check against source hash
        source_hash = get_file_hash(source_path)
        cached_hash = self.hash_cache.get(str(source_path))
        
        if not source_hash:
            self.logger.warning(f"Could not calculate hash for {source_path}")
            return True
        
        if cached_hash is None:  # First time seeing this file
            self.logger.info(f"First time processing {source_path}")
            self.hash_cache[str(source_path)] = source_hash
            return True
            
        if source_hash != cached_hash:
            self.logger.info(f"Content changed for {source_path} (hash mismatch)")
            self.hash_cache[str(source_path)] = source_hash
            return True
        
        return False

    def find_notebooks(self) -> List[Path]:
        """Find all Jupyter notebooks in the content directory."""
        if not self.content_dir.exists():
            self.logger.error(f"Content directory {self.content_dir} does not exist")
            return []
                
        notebooks = []
        self.logger.info(f"Searching in: {self.content_dir}")
        
        # Use specific notebooks if configured
        specific_notebooks = self.config.get('specific_notebooks', {})
        if isinstance(specific_notebooks, dict) and 'files' in specific_notebooks:
            for nb_path in specific_notebooks['files']:
                full_path = self.content_dir / nb_path
                if full_path.exists() and full_path.suffix == '.ipynb':
                    notebooks.append(full_path)
                else:
                    self.logger.warning(f"Specified notebook not found: {nb_path}")
            return notebooks

        # Otherwise find all notebooks
        for root, _, files in os.walk(self.content_dir):
            for file in files:
                if file.endswith('.ipynb'):
                    notebook_path = Path(root) / file
                    notebooks.append(notebook_path)
                            
        self.logger.info(f"Found {len(notebooks)} notebooks")
        return notebooks
    
    def _get_custom_css(self):
        """Return custom CSS for HTML styling."""
        return '''
        <style>
            /* Table styling */
            table {
                border-collapse: collapse;
                margin: 1em 0;
                width: auto;
            }
            th, td {
                padding: 6px 13px;
                border: 1px solid #dfe2e5;
            }
            th {
                background-color: #f6f8fa;
            }
            
            /* Equation styling */
            .MathJax_Display {
                margin: 1.5em auto !important;
                text-align: center !important;
            }
            .MathJax_Display > span {
                text-align: center !important;
                margin: 0 auto !important;
            }
            .MathJax, .MathJax_Display {
                max-width: 100%;
                overflow-x: auto;
                overflow-y: hidden;
            }
            .equation-wrapper {
                width: 100%;
                text-align: center;
                margin: 1.5em 0;
                display: flex;
                justify-content: center;
            }
            
            /* Text and spacing */
            p { margin: 1em 0; }
            .MathJax_Display ~ p,
            p ~ .MathJax_Display {
                margin-top: 1em;
                margin-bottom: 1em;
            }
            
            /* Code and output styling */
            .highlight {
                background: #f6f8fa;
                border-radius: 3px;
                padding: 1em;
                margin: 1em 0;
            }
            pre { white-space: pre; }
            code {
                background-color: rgba(27,31,35,0.05);
                border-radius: 3px;
                padding: 0.2em 0.4em;
                font-size: 85%;
            }
            
            /* Output display */
            .output_wrapper { margin: 1em 0; }
            .output_png {
                text-align: center;
            }
            .output_png img {
                max-width: 100%;
                height: auto;
            }
            
            /* Code outputs */
            .output_text pre {
                background-color: #f8f8f8;
                padding: 0.5em;
                margin: 0.5em 0;
                border-radius: 3px;
                border: 1px solid #ddd;
            }
            
            /* Error outputs */
            .output_stderr pre {
                background-color: #fdd;
                padding: 0.5em;
                margin: 0.5em 0;
                border-radius: 3px;
                border: 1px solid #f88;
            }
            
            /* Print outputs */
            .output_stream pre {
                margin: 0.5em 0;
                padding: 0.5em;
                background-color: #f8f8f8;
                border-left: 3px solid #ccc;
            }
            
            /* Task boxes */
            div[style*="background-color:#AABAB2"],
            div[style*="background-color:#FAE99E"] {
                margin: 1em 0;
                padding: 1em;
                border-radius: 4px;
            }
        </style>
        '''

    def _get_mathjax_config(self):
        """Return MathJax configuration."""
        return '''
        <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
                inlineMath: [['$','$']],
                displayMath: [['$$','$$']],
                processEscapes: true,
                processEnvironments: true,
                skipTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
            },
            "HTML-CSS": { 
                preferredFont: "TeX",
                availableFonts: ["TeX"],
                linebreaks: { automatic: true },
                styles: {
                    ".MathJax_Display": {
                        "text-align": "center !important",
                        margin: "1em 0 !important"
                    }
                }
            },
            SVG: { 
                linebreaks: { automatic: true },
                styles: {
                    "text-align": "center !important",
                    margin: "1em 0 !important"
                }
            },
            CommonHTML: {
                linebreaks: { automatic: true },
                styles: {
                    "text-align": "center !important",
                    margin: "1em 0 !important"
                }
            },
            displayAlign: "center",
            displayIndent: "0",
            TeX: {
                equationNumbers: { autoNumber: "AMS" },
                extensions: ["AMSmath.js", "AMSsymbols.js"]
            }
        });
        </script>
        <script type="text/javascript" id="MathJax-script" async
            src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
        </script>
        '''

    def convert_to_html(self, notebook_path: Path, output_path: Path):
        """Convert notebook to HTML with proper LaTeX rendering and output preservation."""
        try:
            # Read the notebook without clearing outputs
            nb = self.read_notebook_safely(notebook_path)
            
            # Configure HTML exporter with full output preservation
            from nbconvert.exporters.html import HTMLExporter
            from traitlets.config import Config
            import base64
            import re
            
            c = Config()
            
            # Use classic template instead of lab for better compatibility
            c.HTMLExporter.template_name = 'classic'
            
            # Essential preprocessors
            c.HTMLExporter.preprocessors = [
                'nbconvert.preprocessors.ExtractOutputPreprocessor',
                'nbconvert.preprocessors.CSSHTMLHeaderPreprocessor',
                'nbconvert.preprocessors.HighlightMagicsPreprocessor',
            ]
            
            # Ensure all content is included
            c.HTMLExporter.exclude_input = False
            c.HTMLExporter.exclude_output = False
            c.HTMLExporter.exclude_markdown = False
            c.HTMLExporter.exclude_raw = False
            c.HTMLExporter.exclude_unknown = False
            c.HTMLExporter.exclude_code_cell = False
            c.HTMLExporter.anchor_link_text = "¶"
            
            # Enable image embedding
            c.HTMLExporter.embed_images = True
            
            html_exporter = HTMLExporter(config=c)
            
            # Custom CSS for styling
            custom_css = '''
            <style>
                /* Task boxes */
                div[style*="background-color:#AABAB2"] {
                    background-color: rgb(170, 186, 178) !important;
                    color: black !important;
                    padding: 15px !important;
                    margin: 10px !important;
                    border-radius: 10px !important;
                    width: 95% !important;
                }
                
                /* Solution boxes */
                div[style*="background-color:#FAE99E"] {
                    background-color: rgb(250, 233, 158) !important;
                    color: black !important;
                    padding: 15px !important;
                    margin: 10px !important;
                    border-radius: 10px !important;
                    width: 95% !important;
                }
                
                /* Note boxes */
                div[style*="background-color:#C8FFFF"] {
                    background-color: rgb(200, 255, 255) !important;
                    color: black !important;
                    padding: 15px !important;
                    margin: 10px !important;
                    border-radius: 10px !important;
                    width: 95% !important;
                }

                /* Images */
                img {
                    max-width: 100%;
                    height: auto;
                    display: block;
                    margin: 1em auto;
                }

                /* Output cells */
                .output_png img {
                    max-width: 100%;
                    height: auto;
                }

                /* Code cells */
                .input_area {
                    background: #f7f7f7;
                    border: 1px solid #cfcfcf;
                    border-radius: 2px;
                    padding: 0.5em;
                    margin: 0.5em 0;
                }

                /* MathJax */
                .MathJax_Display {
                    margin: 1em 0 !important;
                }
            </style>
            '''

            # Process images and replace references
            def process_images(content, notebook_dir):
                def replace_image(match):
                    path = match.group(1)
                    if path.startswith('./'):
                        full_path = notebook_dir / path[2:]
                    elif path.startswith('../'):
                        full_path = notebook_dir.parent / path[3:]
                    else:
                        full_path = notebook_dir / path
                    
                    if full_path.exists():
                        with open(full_path, 'rb') as f:
                            img_data = f.read()
                            b64_data = base64.b64encode(img_data).decode()
                            mime_type = 'image/png'  # You might want to detect this
                            return f'src="data:{mime_type};base64,{b64_data}"'
                    return match.group(0)
                
                # Handle markdown image syntax
                content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'<img src="\2" alt="\1">', content)
                
                # Handle HTML image tags
                content = re.sub(r'src="([^"]+)"', replace_image, content)
                return content

            # Convert notebook to HTML
            (html, resources) = html_exporter.from_notebook_node(nb)

            # Process any image outputs in resources
            if resources and 'outputs' in resources:
                images_dir = output_path.parent / 'images'
                images_dir.mkdir(exist_ok=True)
                
                for imgname, imgdata in resources['outputs'].items():
                    # Convert to base64
                    b64_data = base64.b64encode(imgdata).decode()
                    html = html.replace(f'src="{imgname}"', 
                                    f'src="data:image/png;base64,{b64_data}"')

            # Process images in markdown and HTML
            html = process_images(html, notebook_path.parent)

            # Fix task box styles
            def fix_boxes(match):
                style = match.group(1)
                if 'background-color:#AABAB2' in style:
                    return f'style="{style}; background-color:#AABAB2 !important;"'
                elif 'background-color:#FAE99E' in style:
                    return f'style="{style}; background-color:#FAE99E !important;"'
                elif 'background-color:#C8FFFF' in style:
                    return f'style="{style}; background-color:#C8FFFF !important;"'
                return match.group(0)
                
            html = re.sub(r'style="([^"]*)"', fix_boxes, html)

            # Add custom CSS and MathJax config
            html = html.replace('</head>', f'{custom_css}{self._get_mathjax_config()}</head>')
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the processed HTML file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)

            return True
            
        except Exception as e:
            self.logger.error(f"Error converting to HTML: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _process_latex_in_html(self, html_content: str) -> str:
        """
        Process LaTeX inside HTML blocks, handling escaped dollar signs.
        
        Args:
            html_content: The HTML content to process
            
        Returns:
            Processed HTML content with proper LaTeX handling
        """
        import re
        
        def replace_latex(match):
            """Helper function to process LaTeX blocks."""
            content = match.group(1)
            # Handle escaped dollar signs
            content = re.sub(r'\\\\(\$)', r'\1', content)  # Double backslash before $
            content = re.sub(r'\\(\$)', r'\1', content)    # Single backslash before $
            return f'\\({content}\\)'  # Use \( \) for inline math
            
        def replace_latex_display(match):
            """Helper function to process display LaTeX blocks."""
            content = match.group(1)
            # Handle escaped dollar signs
            content = re.sub(r'\\\\(\$)', r'\1', content)
            content = re.sub(r'\\(\$)', r'\1', content)
            return f'\\[{content}\\]'  # Use \[ \] for display math
        
        # Process LaTeX inside HTML tags
        html_pattern = r'<[^>]*>'
        parts = re.split(html_pattern, html_content)
        for i in range(len(parts)):
            if i % 2 == 1:  # Inside HTML tag
                # Process inline LaTeX
                parts[i] = re.sub(r'\$([^\$]+)\$', replace_latex, parts[i])
                # Process display LaTeX
                parts[i] = re.sub(r'\$\$([^\$]+)\$\$', replace_latex_display, parts[i])
        
        return ''.join(parts)

    def strip_metadata(self, notebook_path: Path, output_path: Path):
        """Strip metadata from notebook while preserving essential information."""
        try:
            nb = self.read_notebook_safely(notebook_path)
            
            # Remove 'id' field from cells to prevent validation errors
            for cell in nb.cells:
                if 'id' in cell:
                    del cell['id']
            
            clear_metadata = ClearMetadataPreprocessor(
                preserve_cell_metadata=['tags'],
                preserve_notebook_metadata=['kernelspec']
            )
            clear_output = ClearOutputPreprocessor()
            
            nb, _ = clear_metadata.preprocess(nb, {})
            nb, _ = clear_output.preprocess(nb, {})
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8', errors='ignore') as f:
                nbformat.write(nb, f)
                
            return nb
        except Exception as e:
            self.logger.error(f"Error stripping metadata: {str(e)}")
            return None

    def convert_notebook(self, notebook_path: Path, file_index: int, total_files: int):
        try:
            start_time = time.time()
            
            # Calculate relative and output paths first
            relative_path = notebook_path.relative_to(self.content_dir)
            output_base = self.output_dir / relative_path.parent / relative_path.stem
            stripped_path = output_base.with_suffix('.ipynb')
            
            self.logger.info(f"[{file_index}/{total_files}] Checking {relative_path}")
            
            # Get force_update option
            force_update = False
            specific_notebooks = self.config.get('specific_notebooks', {})
            if isinstance(specific_notebooks, dict):
                options = specific_notebooks.get('options', {})
                if isinstance(options, dict):
                    force_update = options.get('force_update', False)

            # Check if any format needs processing
            needs_processing = self.should_process_file(notebook_path, stripped_path, force_update)
            
            if not needs_processing:
                self.logger.info(f"No changes detected for {relative_path}")
                return True
                
            self.logger.info(f"Processing {relative_path}")

            # Strip metadata and get notebook object first
            nb = self.strip_metadata(notebook_path, stripped_path)
            if nb is None:  # If stripping failed
                return False
                    
            # Log stats
            stats = self.get_notebook_stats(nb)
            self.logger.info(f"Stats: {stats['cells']} cells ({stats['code_cells']} code, "
                        f"{stats['markdown_cells']} markdown), {stats['total_lines']} lines")
            # Process all formats
            processed_formats = []
            for fmt in self.config['formats']:
                try:
                    processed = False
                    if fmt == 'html':
                        output_path = output_base.with_suffix('.html')
                        processed = self.convert_to_html(notebook_path, output_path)
                    elif fmt == 'markdown':
                        output_path = output_base.with_suffix('.md')
                        cmd = ['jupytext', '--to', 'markdown',
                            '--opt', 'notebook_metadata_filter=-all',
                            '--opt', 'cell_metadata_filter=-all',
                            str(stripped_path), '-o', str(output_path)]
                        processed = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8').returncode == 0
                        if processed:
                            with open(output_path, 'r+', encoding='utf-8') as f:
                                content = f.read()
                                if content.startswith('---'):
                                    end_idx = content.find('---', 3)
                                    if end_idx != -1:
                                        content = content[end_idx + 3:].lstrip()
                                # Process image paths for markdown
                                content = self._process_image_paths(notebook_path, output_path, content, 'markdown')
                                f.seek(0)
                                f.write("<userStyle>Normal</userStyle>\n\n" + content)
                                f.truncate()
                    elif fmt == 'python':
                        output_path = output_base.with_suffix('.py')
                        cmd = ['jupytext', '--to', 'py:percent', str(stripped_path), '-o', str(output_path)]
                        processed = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8').returncode == 0
                        if processed:
                            with open(output_path, 'r+', encoding='utf-8') as f:
                                content = f.read()
                                # Process image paths for python
                                content = self._process_image_paths(notebook_path, output_path, content, 'python')
                                f.seek(0)
                                f.write(content)
                                f.truncate()
                    elif fmt == 'clean_python':
                        output_path = Path(str(output_base) + '_clean.py')
                        cmd = [
                            'jupytext', '--to', 'py:light',
                            '--opt', 'comment_magics=false',
                            '--opt', 'notebook_metadata_filter=-all',
                            '--opt', 'cell_metadata_filter=-all',
                            str(stripped_path), '-o', str(output_path)
                        ]
                        processed = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8').returncode == 0
                        if processed and output_path.exists():
                            with open(output_path, 'r', encoding='utf-8') as f:
                                lines = f.readlines()
                            clean_lines = [line for line in lines if not line.strip().startswith('#') and line.strip()]
                            content = ''.join(clean_lines)
                            # Process image paths for clean python
                            content = self._process_image_paths(notebook_path, output_path, content, 'python')
                            with open(output_path, 'w', encoding='utf-8') as f:
                                f.write(content)

                    if processed:
                        processed_formats.append(fmt)
                        self.logger.info(f"Successfully processed {fmt} format")
                    else:
                        self.logger.warning(f"Failed to process {fmt} format")
                                    
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Error converting to {fmt}: {e.stderr}")
                    return False
                        
            end_time = time.time()
            self.logger.info(f"Successfully processed formats: {processed_formats}")
            self.logger.info(f"Completed in {end_time - start_time:.2f} seconds")
            return True
        except Exception as e:
            self.logger.error(f"Error processing: {str(e)}")
            return False

    def cleanup_output_dir(self, processed_notebooks: List[Path]):
        """Remove files from output directory that don't correspond to existing notebooks."""
        try:
            if not self.output_dir.exists():
                return

            # Check replace_all option
            replace_all = self.config.get('sync_options', {}).get('replace_all', True)
            if not replace_all:
                self.logger.info("Skipping cleanup due to replace_all=False")
                return

            valid_stems = {nb.relative_to(self.content_dir).stem for nb in processed_notebooks}
            
            for output_file in self.output_dir.glob('**/*'):
                try:
                    if output_file.is_file():
                        relative_path = output_file.relative_to(self.output_dir)
                        stem = relative_path.stem.replace('_clean', '')
                        
                        if stem not in valid_stems:
                            try:
                                output_file.unlink()
                                self.logger.info(f"Removed: {output_file.relative_to(self.output_dir)}")
                                
                                dir_path = output_file.parent
                                while dir_path != self.output_dir:
                                    if not any(dir_path.iterdir()):
                                        dir_path.rmdir()
                                    dir_path = dir_path.parent
                            except (PermissionError, OSError) as e:
                                self.logger.warning(f"Could not remove {output_file.relative_to(self.output_dir)}: {e}")
                except Exception as e:
                    self.logger.warning(f"Error processing {output_file}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

    def sync(self):
        """Main synchronization process."""
        try:
            self.start_time = datetime.now()
            self.logger.info(f"=== Starting synchronization at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')} ===")
            
            # Check if footer checks are enabled in config
            if self.config.get('footer_options', {}).get('enabled', True):
                if not self.check_notebook_footers():
                    self.logger.error("Footer check failed")
                    return False
            
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            notebooks = self.find_notebooks()
            if not notebooks:
                self.logger.info("No notebooks found")
                return True
                
            processed_notebooks = []
            total_notebooks = len(notebooks)
            
            for idx, notebook in enumerate(notebooks, 1):
                if self.convert_notebook(notebook, idx, total_notebooks):
                    processed_notebooks.append(notebook)
                    
            self.cleanup_output_dir(processed_notebooks)
            
            # Save hash cache after processing
            self._save_hash_cache()
            
            end_time = datetime.now()
            duration = end_time - self.start_time
            self.logger.info(f"=== Completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')} (Duration: {duration}) ===")
            return len(processed_notebooks) > 0
            
        except KeyboardInterrupt:
            self.logger.info("Process interrupted by user")
            self._save_hash_cache()  # Save cache even on interrupt
            return False

if __name__ == "__main__":
    syncer = NotebookSyncer("config_sync.yml")
    success = syncer.sync()
    exit(0 if success else 1)