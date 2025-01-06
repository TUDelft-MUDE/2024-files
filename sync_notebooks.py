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
        """Determine if a file needs processing based on hash comparison."""
        if force:
            self.logger.info(f"Force update enabled for {source_path}")
            return True
        
        if not output_path.exists():
            self.logger.info(f"Output file does not exist: {output_path}")
            return True
        
        # For all formats, check against source hash
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

    def convert_to_html(self, input_path: Path, output_path: Path):
        """Convert file to HTML."""
        try:
            self.logger.info(f"Converting {input_path} to HTML at {output_path}")
            if input_path.suffix == '.ipynb':
                nb = self.read_notebook_safely(input_path)
                html_exporter = nbconvert.HTMLExporter()
                html, _ = html_exporter.from_notebook_node(nb)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html)
                self.logger.info("HTML conversion successful")
                return True
            else:
                self.logger.error(f"Unexpected input file type: {input_path.suffix}")
                return False
        except Exception as e:
            self.logger.error(f"Error converting to HTML: {e}")
            return False

    def strip_metadata(self, notebook_path: Path, output_path: Path):
        """Strip metadata from notebook while preserving essential information."""
        try:
            nb = self.read_notebook_safely(notebook_path)
            
            # Move stats logging to convert_notebook method
            # Removing the stats logging from here
            
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
                
            return nb  # Return the notebook object instead of True
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
                        processed = self.convert_to_html(stripped_path, output_path)
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
                                f.seek(0)
                                f.write("<userStyle>Normal</userStyle>\n\n" + content)
                                f.truncate()
                    elif fmt == 'python':
                        output_path = output_base.with_suffix('.py')
                        cmd = ['jupytext', '--to', 'py:percent', str(stripped_path), '-o', str(output_path)]
                        processed = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8').returncode == 0
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
                            with open(output_path, 'w', encoding='utf-8') as f:
                                f.writelines(clean_lines)

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