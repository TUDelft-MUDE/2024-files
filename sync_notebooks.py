#!/usr/bin/env python3
import os
import json
import shutil
import logging
from pathlib import Path
from typing import List, Dict
import subprocess
import nbformat
from nbconvert.preprocessors import ClearMetadataPreprocessor, ClearOutputPreprocessor
import hashlib

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
        return None

class NotebookSyncer:
    def __init__(self, content_dir: str = "content", output_dir: str = "synced_files"):
        self.content_dir = Path(content_dir).absolute()
        self.output_dir = Path(output_dir).absolute()
        self.error_log = "conversion_errors.log"
        
        # Clear existing log file
        if os.path.exists(self.error_log):
            os.remove(self.error_log)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
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

    def find_notebooks(self) -> List[Path]:
        """Find all Jupyter notebooks in the content directory."""
        if not self.content_dir.exists():
            print(f"\nError: Content directory {self.content_dir} does not exist")
            return []
            
        notebooks = []
        print(f"\nSearching for notebooks in: {self.content_dir}")
        
        for root, dirs, files in os.walk(self.content_dir):
            for file in files:
                if file.endswith('.ipynb'):
                    notebook_path = Path(root) / file
                    try:
                        size = os.path.getsize(notebook_path)
                        notebooks.append((notebook_path, size))
                    except Exception as e:
                        print(f"Error accessing {notebook_path}: {e}")
                        
        print(f"Found {len(notebooks)} notebooks:")
        valid_notebooks = []
        for nb_path, size in notebooks:
            print(f"  - {nb_path} (size: {size:,} bytes)")
            if size == 0:
                print(f"    WARNING: Empty file")
                continue
            
            try:
                with open(nb_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    nbformat.reads(content, as_version=4)
                valid_notebooks.append(nb_path)
            except Exception as e:
                print(f"    ERROR: Could not read notebook: {str(e)}")
                
        print(f"\nValid notebooks: {len(valid_notebooks)}/{len(notebooks)}")
        return valid_notebooks

    def strip_metadata(self, notebook_path: Path, output_path: Path):
        """Strip metadata from notebook while preserving essential information."""
        try:
            nb = self.read_notebook_safely(notebook_path)
            
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
                
            return True
        except Exception as e:
            self.logger.error(f"Error stripping metadata from {notebook_path}: {str(e)}")
            return False

    def convert_notebook(self, notebook_path: Path, formats: List[str] = ['markdown', 'python', 'clean_python']):
        """Convert notebook to specified formats using jupytext."""
        try:
            relative_path = notebook_path.relative_to(self.content_dir)
            output_base = self.output_dir / relative_path.parent / relative_path.stem
            output_base.parent.mkdir(parents=True, exist_ok=True)

            src_hash = get_file_hash(notebook_path)
            stripped_path = output_base.with_suffix('.ipynb')
            
            if stripped_path.exists():
                dst_hash = get_file_hash(stripped_path)
                if src_hash == dst_hash:
                    self.logger.info(f"No changes detected for {notebook_path}")
                    return True
                
            if not self.strip_metadata(notebook_path, stripped_path):
                return False
                
            for fmt in formats:
                try:
                    if fmt == 'markdown':
                        output_path = output_base.with_suffix('.md')
                        cmd = ['jupytext', '--to', 'markdown', str(stripped_path), '-o', str(output_path)]
                        if subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8'):
                            with open(output_path, 'r+', encoding='utf-8') as f:
                                content = f.read()
                                f.seek(0)
                                f.write("<userStyle>Normal</userStyle>\n\n" + content)
                                f.truncate()
                    elif fmt == 'python':
                        output_path = output_base.with_suffix('.py')
                        cmd = ['jupytext', '--to', 'py:percent', str(stripped_path), '-o', str(output_path)]
                    elif fmt == 'clean_python':
                        output_path = Path(str(output_base) + '_clean.py')
                        cmd = [
                            'jupytext', '--to', 'py:light',
                            '--opt', 'comment_magics=false',
                            '--opt', 'notebook_metadata_filter=-all',
                            '--opt', 'cell_metadata_filter=-all',
                            str(stripped_path), '-o', str(output_path)
                        ]
                    
                    if fmt != 'markdown':
                        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
                    
                    if fmt == 'clean_python' and output_path.exists():
                        with open(output_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                        clean_lines = [line for line in lines if not line.strip().startswith('#') and line.strip()]
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.writelines(clean_lines)
                            
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Error converting {notebook_path} to {fmt}: {e.stderr}")
                    return False
                    
            return True
        except Exception as e:
            self.logger.error(f"Error processing {notebook_path}: {str(e)}")
            return False

    def cleanup_output_dir(self, processed_notebooks: List[Path]):
        """Remove files from output directory that don't correspond to existing notebooks."""
        try:
            if not self.output_dir.exists():
                return
            
            import gc
            gc.collect()
            
            import time
            time.sleep(0.1)
            
            valid_stems = {nb.relative_to(self.content_dir).stem for nb in processed_notebooks}
            
            for output_file in self.output_dir.glob('**/*'):
                try:
                    if output_file.is_file():
                        relative_path = output_file.relative_to(self.output_dir)
                        stem = relative_path.stem.replace('_clean', '')
                        
                        if stem not in valid_stems:
                            try:
                                output_file.unlink()
                                self.logger.info(f"Removed obsolete file: {output_file}")
                                
                                dir_path = output_file.parent
                                while dir_path != self.output_dir:
                                    try:
                                        if not any(dir_path.iterdir()):
                                            dir_path.rmdir()
                                    except (PermissionError, OSError):
                                        break
                                    dir_path = dir_path.parent
                            except (PermissionError, OSError) as e:
                                self.logger.warning(f"Could not remove file {output_file}: {e}")
                except Exception as e:
                    self.logger.warning(f"Error processing {output_file}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

    def sync(self):
        """Main synchronization process."""
        self.logger.info("Starting notebook synchronization")
        print("Starting notebook synchronization...")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        notebooks = self.find_notebooks()
        if not notebooks:
            self.logger.info("No notebooks found in content directory")
            return True
            
        processed_notebooks = []
        for notebook in notebooks:
            self.logger.info(f"Processing {notebook}")
            
            if self.convert_notebook(notebook):
                processed_notebooks.append(notebook)
                self.logger.info(f"Successfully processed {notebook}")
            else:
                self.logger.error(f"Failed to process {notebook}")
                
        self.cleanup_output_dir(processed_notebooks)
        
        success = len(processed_notebooks) > 0
        status = "completed successfully" if success else "completed with errors"
        self.logger.info(f"Synchronization {status}")
        return success

if __name__ == "__main__":
    syncer = NotebookSyncer("content")
    success = syncer.sync()
    exit(0 if success else 1)