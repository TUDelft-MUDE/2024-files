---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.5
---

# PA 2.4A: Gurobi Environment and License

<h1 style="position: absolute; display: flex; flex-grow: 0; flex-shrink: 0; flex-direction: row-reverse; top: 60px;right: 30px; margin: 0; border: 0">
    <style>
        .markdown {width:100%; position: relative}
        article { position: relative }
    </style>
    <img src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png" style="width:100px" />
    <img src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png" style="width:100px" />
</h1>
<h2 style="height: 10px">
</h2>

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 2.4. Due: complete this PA prior to class on Friday, Dec 6, 2024.*


## Overview of Assignment

This assignment confirms you were able to create and activate a Python environment using Anaconda from an `environment.yml` file, and that your Gurobi license has been set up properly.

**Remember:** PA 2.4 has two notebooks that must be completed (A and B). Follow the instructions in **`README.md`** if you have not already done so.


<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 1:</b>   
    
Apply for your personal license for Gurobi (one of the packages installed in `environment.yml`) and add the license file to your computer (in the default folder!). The instructions for this are in the book.

</p>
</div>


<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 2:</b>   
    
Run the cells below. If you have correctly created the Python environment (as described in the README.md) and installed the Gurobi license, there should be no errors. If there are errors, use the Traceback to figure out what should be fixed.

<em>You don't need to understand what the cells are doing, but we wrote a few notes to explain anyway.</em>
</p>
</div>


This cell sets up an optimization model with 3000 variables. That's a lot! We will do something like this in the optimization week. Since you need a license to process this many variables, an error will be returned if you did not install it correctly.

```python
import gurobipy
model = gurobipy.Model()
x = model.addVars(3000, vtype = gurobipy.GRB.CONTINUOUS, name = 'x')
model.update()
model.optimize()
```

The cell below searches for the license file `gurobi.lic` on your computer, and will create a new file `license.lic` in the working directory of this notebook to confirm that you installed Gurobi correctly. 

```python
import sys
from pathlib import Path
import os

def find_license_in_dir(directory: Path):
    license = directory / "gurobi.lic"

    if (license.exists()):
        return license
    else:
        return None
    
def find_license():
    # By default the license is installed in the home directory; this is the most likely location.
    license = find_license_in_dir(Path.home())
    
    if (license): return license
    
    # Otherwise there are other default paths Gurobi will search for each platform.
    if (sys.platform.startswith("linux")):
        license = find_license_in_dir(Path("/opt/gurobi/"))
    elif (sys.platform.startswith("win32")):
        license = find_license_in_dir(Path("C:\\gurobi\\"))
    elif (sys.platform.startswith("darwin")):
        license = find_license_in_dir(Path("/Library/gurobi/"))
    else:
        print("WARNING: Your operating system may not be supported by this function")
        
    if (license): return license
    
    # If all else fails, maybe it was put somewhere strange and the GRB_LICENSE_FILE environment variable was set
    file_path = os.environ.get("GRB_LICENSE_FILE")
    
    if (file_path is not None):
        file_path = Path(file_path)
        if (file_path.exists()):
            return file_path
    
    # Oh nO!
    raise Exception(("Could not find license. If you have an academic license and "
                    "it couldn't be found, copy the license into your repository and "
                    "remove all the info except 'TYPE' and 'VERSION'"))
    
license = find_license()

with open("license.lic", "w") as f:
    f.write(
        "".join(
            filter(
                lambda l: l.startswith("TYPE") or l.startswith("VERSION") or l.startswith("EXPIRATION"), 
                license.open().readlines()
            )
        )
    )
print("License succesfully found and processed!")
```

If you ran all of the cells above, you are ready to go: you successfully created an environment from a `*.yml` file and installed the Gurobi license! Now there is only one thing left to do.


<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 3:</b>   
    
Commit this notebook and the license file that it created to your repository.
</p>
</div>


**End of notebook.**
<h2 style="height: 60px">
</h2>
<h3 style="position: absolute; display: flex; flex-grow: 0; flex-shrink: 0; flex-direction: row-reverse; bottom: 60px; right: 50px; margin: 0; border: 0">
    <style>
        .markdown {width:100%; position: relative}
        article { position: relative }
    </style>
    <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">
      <img alt="Creative Commons License" style="border-width:; width:88px; height:auto; padding-top:10px" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" />
    </a>
    <a rel="TU Delft" href="https://www.tudelft.nl/en/ceg">
      <img alt="TU Delft" style="border-width:0; width:100px; height:auto; padding-bottom:0px" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png"/>
    </a>
    <a rel="MUDE" href="http://mude.citg.tudelft.nl/">
      <img alt="MUDE" style="border-width:0; width:100px; height:auto; padding-bottom:0px" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png"/>
    </a>
    
</h3>
<span style="font-size: 75%">
&copy; Copyright 2023 <a rel="MUDE Team" href="https://studiegids.tudelft.nl/a101_displayCourse.do?course_id=65595">MUDE Teaching Team</a> TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
