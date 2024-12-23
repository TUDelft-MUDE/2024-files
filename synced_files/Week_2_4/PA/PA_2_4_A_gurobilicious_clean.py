# ----------------------------------------
import gurobipy
model = gurobipy.Model()
x = model.addVars(3000, vtype = gurobipy.GRB.CONTINUOUS, name = 'x')
model.update()
model.optimize()

# ----------------------------------------
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

