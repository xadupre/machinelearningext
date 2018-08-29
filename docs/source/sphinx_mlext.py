"""
Extends Sphinx to easily write documentation.
"""
import sphinx
from docutils.statemachine import StringList
from pyquickhelper.sphinxext.sphinx_runpython_extension import RunPythonDirective

import os
import sys
import shutil
this = os.path.abspath(os.path.dirname(__file__))
dll = os.path.join(this, "..", "..", "machinelearningext", "DocHelperMlExt", "bin", "Release")
folds = [_ for _ in os.listdir(dll) if 'nupkg' not in _]
if len(folds) != 1:
    raise FileNotFoundError("Unable to guess where the DLL is in '{0}'".format(dll))
dll = os.path.join(dll, folds[0])
if not os.path.exists(dll):
    raise FileNotFoundError("Unable to guess where the DLL is in '{0}'".format(dll))
mldll = os.path.join(dll, "DocHelperMlExt.dll")
if not os.path.exists(mldll):
    raise FileNotFoundError("Unable to find '{0}'".format(mldll))
sys.path.append(dll)

from clr import AddReference

AddReference('DocHelperMlExt')

from DocHelperMlExt import MamlHelper

def copy_missing_dll():
    """
    Copies missing DLL.
    """
    misses = [os.path.join(this, "..", "..", "machinelearning", "dist", "Release"),
              os.path.join(this, "..", "..", "machinelearning", "packages", "newtonsoft.json", "10.0.3", "lib", "netstandard1.3")]
    for miss in misses:
        for dl in os.listdir(miss):
            src = os.path.join(miss, dl)
            if not os.path.isfile(src):
                continue
            dst = os.path.join(dll, dl)
            if not os.path.exists(dst):
                print("copy '{0}' to '{1}'".format(dl, dll))
                shutil.copy(os.path.join(miss, dl), dll)


def maml_pythonnet(script, chdir=False):
    """
    Runs a maml_script through :epkg:`ML.net`.

    @param      script          script
    @return                     stdout, stderr
    """
    if chdir:
        cur = os.getcwd()
        os.chdir(dll)
    res = MamlHelper.MamlAll(script, True)
    if chdir:
        os.chdir(cur)
    return res


def maml_test():
    """
    Tests the assembly.
    """
    MamlHelper.TestScikitAPI()


class MlCmdDirective(RunPythonDirective):
    """
    Runs a command line based on :epkg:`ML.net`.
    """
    
    def modify_script_before_running(self, script):
        """
        The methods modifies ``self.content``.
        """
        script = ["from textwrap import dedent",
                  "from sphinx_mlext import maml_pythonnet",
                  "content = dedent('''",
                  script,
                  "''')"
                  "",
                  "out = maml_pythonnet(content)",
                  "print(out)",
                  ]
        return "\n".join(script)


def setup(app):
    """
    Adds the custom directive.
    """
    copy_missing_dll()
    app.add_directive('mlcmd', MlCmdDirective)
    return {'version': sphinx.__display_version__, 'parallel_read_safe': True}


if __name__ == "__main__":
    copy_missing_dll()
    # Test 1
    maml_test()
    # print(maml_pythonnet("?"))
    # Test 2
    from textwrap import dedent
    from pyquickhelper.helpgen import rst2html
    rst = dedent("""
    .. mamlcmd::
        :showcode:
    
        ?
    """)
    out = rst2html(rst, layout="sphinx", writer="rst",
                   directives=[("mamlcmd", MlCmdDirective)])
    print(out)
    