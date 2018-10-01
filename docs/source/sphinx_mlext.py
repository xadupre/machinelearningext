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
dll = os.path.normpath(os.path.join(this, "..", "..", "machinelearningext", "bin",
                                    "AnyCPU.Release", "DocHelperMlExt"))
if not os.path.exists(dll):
    raise FileNotFoundError("Unable to find '{0}'.".format(dll))
folds = [_ for _ in os.listdir(dll) if 'nupkg' not in _]
if len(folds) != 1:
    raise FileNotFoundError("Unable to guess where the DLL is in '{0}' (1)".format(dll))
dll = os.path.join(dll, folds[0])
if not os.path.exists(dll):
    raise FileNotFoundError("Unable to guess where the DLL is in '{0}' (2)".format(dll))
mldll = os.path.join(dll, "Scikit.ML.DocHelperMlExt.dll")
if not os.path.exists(mldll):
    raise FileNotFoundError("Unable to find '{0}'".format(mldll))
sys.path.append(dll)

from clr import AddReference

AddReference('Scikit.ML.DocHelperMlExt')

from System.IO import IOException

from Scikit.ML.DocHelperMlExt import MamlHelper

def copy_missing_dll():
    """
    Copies missing DLL.
    """
    rootpkg = os.path.normpath(os.path.join(this, "..", "..", "machinelearning", "packages"))
    misses = [os.path.join(this, "..", "..", "machinelearning", "dist", "Release"),
              os.path.join(rootpkg, "newtonsoft.json", "10.0.3", "lib", "netstandard1.3"),
              os.path.join(rootpkg, "system.memory", "4.5.1", "lib", "netstandard2.0"),
              os.path.join(rootpkg, "system.runtime.compilerservices.unsafe", "4.5.0", "lib", "netstandard2.0"),
              ]

    for miss in misses:
        for dl in os.listdir(miss):
            src = os.path.join(miss, dl)
            if not os.path.isfile(src):
                continue
            dst = os.path.join(dll, dl)
            if not os.path.exists(dst):
                print("copy '{0}' from '{1}'".format(dl, miss))
                shutil.copy(os.path.join(miss, dl), dll)


def maml_pythonnet(script, chdir=False):
    """
    Runs a *maml script* through :epkg:`ML.net`.

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
    MamlHelper.TestScikitAPI2()


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
    
        ? ap
    """)
    out = rst2html(rst, layout="sphinx", writer="rst",
                   directives=[("mamlcmd", MlCmdDirective)])
    print(out)
    