"""
Extends Sphinx to easily write documentation.
"""
import sphinx
from docutils.statemachine import StringList
from pyquickhelper.sphinxext.sphinx_runpython_extension import RunPythonDirective

import os
import sys
import shutil
import jinja2
this = os.path.abspath(os.path.dirname(__file__))
dll = os.path.normpath(os.path.join(this, "..", "..", "machinelearningext", "bin",
                                    "AnyCPU.Debug", "DocHelperMlExt"))
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


def read_assemblies_to_copy(filename):
    with open(filename, "r") as f:
        lines = [_.split(" - ") for _ in f.readlines()]
    res = []
    for line in lines:
        if len(line) != 2:
            continue
        ass = line[1].strip("\n\r ")
        if "TestMachineLearningExt.dll" in ass:
            continue
        if "testhost" in ass:
            continue
        if os.path.exists(ass):
            res.append(ass)
    return res

def copy_missing_dll():
    """
    Copies missing DLL.
    """
    rootpkg = os.path.normpath(os.path.join(this, "..", "..", "machinelearning", "packages"))
              
    source = os.path.normpath(os.path.join(rootpkg, '..', '..', '_tests', '1.0.0.0', 'Debug', 'TestScikitAPITrain', 'loaded_assemblies.txt'))
    if not os.path.exists(source):
        raise FileNotFoundError("Unable to find '{0}'.\nYou should run test 'TestMamlHelperTest2_AssemblyList'.".format(source))

    misses = [os.path.join(this, "..", "..", "machinelearning", "dist", "Debug"),
              os.path.join(rootpkg, "newtonsoft.json", "10.0.3", "lib", "netstandard1.3"),
              os.path.join(rootpkg, "system.memory", "4.5.1", "lib", "netstandard2.0"),
              os.path.join(rootpkg, "system.runtime.compilerservices.unsafe", "4.5.0", "lib", "netstandard2.0"),
              ]
    misses += read_assemblies_to_copy(source)

    for miss in misses:
        if os.path.isfile(miss):
            miss, dl = os.path.split(miss)
            dst = os.path.join(dll, dl)
            if not os.path.exists(dst):
                print("1>copy '{0}' from '{1}'".format(dl, miss))
                shutil.copy(os.path.join(miss, dl), dll)
        else:
            for dl in os.listdir(miss):
                src = os.path.join(miss, dl)
                if not os.path.isfile(src):
                    continue
                dst = os.path.join(dll, dl)
                if not os.path.exists(dst):
                    print("2>copy '{0}' from '{1}'".format(dl, miss))
                    shutil.copy(os.path.join(miss, dl), dll)


def maml_pythonnet(script, chdir=False, verbose=2):
    """
    Runs a *maml script* through :epkg:`ML.net`.

    @param      script          script
    @param      chdir           to change directory to the DLL location
    @param      verbose         adjust the verbosity
    @return                     stdout and stderr
    """
    if chdir:
        cur = os.getcwd()
        os.chdir(dll)
    res = MamlHelper.MamlScriptConsole(script, True, verbose)
    if chdir:
        os.chdir(cur)
    return res


def maml_test():
    """
    Tests the assembly.
    """
    MamlHelper.TestScikitAPI()
    MamlHelper.TestScikitAPI2()
    iris = os.path.abspath(os.path.join(os.path.dirname(__file__), "iris.txt"))
    if not os.path.exists(iris):
        raise FileNotFoundError("Unable to find '{0}'.".format(iris))
    print(dll)
    cwd = os.getcwd()
    os.chdir(dll)    
    MamlHelper.TestScikitAPITrain(iris)
    os.chdir(cwd)


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


def mlnet_components_kinds():
    """
    Retrieves all kinds.
    """
    from Scikit.ML.DocHelperMlExt import MamlHelper

    kinds = list(MamlHelper.GetAllKinds())
    titles = {
        'anomalydetectortrainer': 'Anomaly Detection', 
        'binaryclassifiertrainer': 'Binary Classification', 
        'clusteringtrainer': 'Clustering', 
        'dataloader': 'Data Loader', 
        'datasaver': 'Data Loader', 
        'datascorer': 'Score (= compute the predictions)', 
        'datatransform': 'Transforms', 
        'ensembledataselector': 'Data Selection', 
        'evaluator': 'Evaluation', 
        'featurescorertrainer': 'Feature Selection (2)', 
        'fourierdistributionsampler': 'Fourrier Sampling', 
        'multiclassclassifiertrainer': 'Multiclass Classification',
        'ngramextractorfactory': 'N-Grams',
        'rankertrainer': 'Ranking',
        'regressortrainer': 'Regression', 
        'tokenizetransform': 'Tokenization'
    }
    return {k: titles[k] for k in kinds if k in titles}
        
    
def builds_components_pages():
    """
    Returns components pages.
    """
    try:
        from .sphinx_mlext_templates import index_template, kind_template, component_template
    except (ModuleNotFoundError, ImportError):
        from sphinx_mlext_templates import index_template, kind_template, component_template
    
    kinds = mlnet_components_kinds()
    pages = {}
    
    # index
    sorted_kinds = list(sorted((v, k) for k, v in kinds.items()))
    template = jinja2.Template(index_template)
    pages["index"] = template.render(sorted_kinds=sorted_kinds)
    
    kind_tpl = jinja2.Template(kind_template)
    comp_tpl = jinja2.Template(component_template)
    
    # kinds and components
    for v, k in sorted_kinds:
        enumc = MamlHelper.EnumerateComponents(k)
        try:
            comps = list(enumc)
        except Exception as e:
            print("Issue with kind {0}\n{1}".format(k, e))
            continue
        if len(comps) == 0:
            continue
            
        comp_names = list(sorted(c.Name.replace(" ", "_") for c in comps))
        kind_name = v
        pages[k] = kind_tpl.render(title=kind_name, fnames=comp_names, len=len)
        
        for comp in comps:
            
            if comp.Arguments is None:
                print(k, comp.Name, comp.Description)
            else:
                args = {}
                for arg in comp.Arguments:
                    args[arg.Name] = dict(Name=arg.Name, ShortName=arg.ShortName or '',
                                          Default=arg.DefaultValue or '',
                                          Description=arg.Help)
                sorted_params = [v for k, v in sorted(args.items())]
                aliases = ", ".join(comp.Aliases)
                
                comp_name = comp.Name.replace(" ", "_")
                pages[comp_name] = comp_tpl.render(title=comp.Name,
                                        aliases=aliases, 
                                        summary=comp.Description,
                                        kind=kind_name, 
                                        sorted_params=sorted_params, 
                                        len=len)
    
    return pages
    

def write_components_pages(app, env, docnames):
    """
    Writes documentation pages.
    """
    pages = builds_components_pages()
    docdir = env.srcdir
    dest = os.path.join(docdir, "components")
    if not os.path.exists(dest):
        os.mkdir(dest)
    for k, v in pages.items():
        d = os.path.join(dest, k) + ".rst"
        with open(d, "w", encoding="utf-8") as f:
            f.write(v)

def setup(app):
    """
    Adds the custom directive.
    """
    copy_missing_dll()
    app.add_directive('mlcmd', MlCmdDirective)
    app.connect("env-before-read-docs", write_components_pages)
    return {'version': sphinx.__display_version__, 'parallel_read_safe': True}


if __name__ == "__main__":
    copy_missing_dll()
    from clr import AddReference
    AddReference('Scikit.ML.DocHelperMlExt')
    from Scikit.ML.DocHelperMlExt import MamlHelper
    pages = builds_components_pages()
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
else:
    from clr import AddReference
    AddReference('Scikit.ML.DocHelperMlExt')
    from System.IO import IOException
    from Scikit.ML.DocHelperMlExt import MamlHelper
