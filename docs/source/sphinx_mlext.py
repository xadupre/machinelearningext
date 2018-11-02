"""
Extends Sphinx to easily write documentation.
See the LICENSE file in the project root for more information.
"""
import sphinx
from docutils.statemachine import StringList
from pyquickhelper.sphinxext.sphinx_runpython_extension import RunPythonDirective
from csharpy.sphinxext import RunCSharpDirective

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

docs = os.path.normpath(os.path.join(this, "..", "..", "machinelearning", "docs"))
if not os.path.exists(docs):
    raise FileNotFoundError("Unable to guess where the documentation is in '{0}' (2)".format(docs))


def copy_missing_dll():
    """
    Copies missing DLL.
    """
    rootpkg = os.path.normpath(os.path.join(this, "..", "..", "machinelearning", "packages"))
              
    misses = []

    # ML.net
    misses += [os.path.join(this, "..", "..", "machinelearning", "dist", "Debug")]
    
    # dependencies
    misses += [os.path.join(rootpkg, "newtonsoft.json", "10.0.3", "lib", "netstandard1.3")]
    misses += [os.path.join(rootpkg, "system.memory", "4.5.1", "lib", "netstandard2.0")]
    misses += [os.path.join(rootpkg, "system.runtime.compilerservices.unsafe", "4.5.0", "lib", "netstandard2.0")]
    misses += [os.path.join(rootpkg, "system.collections.immutable", "1.5.0", "lib", "netstandard2.0")]
    misses += [os.path.join(rootpkg, "system.numerics.vectors", "4.4.0", "lib", "netstandard2.0")]
    misses += [os.path.join(rootpkg, "google.protobuf", "3.5.1", "lib", "netstandard1.0")]

    skipif = ['testhost', 'TestPlatform']
    for miss in misses:
        cont = True
        for skip in skipif:
            if skip.lower() in miss.lower():
                cont = False
                break
        if not cont:
            continue
        
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
                if "TestPlatform" in dl:
                    continue
                if not os.path.exists(dst):
                    print("2>copy '{0}' from '{1}'".format(dl, miss))
                    shutil.copy(os.path.join(miss, dl), dll)

def copy_missing_md_docs(source):
    """
    Copies missing markdown documentation.
    """
    try:
        from .sphinx_mlext_templates import mddocs_index_template_docs, mddocs_index_template_releases
    except (ModuleNotFoundError, ImportError):
        from sphinx_mlext_templates import mddocs_index_template_docs, mddocs_index_template_releases

    dest = os.path.join(os.path.dirname(__file__), "mlnetdocs")
    if not os.path.exists(dest):
        os.mkdir(dest)
    rel = os.path.join(dest, "releases")
    if not os.path.exists(rel):
        os.mkdir(rel)
            
    # code
    docs = []
    code = os.path.join(source, "code")
    for name_ in os.listdir(code):
        name = name_.lower()
        docs.append(os.path.splitext(name)[0])
        print("3> copy '{0}'".format(name))
        dst = os.path.join(dest, name)
        shutil.copy(os.path.join(code, name), dst)
    code = os.path.join(source, "specs")
    for name_ in os.listdir(code):
        name = name_.lower()
        docs.append(os.path.splitext(name)[0])
        print("3> copy '{0}'".format(name))
        dst = os.path.join(dest, name)
        shutil.copy(os.path.join(code, name), dst)
    
    # release notes
    releases = []
    rele = os.path.join(source, "release-notes")
    for sub in os.listdir(rele):
        for name_ in os.listdir(os.path.join(rele, sub)):
            name = name_.lower().replace(".md", "").replace(".", "").replace("-", "") + ".md"
            releases.append(os.path.splitext(name)[0])
            print("3> copy '{0}'".format(name))
            dst = os.path.join(rel, name)
            shutil.copy(os.path.join(rele, sub, name_), dst)
    
    tpl = jinja2.Template(mddocs_index_template_docs)
    page = tpl.render(docs=docs)
    with open(os.path.join(dest, "index.rst"), "w", encoding="utf-8") as f:
        f.write(page)    

    tpl = jinja2.Template(mddocs_index_template_releases)
    page = tpl.render(releases=releases)
    with open(os.path.join(dest, "changes.rst"), "w", encoding="utf-8") as f:
        f.write(page)    


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
    MamlHelper.TestScikitAPITrain(iris)


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
    kinds += ["argument", "command"]
    kinds = list(set(kinds))
    titles = {
        'anomalydetectortrainer': 'Anomaly Detection', 
        'binaryclassifiertrainer': 'Binary Classification', 
        'clusteringtrainer': 'Clustering', 
        'dataloader': 'Data Loader', 
        'datasaver': 'Data Saver', 
        'datascorer': 'Scoring',
        'datatransform': 'Transforms (all)',
        'ensembledataselector': 'Data Selection',
        'evaluator': 'Evaluation',
        'multiclassclassifiertrainer': 'Multiclass Classification',
        'ngramextractorfactory': 'N-Grams',
        'rankertrainer': 'Ranking',
        'regressortrainer': 'Regression', 
        'tokenizetransform': 'Tokenization',
        'argument': 'Arguments',
        'command': 'Commands',
    }
    return {k: titles[k] for k in kinds if k in titles}


def builds_components_pages(epkg):
    """
    Returns components pages.
    """
    try:
        from .sphinx_mlext_templates import index_template, kind_template, component_template
    except (ModuleNotFoundError, ImportError):
        from sphinx_mlext_templates import index_template, kind_template, component_template
    try:
        from .machinelearning_docs import components
    except (ModuleNotFoundError, ImportError):
        from machinelearning_docs import components
    try:
        from pyquickhelper.texthelper import add_rst_links
    except ImportError:
        warnings.warn("Update pyquickhelper to a newer version.")
    
    if "OPTICS" not in epkg:
        raise KeyErro("OPTICS not found in epkg")
    
    def process_default(default_value):
        if not default_value:
            return ''
        if "+" in default_value:
            default_value = default_value.split(".")[-1].replace("+", ".")
            return default_value
        if len(default_value) > 28:            
            if len(default_value.split(".")) > 2:
                default_value = default_value.replace(".", ". ")
            elif len(default_value.split(",")) > 2:
                default_value = default_value.replace(",", ", ")
            else:
                raise ValueError("Unable to shorten default value '{0}' len={1}.".format(default_value, len(default_value)))
        return default_value
        
    def process_description(desc):
        if desc is None:
            return ''
        if not isinstance(desc, str):
            raise TypeError("desc must be a string not {0}".format(type(desc)))
        return add_rst_links(desc, epkg)
    
    kinds = mlnet_components_kinds()
    pages = {}
    
    # index
    sorted_kinds = list(sorted((v, k) for k, v in kinds.items()))
    template = jinja2.Template(index_template)
    pages["index"] = template.render(sorted_kinds=sorted_kinds)
    
    kind_tpl = jinja2.Template(kind_template)
    comp_tpl = jinja2.Template(component_template)
    
    # builds references
    refs = {}
    for v, k in sorted_kinds:
        enumc = MamlHelper.EnumerateComponents(k)
        try:
            comps = list(enumc)
        except Exception as e:
            print("Issue with kind '{0}'\n{1}".format(k, e))
            continue
        if len(comps) == 0:
            print("Empty kind '{0}'\n{1}".format(k, e))
            continue
        for comp in comps:
            refs[comp.Name] = ":ref:`l-{0}`".format(comp.Name.lower().replace(".", "-"))
    
    # kinds and components
    for v, k in sorted_kinds:
        enumc = MamlHelper.EnumerateComponents(k)
        try:
            comps = list(enumc)
        except Exception as e:
            print("Issue with kind '{0}'\n{1}".format(k, e))
            continue
        if len(comps) == 0:
            print("Empty kind '{0}'\n{1}".format(k, e))
            continue
            
        comp_names = list(sorted(c.Name.replace(" ", "_").replace(".", "_").lower() for c in comps))
        kind_name = v
        kind_kind = k
        pages[k] = kind_tpl.render(title=kind_name, fnames=comp_names, len=len)
        
        for comp in comps:
            
            if comp.Arguments is None and "version" not in comp.Name.lower():
                print("---- SKIP ----", k, comp.Name, comp.Description)
            else:
                assembly_name = comp.AssemblyName
                args = {}
                if comp.Arguments is not None:
                    for arg in comp.Arguments:
                        dv = process_default(arg.DefaultValue)
                        args[arg.Name] = dict(Name=arg.Name, ShortName=arg.ShortName or '',
                                          Default=refs.get(dv, dv), Description=arg.Help)
                sorted_params = [v for k, v in sorted(args.items())]
                aliases = ", ".join(comp.Aliases)

                if assembly_name.startswith("Microsoft.ML"):
                    linkdocs = "**Microsoft Documentation:** `{0} <https://docs.microsoft.com/dotnet/api/{1}.{2}>`_"
                    linkdocs = linkdocs.format(comp.Name, comp.Namespace.lower(), comp.Name.lower())
                else:
                    linkdocs = ""


                comp_name = comp.Name.replace(" ", "_").replace(".", "_").lower()
                pages[comp_name] = comp_tpl.render(title=comp.Name,
                                        aliases=aliases, 
                                        summary=process_description(comp.Description),
                                        kind=kind_kind, 
                                        namespace=comp.Namespace,
                                        sorted_params=sorted_params,
                                        assembly=assembly_name,
                                        len=len, linkdocs=linkdocs,
                                        docadd=components.get(comp.Name, ''),
                                        MicrosoftML="Microsoft.ML" in assembly_name,
                                        ScikitML="Scikit.ML" in assembly_name)
    
    return pages
    

def write_components_pages(app, env, docnames):
    """
    Writes documentation pages.
    """
    pages = builds_components_pages(app.config.epkg_dictionary)
    docdir = env.srcdir
    dest = os.path.join(docdir, "components")
    if not os.path.exists(dest):
        os.mkdir(dest)
    for k, v in pages.items():
        d = os.path.join(dest, k) + ".rst"
        if os.path.exists(d):
            with open(d, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            content = None
            
        if content != v:
            with open(d, "w", encoding="utf-8") as f:
                f.write(v)


def get_mlnet_assemblies(chdir=False):
    """
    Retrieves assemblies.
    """
    if chdir:
        cur = os.getcwd()
        os.chdir(dll)
    res = MamlHelper.GetLoadedAssembliesLocation(True)
    if chdir:
        os.chdir(cur)
    dependencies = []
    # addition = ["Core", "Data", "Maml", "Api"]
    # root = os.path.dirname(res[0].Location)
    # dependencies = [os.path.join(root, "Microsoft.ML.{0}.dll").format(a) for a in addition]
    dependencies.extend([a for a in res if ".pyd" not in a and ".so" not in a])
    usings = ["System", "System.Linq", "System.Collections.Generic", "System.IO",
              "System.Text"]
    usings.extend([
            "Microsoft.ML.Runtime",
            "Microsoft.ML.Runtime.Api",
            "Microsoft.ML.Runtime.Data",
            "Microsoft.ML.Runtime.Learners",
            "Microsoft.ML.Runtime.Ensemble",
            "Microsoft.ML.Runtime.LightGBM",
            "Microsoft.ML.Runtime.Model.Onnx",
            "Microsoft.ML.Runtime.TimeSeriesProcessing",
            "Microsoft.ML.Runtime.Tools",
            "Microsoft.ML.Trainers",
            "Microsoft.ML.Trainers.HalLearners",
            "Microsoft.ML.Trainers.KMeans",
            "Microsoft.ML.Trainers.FastTree",
            "Microsoft.ML.Trainers.Online",
            "Microsoft.ML.Trainers.PCA",
            "Microsoft.ML.Transforms",
            "Microsoft.ML.Transforms.Categorical",
            "Microsoft.ML.Transforms.Normalizers",
            "Microsoft.ML.Transforms.Projections",
            "Microsoft.ML.Transforms.TensorFlow",
            "Microsoft.ML.Transforms.Text",
            "Microsoft.ML.Runtime.Sweeper",
        ])
    res = MamlHelper.GetAssemblies()
    usings.extend([a.FullName.split(',')[0] for a in res if "Scikit" in a.FullName])
    return dependencies, usings
    

class RunCSharpMLDirective(RunCSharpDirective):
    """
    Implicits "and dependencies.
    """

    def modify_script_before_running(self, script):
        """
        The methods modifies the script to *csharpy* to
        run :epkg:`C#` from :epkg:`Python`.
        """
        if not hasattr(RunCSharpDirective, 'deps_using'):
            RunCSharpDirective.deps_using = get_mlnet_assemblies()
        dependencies, usings = RunCSharpMLDirective.deps_using
        return self._modify_script_before_running(script, usings, dependencies)
    

def setup(app):
    """
    Adds the custom directive.
    """
    copy_missing_md_docs(docs)
    copy_missing_dll()
    app.add_directive('mlcmd', MlCmdDirective)
    app.connect("env-before-read-docs", write_components_pages)
    app.add_directive('runcsharpml', RunCSharpMLDirective)
    return {'version': sphinx.__display_version__, 'parallel_read_safe': True}


if __name__ == "__main__":
    copy_missing_md_docs(docs)
    copy_missing_dll()
    from clr import AddReference
    AddReference('Scikit.ML.DocHelperMlExt')
    from Scikit.ML.DocHelperMlExt import MamlHelper
    class dummy:
        pass
    deps, uss = get_mlnet_assemblies()
    
    app = dummy()
    app.config = dummy()
    app.config.epkg_dictionary = {"OPTICS": "http://OPTICS"}
    app.env = dummy()
    app.env.srcdir = os.path.dirname(__file__)
    write_components_pages(app, app.env, None)
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
