__version__ = "0.3.0.dev"

import os
from .bootstrap import _logging, setup_terrier, setup_java_bridge

import importlib

#sub modules
anserini = None
cache = None
index = None
io = None
model = None
pipelines = None
rewrite = None
transformer = None

file_path = os.path.dirname(os.path.abspath(__file__))
firstInit = False
ApplicationSetup = None
IndexFactory = None
IndexRef = None
properties = None
tqdm = None
HOME_DIR = None
use_jpype = None

def init(version=None, mem=None, packages=[], jvm_opts=[], redirect_io=True, logging='WARN', home_dir=None, boot_packages=[], tqdm=None, java_bridge='jpype'):
    """
    Function necessary to be called before Terrier classes and methods can be used.
    Loads the Terrier.jar file and imports classes. Also finds the correct version of Terrier to download if no version is specified.

    Args:
        version(str): Which version of Terrier to download. Default=None.
            If None, find the newest Terrier released version in Maven Central and download it.
            If "snapshot", will download the latest build from Jitpack.
        mem(str): Maximum memory allocated for the Java virtual machine heap in MB. Corresponds to java -Xmx commandline argument. Default is 1/4 of physical memory.
        boot_packages(list(str)): Extra maven package coordinates files to load before starting Java. Default=[]. More information at https://github.com/terrier-org/terrier-core/blob/5.x/doc/terrier_develop.md
        packages(list(str)): Extra maven package coordinates files to load, using the Terrier classloader. Default=[]. More information at https://github.com/terrier-org/terrier-core/blob/5.x/doc/terrier_develop.md
        jvm_opts(list(str)): Extra options to pass to the JVM. Default=[].
            For instance, you may enable Java assertions by setting jvm_opts=['-ea']
        redirect_io(boolean): If True, the Java System.out and System.err will be redirected to Pythons sys.out and sys.err. Default=True.
        logging(str): the logging level to use.
                      Can be one of 'INFO', 'DEBUG', 'TRACE', 'WARN', 'ERROR'. The latter is the quietest.
                      Default='WARN'.
        home_dir(str): the home directory to use. Default to PYTERRIER_HOME environment variable.
        tqdm: The tqdm instance to use for progress bars. Defaults to tqdm.tqdm
        use_jpype(boolean): If True then JPype will be used as the python-java glue, else PyJnius will be used. Defaults to True
    """
    set_tqdm(tqdm)

    global ApplicationSetup
    global properties
    global firstInit
    global file_path
    global HOME_DIR
    global use_jpype

    if java_bridge.lower() == 'jpype':
        use_jpype = True
    elif java_bridge.lower() == 'pyjnius':
        use_jpype = False
    else:
        raise RuntimeError("{} is not a valid java bridge for PyTerrier, it only supports 'jpype' or 'pyjnius'".format(java_bridge))

    # we keep a local directory
    if home_dir is not None:
        HOME_DIR = home_dir
    if "PYTERRIER_HOME" in os.environ:
        HOME_DIR = os.environ["PYTERRIER_HOME"]
    else:
        from os.path import expanduser
        userhome = expanduser("~")
        HOME_DIR = os.path.join(userhome, ".pyterrier")
        if not os.path.exists(HOME_DIR):
            os.mkdir(HOME_DIR)

    # get the initial classpath for the JVM
    classpathTrJars = setup_terrier(HOME_DIR, version, boot_packages=boot_packages)

    if use_jpype:
        # Import jpype and other classes
        import jpype
        from jpype.types import JClass, JObject

        if mem is not None:
            jvm_opts.append('-Xmx' + str(mem) + 'm')
        # jvm_opts is in format ready to give as varargs to startJVM method
        # We want convertStrings to be true so that we don't have explicitly cast returned java strings when needed
        jpype.startJVM(convertStrings=True, classpath=classpathTrJars, *jvm_opts)

        # we only accept Java version 11 and newer; so anything starting 1. or 9. is too old
        java_version = JClass("java.lang.System").getProperty("java.version")
        if java_version.startswith("1.") or java_version.startswith("9."):
            raise RuntimeError("Pyterrier requires Java 11 or newer, we only found Java version %s;"
                +" install a more recent Java, or change os.environ['JAVA_HOME'] to point to the proper Java installation",
                java_version)

        properties = JClass('java.util.Properties')()
        ApplicationSetup = JClass('org.terrier.utility.ApplicationSetup')

        # Alias Jpype class and cast functions
        globals()["Class"] = JClass
        globals()["Cast"] = JObject

        # Make java class import global
        globals()["IndexFactory"] = JClass("org.terrier.structures.IndexFactory")
        globals()["IndexRef"] = JClass("org.terrier.querying.IndexRef")
    else:
        # Import pyjnius and other classes
        import jnius_config
        for jar in classpathTrJars:
            jnius_config.add_classpath(jar)
        if jvm_opts is not None:
            for opt in jvm_opts:
                jnius_config.add_options(opt)
        if mem is not None:
            jnius_config.add_options('-Xmx' + str(mem) + 'm')
        from jnius import autoclass, cast

        # we only accept Java version 11 and newer; so anything starting 1. or 9. is too old
        java_version = autoclass("java.lang.System").getProperty("java.version")
        if java_version.startswith("1.") or java_version.startswith("9."):
            raise RuntimeError("Pyterrier requires Java 11 or newer, we only found Java version %s;"
                               + " install a more recent Java, or change os.environ['JAVA_HOME'] to point to the proper Java installation",
                               java_version)

        properties = autoclass('java.util.Properties')()
        ApplicationSetup = autoclass('org.terrier.utility.ApplicationSetup')

        def jnius_cast(value, cast_type):
            return cast(cast_type, value)

        # Alias Jpype class and cast functions
        globals()["Class"] = autoclass
        globals()["Cast"] = jnius_cast

        # Make java class import global
        globals()["IndexFactory"] = autoclass("org.terrier.structures.IndexFactory")
        globals()["IndexRef"] = autoclass("org.terrier.querying.IndexRef")

    from .batchretrieve import BatchRetrieve, FeaturesBatchRetrieve
    from .utils import Utils
    from .datasets import get_dataset, list_datasets
    from .index import Indexer, FilesIndexer, TRECCollectionIndexer, DFIndexer, DFIndexUtils, IterDictIndexer, FlatJSONDocumentIterator, IndexingType
    from .pipelines import LTR_pipeline, XGBoostLTR_pipeline, Experiment

    # Make imports global
    globals()["ApplicationSetup"] = ApplicationSetup

    
    global anserini
    global cache
    global index
    global io
    global model
    global pipelines
    global rewrite
    global transformer
    anserini = importlib.import_module('.anserini', package='pyterrier') 
    cache = importlib.import_module('.cache', package='pyterrier')
    index = importlib.import_module('.index', package='pyterrier') 
    io = importlib.import_module('.io', package='pyterrier')
    model = importlib.import_module('.model', package='pyterrier')
    pipelines = importlib.import_module('.pipelines', package='pyterrier') 
    rewrite = importlib.import_module('.rewrite', package='pyterrier')
    transformer = importlib.import_module('.transformer', package='pyterrier')

    # append the python helpers
    if packages is None:
        packages = []

    # Import other java packages
    if packages != []:
        pkgs_string = ",".join(packages)
        properties.put("terrier.mvn.coords", pkgs_string)
    ApplicationSetup.bootstrapInitialisation(properties)

    if redirect_io:
        # this ensures that the python stdout/stderr and the Java are matched
        redirect_stdouterr()
    _logging(logging, use_jpype)
    setup_java_bridge(use_jpype)

    globals()["get_dataset"] = get_dataset
    globals()["list_datasets"] = list_datasets
    globals()["Experiment"] = Experiment
    globals()["BatchRetrieve"] = BatchRetrieve
    globals()["Indexer"] = Indexer
    globals()["FeaturesBatchRetrieve"] = FeaturesBatchRetrieve
    globals()["TRECCollectionIndexer"] = TRECCollectionIndexer
    globals()["FilesIndexer"] = FilesIndexer
    globals()["DFIndexer"] = DFIndexer
    globals()["DFIndexUtils"] = DFIndexUtils
    globals()["IterDictIndexer"] = IterDictIndexer
    globals()["FlatJSONDocumentIterator"] = FlatJSONDocumentIterator
    globals()["Utils"] = Utils
    globals()["LTR_pipeline"] = LTR_pipeline
    globals()["XGBoostLTR_pipeline"] = XGBoostLTR_pipeline
    globals()["IndexingType"] = IndexingType

    firstInit = True

def set_tqdm(type):
    global tqdm
    
    if type is None or type == 'tqdm':
        from tqdm import tqdm as bartype
        tqdm = bartype
    elif type == 'notebook':
        from tqdm.notebook import tqdm as bartype
        tqdm = bartype
    elif type == 'auto':
        from tqdm.auto import tqdm as bartype
        tqdm = bartype
    else:
        raise ValueError("Unknown tqdm type %s" % str(type))
    

def started():
    return(firstInit)

def version():
    if use_jpype:
        from jpype import JClass
        return JClass("org.terrier.Version").VERSION
    else:
        from jnius import autoclass
        return autoclass("org.terrier.Version").VERSION

def check_version(min):
    from packaging.version import Version
    min = Version(str(min))
    currentVer = Version(version().replace("-SNAPSHOT", ""))
    return currentVer >= min

def redirect_stdouterr():
    from . import bootstrap
    bootstrap.redirect_stdouterr(use_jpype)

def logging(level):
    from . import bootstrap
    bootstrap.logging(level, use_jpype)

def set_property(k, v):
    # properties = Properties()
    properties[k] = v
    ApplicationSetup.bootstrapInitialisation(properties)

def set_properties(kwargs):
    # properties = Properties()
    for control, value in kwargs.items():
        properties.put(control, value)
    ApplicationSetup.bootstrapInitialisation(properties)

def run(cmd, args=[]):
    from jpype import JClass
    JClass("org.terrier.applications.CLITool").main([cmd] + args)

def extend_classpath(mvnpackages):
    assert check_version(5.3), "Terrier 5.3 required for this functionality"
    if isinstance(mvnpackages, str):
        mvnpackages = [mvnpackages]
    if use_jpype:
        from jpype.types import JClass, JObject
        thelist = JClass("java.util.ArrayList")()
        for pkg in mvnpackages:
            thelist.add(pkg)
        mvnr = ApplicationSetup.getPlugin("MavenResolver")
        assert mvnr is not None
        mvnr = JObject(mvnr, "org.terrier.utility.MavenResolver")
    else:
        from jnius import autoclass, cast
        thelist = autoclass("java.util.ArrayList")()
        for pkg in mvnpackages:
            thelist.add(pkg)
        mvnr = ApplicationSetup.getPlugin("MavenResolver")
        assert mvnr is not None
        mvnr = cast("org.terrier.utility.MavenResolver", mvnr)
    mvnr.addDependencies(thelist)
