import os
import sys
import shutil


def main(args):
    """
    Copy binaires.
    """
    if len(args) != 3:
        print("Usage copy_binaries.py Configuration destination")
        return
    conf = args[1]
    if conf not in ['Debug', 'Release']:
        raise ValueError("Unknown configuration '{0}'".format(conf))
    dest = args[2]
    if not os.path.exists(dest):
        os.makedirs(dest)
    
    folder = os.path.join("machinelearningext")
    names = {}
    file, rep = [], []
    for r, d, f in os.walk(folder):
        for a in f:
            full = os.path.join(r, a)
            ext = os.path.splitext(a)[-1]
            if ext not in {'.so', '.dll', '.pdb', '.json'}:
                continue
            if "Scikit.ML" not in a:
                continue
            last_name = os.path.split(a)[-1]
            if last_name not in names:
                names[last_name] = full
    
    for _, name in sorted(names.items()):
        print("copy '{0}'".format(name))
        shutil.copy(name, dest)
    
    shutil.copy(os.path.join("machinelearning", "BuildToolsVersion.txt"), dest)
    shutil.copy(os.path.join("machinelearning", "THIRD-PARTY-NOTICES.TXT"), dest)


if __name__ == "__main__":
    args = sys.argv
    main(args)
    