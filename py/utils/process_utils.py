from __future__ import absolute_import

def str2bool(v):
    # susendberg's function
    return v.lower() in ("yes", "true", "t", "1")


def display_args(args):
    arg_vars = vars(args)
    print("# ===============================================")
    print("## parameters: ")
    for arg_key in arg_vars.keys():
        if arg_key != 'func':
            print("{}:\n\t{}".format(arg_key, arg_vars[arg_key]))
    print("# ===============================================")
