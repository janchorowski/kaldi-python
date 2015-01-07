'''
Created on Aug 14, 2014

@author: chorows
'''

import os
import sys
import argparse

#import __main__

class AddConfig(argparse.Action):
    def __init__(self, *args, **kwargs):
        argparse.Action.__init__(self, *args, **kwargs)
        
    def __call__(self, parser, namespace, values, option_string=None):
        with open(values,'r') as f:
            opts = [l.split('#')[0].strip() for l in f]
        parser.parse_args(args=opts, namespace=namespace) 

class KaldiArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs['add_help']=False
        #kwargs['fromfile_prefix_chars']='--config='
        version = kwargs.pop('version', None)
        super(KaldiArgumentParser, self).__init__(*args, **kwargs)
        self.version = version
        
    def add_standard_arguments(self):
        grp = self.add_argument_group('Standard options')

        default_prefix = '-' 
        grp.add_argument(
            default_prefix+'h', default_prefix*2+'help',
            action='help', default=argparse.SUPPRESS,
            help=argparse._('show this help message and exit'))
        if self.version:
            grp.add_argument(
                default_prefix+'v', default_prefix*2+'version',
                action='version', default=argparse.SUPPRESS,
                version=self.version,
                help=argparse._("show program's version number and exit"))
        grp.add_argument('--print-args', type=bool, default=True, help='Print the command line arguments (to stderr)')
        #grp.add_argument('--config', action=AddConfig, help='Configuration file with options')
        grp.add_argument('--config', default=argparse.SUPPRESS, help='Configuration file with options')
    
    
    def parse_known_args(self, args=None, namespace=None):
        if args is None:
            args = sys.argv[1:]
        expanded_args = []
        
        next_arg_is_conf = False
        conf_file = None
        
        for arg in args:
            if arg.startswith('--config') or next_arg_is_conf:
                if next_arg_is_conf:
                    conf_file = arg
                elif arg.startswith('--config='):
                    conf_file = arg[9:].strip() #eat --config=
                else:
                    next_arg_is_conf = True
                if conf_file:
                    with open(conf_file,'r') as f:
                        expanded_args.extend(l.split('#')[0].strip() for l in f)
                    next_arg_is_conf = False
                    conf_file = None
            else:
                expanded_args.append(arg)
        return argparse.ArgumentParser.parse_known_args(self, args=expanded_args, namespace=namespace)
    
    def parse_args(self, args=None, namespace=None):
        args = argparse.ArgumentParser.parse_args(self, args=args, namespace=namespace)
        if args.print_args:
            print >>sys.stderr, os.path.basename(sys.argv[0]), " ".join(sys.argv[1:])
        return args
