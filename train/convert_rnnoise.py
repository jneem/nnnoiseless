#!/usr/bin/python

# This script is for converting rnnoise's text-based model file format to
# nnnoiseless's binary model file format.
#
# Usage: `convert_rnnoise.py INPUT OUTPUT`

import sys

if len(sys.argv) != 3:
    print("Expected two arguments.")
    print("USAGE: convert_rnnoise.py INPUT OUTPUT")
    sys.exit(1)

in_name = sys.argv[1]
out_name = sys.argv[2]

with open(in_name, 'r') as in_file:
    if in_file.readline().strip() != 'rnnoise-nu model file version 1':
        print("Unexpected input file format")
        sys.exit(1)

    data = in_file.read()
    in_nums = [int(s) for s in data.split()]

    # Convert to an unsigned byte (which, in twos complement, represents the same number)
    def cvt(n):
        if n < 0:
            return 256 + n
        else:
            return n

    nums = [cvt(n) for n in in_nums]

    with open(out_name, 'wb') as out_file:
        out_file.write(bytearray(nums))

        print(f'Converted {in_name} to {out_name}')
