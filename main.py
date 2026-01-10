#!/usr/bin/env python3
import sys
import argparse
from capture.record_pose import main as record_main   # adjust import as needed

def main():
    parser = argparse.ArgumentParser(description="Difo Human → Robot Motion Training")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Record command
    record = subparsers.add_parser("record", help="Record human motion")
    
    args = parser.parse_args()

    if args.command == "record":
        record_main()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()