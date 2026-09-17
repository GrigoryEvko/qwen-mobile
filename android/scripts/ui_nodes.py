"""Print the nodes of a uiautomator dump, one per line: text | resource-id | bounds.

The dump comes on stdin. Nodes without text and without an id are not printed.
"""

import re
import sys


def main() -> None:
    xml = sys.stdin.read()
    for m in re.finditer(r"<node[^>]*>", xml):
        node = m.group(0)
        text = re.search(r' text="([^"]*)"', node)
        rid = re.search(r' resource-id="([^"]*)"', node)
        bounds = re.search(r' bounds="(\[[^"]*\])"', node)
        t = text.group(1) if text else ""
        r = rid.group(1).split("/")[-1] if rid else ""
        if t or r:
            print(f"{t} | {r} | {bounds.group(1) if bounds else ''}")


if __name__ == "__main__":
    main()
