#!/usr/bin/python3
"""Set Boots's options in /etc/bluetooth/main.conf (2026-11-18). Idempotent.
Name and class make phones show "Boots" with a speaker icon; AutoEnable powers
the radio at boot. Pairing stays off except in the agent's pairing window."""
import re, shutil, sys
P = sys.argv[1] if len(sys.argv) > 1 else "/etc/bluetooth/main.conf"
WANT = {"General": {"Name": "Boots", "Class": "0x240414", "DiscoverableTimeout": "180",
                    "PairableTimeout": "180", "AlwaysPairable": "false"},
        "Policy": {"AutoEnable": "true"}}
s = open(P).read()
shutil.copy(P, P + ".bak-boots")
for sect, kv in WANT.items():
    for k, v in kv.items():
        pat = re.compile(r"^\s*#?\s*%s\s*=.*$" % re.escape(k), re.M)
        block = re.search(r"^\[%s\]\s*$(.*?)(?=^\[|\Z)" % sect, s, re.M | re.S)
        if block and pat.search(block.group(1)):
            body = pat.sub(f"{k} = {v}", block.group(1), count=1)
            s = s[:block.start(1)] + body + s[block.end(1):]
        else:
            s = re.sub(r"^\[%s\]\s*$" % sect, f"[{sect}]\n{k} = {v}", s, count=1, flags=re.M)
open(P, "w").write(s)
print("main.conf updated")
