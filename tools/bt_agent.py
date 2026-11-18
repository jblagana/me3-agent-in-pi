#!/usr/bin/python3
"""Boots Bluetooth pairing agent (2026-11-18). Runs as root with the system Python
(python3-dbus, python3-gi).

Boots has no screen to confirm a pairing, so:
  * Normally the adapter is NOT pairable and NOT discoverable: nobody new can
    pair. Devices paired before reconnect on their own (they are trusted).
  * Pairing mode (PAIR_WINDOW_S, 3 min) starts when something touches
    /run/boots-bt/pair: the dashboard's Bluetooth button or the voice tool.
    During it Boots shows up as "Boots" and the first confirmations are
    accepted ("Just Works", no PIN); each device paired then is trusted.
  * Only audio (A2DP) and remote control (AVRCP) are authorized, and only for
    paired devices. Calls, file transfer, networking and the like are refused.

Status for the dashboard goes to /run/boots-bt/status.json once a second.
"""
import json
import os
import time

import dbus
import dbus.mainloop.glib
import dbus.service
from gi.repository import GLib

BUS = "org.bluez"
AGENT_PATH = "/boots/agent"
RUN = os.environ.get("BOOTS_BT_RUN", "/run/boots-bt")
TRIGGER = os.path.join(RUN, "pair")
STATUS = os.path.join(RUN, "status.json")
PAIR_WINDOW_S = int(os.environ.get("BOOTS_BT_PAIR_S", "180"))
NAME = os.environ.get("BOOTS_BT_NAME", "Boots")
ALLOWED_UUIDS = {
    "0000110b-0000-1000-8000-00805f9b34fb",   # A2DP audio sink
    "0000110d-0000-1000-8000-00805f9b34fb",   # A2DP (advanced audio distribution)
    "0000110c-0000-1000-8000-00805f9b34fb",   # AVRCP target
    "0000110e-0000-1000-8000-00805f9b34fb",   # AVRCP controller
    "0000110f-0000-1000-8000-00805f9b34fb",   # AVRCP controller (newer)
}

STATE = {"until": 0.0, "trigger_mtime": None, "last_status": None}


class Rejected(dbus.DBusException):
    _dbus_error_name = "org.bluez.Error.Rejected"


def bus():
    return dbus.SystemBus()


def managed():
    om = dbus.Interface(bus().get_object(BUS, "/"), "org.freedesktop.DBus.ObjectManager")
    return om.GetManagedObjects()


def adapter_path():
    for path, ifaces in managed().items():
        if "org.bluez.Adapter1" in ifaces:
            return path
    return None


def props(path):
    return dbus.Interface(bus().get_object(BUS, path), "org.freedesktop.DBus.Properties")


def set_adapter(**kw):
    p = props(adapter_path())
    for k, v in kw.items():
        p.Set("org.bluez.Adapter1", k, v)


def pairing():
    return time.time() < STATE["until"]


def device_props(path):
    return props(path).GetAll("org.bluez.Device1")


def trust(path):
    props(path).Set("org.bluez.Device1", "Trusted", dbus.Boolean(True))


class Agent(dbus.service.Object):
    @dbus.service.method("org.bluez.Agent1", in_signature="", out_signature="")
    def Release(self):
        pass

    @dbus.service.method("org.bluez.Agent1", in_signature="os", out_signature="")
    def AuthorizeService(self, device, uuid):
        d = device_props(device)
        if bool(d.get("Paired")) and str(uuid).lower() in ALLOWED_UUIDS:
            return
        print(f"refused service {uuid} for {d.get('Alias')}", flush=True)
        raise Rejected("service not allowed")

    def _accept_pairing(self, device):
        if not pairing():
            print(f"refused pairing from {device} (pairing mode is off)", flush=True)
            raise Rejected("pairing mode is off")
        trust(device)
        print(f"pairing accepted: {device_props(device).get('Alias')}", flush=True)

    @dbus.service.method("org.bluez.Agent1", in_signature="o", out_signature="s")
    def RequestPinCode(self, device):
        self._accept_pairing(device)
        return "0000"

    @dbus.service.method("org.bluez.Agent1", in_signature="o", out_signature="u")
    def RequestPasskey(self, device):
        self._accept_pairing(device)
        return dbus.UInt32(0)

    @dbus.service.method("org.bluez.Agent1", in_signature="ouq", out_signature="")
    def DisplayPasskey(self, device, passkey, entered):
        pass

    @dbus.service.method("org.bluez.Agent1", in_signature="os", out_signature="")
    def DisplayPinCode(self, device, pincode):
        pass

    @dbus.service.method("org.bluez.Agent1", in_signature="ou", out_signature="")
    def RequestConfirmation(self, device, passkey):
        self._accept_pairing(device)

    @dbus.service.method("org.bluez.Agent1", in_signature="o", out_signature="")
    def RequestAuthorization(self, device):
        self._accept_pairing(device)

    @dbus.service.method("org.bluez.Agent1", in_signature="", out_signature="")
    def Cancel(self):
        pass


def start_pairing():
    STATE["until"] = time.time() + PAIR_WINDOW_S
    set_adapter(Pairable=dbus.Boolean(True), PairableTimeout=dbus.UInt32(PAIR_WINDOW_S),
                Discoverable=dbus.Boolean(True), DiscoverableTimeout=dbus.UInt32(PAIR_WINDOW_S))
    print(f"pairing mode on for {PAIR_WINDOW_S} s", flush=True)


def stop_pairing():
    STATE["until"] = 0.0
    try:
        set_adapter(Pairable=dbus.Boolean(False), Discoverable=dbus.Boolean(False))
    except Exception:
        pass
    print("pairing mode off", flush=True)


def tick():
    try:
        try:
            m = os.stat(TRIGGER).st_mtime
        except FileNotFoundError:
            m = None
        if m is not None and m != STATE["trigger_mtime"]:
            if STATE["trigger_mtime"] is not None or time.time() - m < 5:
                start_pairing()
            STATE["trigger_mtime"] = m
        if STATE["until"] and not pairing():
            stop_pairing()
        if not pairing():
            # Anything else that turns pairing on (bluetoothctl registers its
            # own agent and can flip Pairable) is switched back off.
            a = props(adapter_path()).GetAll("org.bluez.Adapter1")
            if a.get("Pairable") or a.get("Discoverable"):
                set_adapter(Pairable=dbus.Boolean(False), Discoverable=dbus.Boolean(False))
                print("pairing was on outside pairing mode; switched off", flush=True)
        devs = [(p, i["org.bluez.Device1"]) for p, i in managed().items() if "org.bluez.Device1" in i]
        if pairing():               # trust whatever finished pairing in the window
            for p, d in devs:
                if d.get("Paired") and not d.get("Trusted"):
                    trust(p)
        status = {
            "pairing": pairing(),
            "seconds_left": max(0, int(STATE["until"] - time.time())),
            "connected": sorted(str(d.get("Alias")) for _, d in devs if d.get("Connected")),
            "paired": sorted(str(d.get("Alias")) for _, d in devs if d.get("Paired")),
            "name": NAME,
        }
        if status != STATE["last_status"]:
            tmp = STATUS + ".tmp"
            with open(tmp, "w") as f:
                json.dump(status, f)
            os.chmod(tmp, 0o644)
            os.replace(tmp, STATUS)
            STATE["last_status"] = status
    except Exception as e:
        print(f"tick error: {e!r}", flush=True)
    return True


def main():
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    os.makedirs(RUN, exist_ok=True)
    for _ in range(30):                       # wait for bluetoothd and the adapter
        try:
            if adapter_path():
                break
        except Exception:
            pass
        time.sleep(1)
    set_adapter(Powered=dbus.Boolean(True), Alias=NAME,
                Pairable=dbus.Boolean(False), Discoverable=dbus.Boolean(False))
    Agent(bus(), AGENT_PATH)
    mgr = dbus.Interface(bus().get_object(BUS, "/org/bluez"), "org.bluez.AgentManager1")
    mgr.RegisterAgent(AGENT_PATH, "NoInputNoOutput")
    mgr.RequestDefaultAgent(AGENT_PATH)
    print(f"bluetooth agent ready: '{NAME}', not pairable until pairing mode", flush=True)
    GLib.timeout_add(1000, tick)
    GLib.MainLoop().run()


if __name__ == "__main__":
    main()
