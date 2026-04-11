#!/usr/bin/env python3
"""
Send MIDI messages to a running Cloudbuster instance via amidi.

SETUP (one-time per session):
  ./midi.py --list                    # find Cloudbuster's sequencer port
  aconnect <virmidi_port> <cloudbuster_port>
  e.g.: aconnect 32:0 128:0

EXAMPLES:
  ./midi.py note 60                   # play middle C (0.5s)
  ./midi.py note 60 100 -d 2          # note 60, velocity 100, 2s duration
  ./midi.py cc 7 100                  # raw CC
  ./midi.py volume 100
  ./midi.py stretch 64                # CC22, ~2.1x
  ./midi.py N 64                      # CC20, → N=1024
  ./midi.py hop 96                    # CC21, → hop_size_div=16
  ./midi.py attack 64
  ./midi.py decay 32
  ./midi.py sustain 100
  ./midi.py release 80
  ./midi.py next                      # next file (CC48)
  ./midi.py prev                      # prev file (CC47)
  ./midi.py roboto                    # toggle roboto
  ./midi.py whisper                   # toggle whisper
  ./midi.py alien                     # toggle alien
"""

import argparse
import re
import subprocess
import sys
import time

# Cloudbuster CC assignments
CC_MAP = {
    "volume":  7,
    "N":       20,   # stepped: 0-31→512, 32-63→1024, 64-95→2048, 96-127→4096
    "hop":     21,   # stepped: 0-31→2,   32-63→4,    64-95→8,    96-127→16
    "stretch": 22,   # continuous: 0→0.25, 127→4.0
    "release": 72,
    "attack":  73,
    "decay":   75,
    "sustain": 79,
    "roboto":  80,
    "whisper": 81,
    "alien":   82,
    "prev":    47,
    "next":    48,
}

TOGGLE_CMDS = {"roboto", "whisper", "alien", "prev", "next"}

VIRMIDI_CLIENT = "Virtual Raw MIDI"
CLOUDBUSTER_CLIENT = "RtMidi Input Client"


def autoconnect(virmidi_port="hw:4,0"):
    """Find VirMIDI sequencer port matching virmidi_port and connect it to Cloudbuster."""
    out = subprocess.run(["aconnect", "-l"], capture_output=True, text=True).stdout

    # Extract card number from amidi port, e.g. hw:4,0 → card 4
    m = re.match(r"hw:(\d+)", virmidi_port)
    card = m.group(1) if m else None

    src = dst = None
    current_client = None
    for line in out.splitlines():
        cm = re.match(r"client (\d+): '([^']+)'", line)
        if cm:
            current_client = (cm.group(1), cm.group(2))
        pm = re.match(r"\s+(\d+) '", line)
        if pm and current_client:
            port_id = f"{current_client[0]}:{pm.group(1)}"
            if card and VIRMIDI_CLIENT in current_client[1] and f"card={card}" in line:
                src = port_id
            elif VIRMIDI_CLIENT in current_client[1] and src is None:
                src = port_id
            elif CLOUDBUSTER_CLIENT in current_client[1]:
                dst = port_id

    if not dst:
        print("error: Cloudbuster (RtMidi Input Client) not found. Is it running?", file=sys.stderr)
        sys.exit(1)
    if not src:
        print(f"error: VirMIDI sequencer port not found for {virmidi_port}.", file=sys.stderr)
        sys.exit(1)

    result = subprocess.run(["aconnect", src, dst], capture_output=True, text=True)
    if result.returncode != 0 and "already subscribed" not in result.stderr:
        print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)
    print(f"connected {src} → {dst}", file=sys.stderr)


def send_bytes(port, *bytes_):
    hex_str = " ".join(f"{b:02X}" for b in bytes_)
    subprocess.run(["amidi", "-p", port, "-S", hex_str], check=True)


def note_on(port, note, velocity, channel=0):
    send_bytes(port, 0x90 | channel, note, velocity)


def note_off(port, note, channel=0):
    send_bytes(port, 0x80 | channel, note, 0)


def cc(port, number, value, channel=0):
    send_bytes(port, 0xB0 | channel, number, value)


def list_ports():
    print("Raw MIDI ports (for --port):")
    subprocess.run(["amidi", "-l"])
    print()
    print("ALSA sequencer clients (for aconnect):")
    subprocess.run(["aconnect", "-l"])


def main():
    parser = argparse.ArgumentParser(
        description="Send MIDI to Cloudbuster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--port", "-p", default="hw:4,0",
                        help="amidi port (default: hw:4,0, first VirMIDI)")
    parser.add_argument("--channel", "-c", type=int, default=0,
                        help="MIDI channel 0-15 (default: 0)")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List available MIDI ports and exit")
    parser.add_argument("--connect", action="store_true",
                        help="(Re)connect VirMIDI → Cloudbuster via aconnect and exit")

    sub = parser.add_subparsers(dest="cmd")

    p_note = sub.add_parser("note", help="Play a note (note-on + sleep + note-off)")
    p_note.add_argument("note", type=int, help="MIDI note number (0-127)")
    p_note.add_argument("velocity", type=int, nargs="?", default=100)
    p_note.add_argument("--duration", "-d", type=float, default=0.5,
                        help="Duration in seconds (default: 0.5)")

    p_cc = sub.add_parser("cc", help="Send a raw CC message")
    p_cc.add_argument("number", type=int, help="CC number (0-127)")
    p_cc.add_argument("value", type=int, help="CC value (0-127)")

    for name in CC_MAP:
        p = sub.add_parser(name, help=f"CC{CC_MAP[name]}")
        if name in TOGGLE_CMDS:
            p.add_argument("value", type=int, nargs="?", default=127)
        else:
            p.add_argument("value", type=int, help="Value 0-127")

    args = parser.parse_args()

    if args.list:
        list_ports()
        return

    if args.connect:
        autoconnect(args.port)
        return

    if not args.cmd:
        parser.print_help()
        return

    autoconnect(args.port)

    ch = args.channel
    p = args.port

    try:
        if args.cmd == "note":
            note_on(p, args.note, args.velocity, ch)
            time.sleep(args.duration)
            note_off(p, args.note, ch)
        elif args.cmd == "cc":
            cc(p, args.number, args.value, ch)
        elif args.cmd in CC_MAP:
            cc(p, CC_MAP[args.cmd], args.value, ch)
    except subprocess.CalledProcessError:
        print(f"error: amidi failed on port {p!r}. Run ./midi.py --list to see available ports.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
