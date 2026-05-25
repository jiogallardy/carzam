"""Command-line entry: `epd show`, `epd clear`, `epd serve`."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .driver import Spectra6
from .image_prep import prepare_and_pack


def cmd_show(args):
    preview = args.preview
    packed = prepare_and_pack(
        args.path,
        fit=args.fit,
        rotate=args.rotate,
        preview_path=preview,
    )
    panel = Spectra6()
    try:
        print("initializing panel...", flush=True)
        panel.init()
        print("pushing frame + refreshing (~26s)...", flush=True)
        t0 = time.monotonic()
        panel.display(packed)
        print(f"done in {time.monotonic() - t0:.1f}s", flush=True)
    finally:
        panel.sleep()
        panel.close()


def cmd_clear(args):
    panel = Spectra6()
    try:
        panel.init()
        panel.clear(color=0x1)
    finally:
        panel.sleep()
        panel.close()


def cmd_serve(args):
    import uvicorn
    uvicorn.run("epd.api:app", host="0.0.0.0", port=args.port, log_level="info")


def cmd_prompt(args):
    import tempfile
    from .gen import generate_png
    tmp = Path(tempfile.mkstemp(suffix=".png")[1])
    print(f"generating image for: {args.subject!r} (title={args.title!r})...", flush=True)
    generate_png(
        subject=args.subject,
        title=args.title,
        setting=args.setting,
        out_path=tmp,
    )
    print(f"saved {tmp} ({tmp.stat().st_size} bytes), pushing to display...", flush=True)
    args.path = tmp
    args.fit = "letterbox"
    args.rotate = 0
    args.preview = None
    cmd_show(args)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="epd")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_show = sub.add_parser("show", help="display an image file")
    p_show.add_argument("path", type=Path)
    p_show.add_argument("--fit", choices=["letterbox", "cover"], default="letterbox")
    p_show.add_argument("--rotate", type=int, default=0)
    p_show.add_argument("--preview", type=Path, default=None,
                        help="save a preview PNG showing exactly what will display")
    p_show.set_defaults(func=cmd_show)

    p_clear = sub.add_parser("clear", help="clear panel to white")
    p_clear.set_defaults(func=cmd_clear)

    p_serve = sub.add_parser("serve", help="run HTTP push API")
    p_serve.add_argument("--port", type=int, default=8765)
    p_serve.set_defaults(func=cmd_serve)

    p_prompt = sub.add_parser("prompt", help="generate via Gemini and display")
    p_prompt.add_argument("subject", help="e.g. 'orange Porsche GT3RS'")
    p_prompt.add_argument("--title", default=None, help="badge title (defaults to subject upper-cased)")
    p_prompt.add_argument("--setting", default="european cobblestone")
    p_prompt.set_defaults(func=cmd_prompt)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
