"""CLI argument parsing and command dispatch."""

from __future__ import annotations

import argparse

from run import build_parser


def test_default_command_is_all():
    args = build_parser().parse_args([])
    assert args.command == "all"


def test_accepted_commands():
    for command in ("all", "data", "report", "dashboard", "check"):
        args = build_parser().parse_args([command])
        assert args.command == command


def test_unknown_command_is_rejected():
    with __import__("pytest").raises(SystemExit):
        build_parser().parse_args(["frobnicate"])
