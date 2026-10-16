"""One CUDA owner process per P/D worker. Control messages never carry tensors."""

import hashlib
import json
import os
import time
import traceback
from dataclasses import asdict

import torch

from vllm_rlt.engine.stage_config import StageConfig
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.models import AutoModelForCausalLM, OuroConfig, OuroForCausalLM
from vllm_rlt.profiling import ProfileConfig

from .connector import PDStageConnector
from .transport import NixlConnector


class PDWorker:
    def __init__(self, role, device, name, channel, model_source, options, pd_config):
        torch.set_num_threads(1)
        torch.cuda.set_device(device)
        torch.manual_seed(options["seed"])
        self.role, self.channel, self.config = role, channel, pd_config
        dtype = getattr(torch, options["dtype"])
        self.model = (
            OuroForCausalLM(model_source).to(device=f"cuda:{device}", dtype=dtype)
            if isinstance(model_source, OuroConfig)
            else AutoModelForCausalLM.from_pretrained(
                model_source,
                revision=options["revision"],
                device=f"cuda:{device}",
                dtype=dtype,
            )
        )
        self.engine = StageEngine(
            self.model, stage_config=StageConfig(stage_id=name, role=role), **options["engine"]
        )
        self.cache = self.engine.cache_manager
        self.active_limit = self.engine.scheduler.config.max_num_seqs
        self.limit = self.active_limit + (
            pd_config.max_receiving_requests
            if role == "decode"
            else pd_config.max_draining_requests
        )
        self.hidden = torch.empty(
            (self.limit, self.model.config.hidden_size), device=f"cuda:{device}", dtype=dtype
        )
        self.connector = NixlConnector(name, self.cache, self.hidden, pd_config)
        self.stage_connector = PDStageConnector(
            self.engine, self.hidden, self.connector, pd_config, notify=self.send
        )
        self.engine.connector = self.stage_connector
        self.running = True
        digest = hashlib.sha256()
        for parameter in self.model.state_dict().values():
            digest.update(parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
        self.fingerprint = hashlib.sha256(
            json.dumps(
                dict(
                    model=asdict(self.model.config),
                    weights=digest.hexdigest(),
                    dtype=options["dtype"],
                    layout=self.cache.layout,
                    attention=self.cache.attention_info,
                ),
                sort_keys=True,
            ).encode()
        ).hexdigest()
        self.send(
            "ready",
            info=self.connector.info,
            fingerprint=self.fingerprint,
            model=asdict(self.model.config),
            num_blocks=self.cache.num_blocks,
            block_size=self.cache.block_size,
            depths=self.cache.storage_depths,
            max_seqs=self.limit,
            active_limit=self.active_limit,
            watermark_blocks=self.cache.watermark_blocks,
        )

    def send(self, kind, **fields):
        self.channel.send(dict(kind=kind, **fields))

    def command(self, command):
        kind = command["kind"]
        if kind.startswith("profile_"):
            try:
                action = kind.removeprefix("profile_")
                if action == "start":
                    result = self.engine.start_profile(
                        ProfileConfig(**command["config"]),
                        scheduled=command["scheduled"],
                        session_id=command["session_id"],
                        capture_time=command["capture_time"],
                        rank=command["rank"],
                        role=self.role,
                        managed_manifest=False,
                    )
                else:
                    current = self.engine.profile_status().get("session_id")
                    if current is not None and current != command["session_id"]:
                        raise ValueError("profile session ID mismatch")
                    if action == "stop":
                        result = self.engine.stop_profile()
                    elif action == "status":
                        result = self.engine.profile_status()
                    else:
                        raise ValueError("unknown profile action")
                self.send(
                    "profile_reply",
                    control_id=command["control_id"],
                    session_id=command["session_id"],
                    result=result,
                )
            except Exception as exc:
                self.send(
                    "profile_reply",
                    control_id=command["control_id"],
                    session_id=command["session_id"],
                    error=str(exc),
                )
            return
        if kind == "connect":
            for peer in command["peers"]:
                self.connector.connect(peer)
            self.send("connected")
        elif kind in ("reserve", "prefill"):
            self.stage_connector.reserve(
                command["tid"],
                command["tokens"],
                command["params"],
                receiving=kind == "reserve",
                peer=command["peer"],
                trace_id=command.get("trace_id"),
                target_tables=command.get("tables", ()),
                target_slot=command.get("slot", 0),
                target_cached_tokens=command.get("cached_tokens", 0),
            )
        elif kind == "cancel":
            self.stage_connector.cancel(command["tid"])
        elif kind == "cancel_safe":
            self.stage_connector.cancel_safe(command["tid"])
        elif kind == "commit":
            self.stage_connector.commit(command["tid"], command["count"])
        elif kind == "ack":
            self.stage_connector.acknowledge(command["tid"])
        elif kind == "stop":
            if self.stage_connector.transfers or self.connector.pending:
                raise RuntimeError("stop requested before PD requests drained")
            self.running = False
        else:
            raise ValueError(f"unknown PD command {kind}")

    def run(self):
        while self.running:
            for _ in range(self.config.max_control_messages):
                if not self.channel.poll():
                    break
                self.command(self.channel.recv())
            self.stage_connector.progress()
            if self.engine.has_unfinished_requests():
                for output in self.engine.step():
                    self.send("output", tid=output.request_id, output=output)
                    if output.finished:
                        self.stage_connector.output_finished(output.request_id)
            if not self.stage_connector.transfers and not self.channel.poll():
                time.sleep(0.001)
        self.engine.close()
        self.connector.close()
        self.send(
            "stopped",
            prefill_tokens=self.stage_connector.prefill_tokens,
            bytes_sent=self.connector.bytes_sent,
            transfers=self.connector.transfers,
            transfer_seconds=self.connector.transfer_seconds,
            used_blocks=self.cache.num_used_blocks,
        )


def worker_main(role, device, name, channel, model_source, options, pd_config):
    # Keep owner reachable on fatal errors. The coordinator terminates senders
    # before receivers; no failed receiver reuses a potentially writable page.
    owner = None
    try:
        owner = PDWorker(role, device, name, channel, model_source, options, pd_config)
        owner.run()
    except BaseException:
        try:
            channel.send(dict(kind="fatal", error=traceback.format_exc(), pid=os.getpid()))
            while True:
                time.sleep(1)
        except (BrokenPipeError, EOFError):
            pass
        raise
    finally:
        channel.close()
