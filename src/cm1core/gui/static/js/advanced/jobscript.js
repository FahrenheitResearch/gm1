// The job script for a cluster: gpuwm-cm1 jobscript RUNDIR --scheduler slurm|pbs. The script runs
// gpuwm-cm1 run RUNDIR --continue on the GPU node, so submitting it again goes on from the last restart, and
// --until extends the run; the two sidecars write the CM1 files and the pictures on that node's CPUs. Used by
// Estimate (the first submission) and Continue or extend (a longer run on a cluster).

import { h, copyline, copyButton } from "../core.js";
import * as api from "../api.js";
import { actionButton } from "../command.js";
import { resultBox, act, hint } from "../setup/common.js";

const TIME_RE = /^\d{1,3}:\d{2}:\d{2}$/;

function field(label, input, cm1) {
  return h("label", { class: "jsfield" }, h("span", { class: "muted" }, label, cm1 ? hint(cm1) : null), input);
}

// opts: {id: "jobscript", title, open: bool, until: () => seconds | null, note}
export function jobscriptPanel(runId, opts = {}) {
  const box = resultBox();
  const sched = h("select", { class: "input", "aria-label": "scheduler" },
    h("option", { value: "slurm" }, "Slurm"), h("option", { value: "pbs" }, "PBS"));
  const time = h("input", { class: "input", value: "04:00:00", "aria-label": "time limit", spellcheck: "false" });
  const part = h("input", { class: "input", placeholder: "partition or queue", "aria-label": "partition" });
  const acct = h("input", { class: "input", placeholder: "account", "aria-label": "account" });
  const qos = h("input", { class: "input", placeholder: "(Slurm only)", "aria-label": "QOS" });
  const mem = h("input", { class: "input", placeholder: "from the grid", "aria-label": "host memory" });
  const cpus = h("input", { class: "input num", value: "4", inputmode: "numeric", "aria-label": "CPUs" });
  const gpuType = h("input", { class: "input", placeholder: "any", "aria-label": "GPU type" });
  // What the node needs before gpuwm-cm1 runs (a module, a venv), and its Python when it is not this one.
  const setup = h("textarea", { class: "input mono", rows: "2", spellcheck: "false", "aria-label": "commands to run first",
    placeholder: "module load cuda\nsource ~/gm1/bin/activate" });
  const python = h("input", { class: "input mono", placeholder: "the Python running this page", spellcheck: "false",
    "aria-label": "Python on the node" });
  const write = h("input", { type: "checkbox", "aria-label": "write it into the run folder" });
  const out = h("div", {});
  // The file --write makes, named for the scheduler chosen.
  const fileName = h("code", {}, "job-slurm.sh");
  sched.addEventListener("change", () => { fileName.textContent = `job-${sched.value}.sh`; });

  const request = () => {
    const body = { scheduler: sched.value, time: time.value.trim() };
    if (!TIME_RE.test(body.time)) return { error: "The time limit must be HH:MM:SS, for example 04:00:00." };
    for (const [key, input] of [["partition", part], ["account", acct], ["qos", qos], ["mem", mem], ["gpu_type", gpuType]]) {
      if (input.value.trim()) body[key] = input.value.trim();
    }
    const n = Number(cpus.value);
    if (cpus.value.trim()) {
      if (!Number.isInteger(n) || n < 3) return { error: "Ask for 3 CPUs or more: one for the run and one for each helper." };
      body.cpus = n;
    }
    const lines = setup.value.split("\n").map((line) => line.trim()).filter(Boolean);
    if (lines.some((line) => line.startsWith("-"))) return { error: "Each line first on the node is a shell command, such as module load cuda." };
    if (lines.length) body.setup = lines;
    if (python.value.trim()) body.python = python.value.trim();
    const until = opts.until ? opts.until() : null;
    if (until) body.until_s = until;
    if (write.checked) body.write = true;
    return { path: `${api.runPath(runId)}/jobscript`, body };
  };

  const form = h("div", { class: "jsform" },
    field("Scheduler", sched), field("Time limit", time, "HH:MM:SS"), field("Partition or queue", part),
    field("Account", acct), field("QOS", qos), field("Host memory", mem, "e.g. 32G"), field("CPUs", cpus),
    field("GPU type", gpuType, "e.g. a100"), field("Node's Python", python, "--python"),
    h("label", { class: "jsfield jswide" }, h("span", { class: "muted" }, "Run first on the node, one command a line",
      hint("--setup")), setup));
  const make = actionButton("Write the job script", {
    action: "jobscript", kind: "primary", small: true, request, watch: [form, write],
    onclick: async () => {
      const req = request();
      if (req.error) { box.show({ ok: false, message: req.error }); return; }
      // The script, its Copy button and the submit line say it all once: the result box keeps only the command
      // (or, when it was not written, why).
      const reply = await act(make.button, req.path, req.body, null, { notice: false });
      out.replaceChildren();
      box.show(reply && reply.ok ? { command: reply.command } : reply, { diff: false });
      if (reply && reply.ok && reply.script) {
        const name = reply.file || `job-${sched.value}.sh`;
        const submit = reply.submit || `${sched.value === "slurm" ? "sbatch" : "qsub"} ${name}`;
        const text = h("pre", { class: "log", id: "jobscript-text" }, reply.script);
        out.append(text,
          h("div", { class: "btns" }, copyButton("Copy the script", reply.script, { select: text })),
          h("p", { class: "note" }, reply.file ? `Written as ${reply.file}. On the login node:` :
            `Save it as ${name} on the login node, then:`),
          copyline(submit, "submit"));
      }
    },
  });
  const panel = h("section", { class: `panel section${opts.open ? "" : " hidden"}`, id: opts.id || "jobscript" },
    h("h2", { class: "cap" }, opts.title || "Job script for a cluster"),
    h("p", { class: "note", style: "margin-top:0" }, opts.note ||
      "Submit it on the login node. The run owns the GPU node's GPU; two helpers on that node's CPUs write the CM1 " +
      "files and the pictures, so the login node only reads the run folder. Submitting the same script again goes " +
      "on from the last restart."),
    form,
    h("label", { class: "check", style: "margin:8px 0" }, write, "Write it into the run folder as ", fileName,
      hint("--write")),
    h("div", { class: "btns" }, make),
    out, box);
  panel.refreshCommand = () => make.cmd && make.cmd.refresh();
  return panel;
}
