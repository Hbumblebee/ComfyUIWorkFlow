"""Validate a ComfyUI UI workflow against a running server, without executing it.

The backend validates a *flat* node graph, so subgraph instances are inlined here
the way the frontend does it, using each subgraph's own interior link map. POSTing
to /prompt runs the real validate_prompt() -> validate_inputs() path, which checks
class_type resolution, required inputs, widget types and link type compatibility.
A 200 means every node was validated; execution is then interrupted immediately so
nothing actually renders.

usage: python validate_workflow.py <workflow.json> [first.png] [last.png]
"""

import json
import sys
import urllib.error
import urllib.request
import uuid

BASE = "http://127.0.0.1:8188"
BOUNDARY_IN = -10      # subgraph input node id
BOUNDARY_OUT = -20     # subgraph output node id


class Expander:
    def __init__(self, workflow):
        self.wf = workflow
        self.subgraphs = {s["id"]: s for s in workflow.get("definitions", {}).get("subgraphs", [])}
        self.prompt = {}
        self.top_by_id = {n["id"]: n for n in workflow["nodes"]}
        self.top_links = {l[0]: l for l in workflow["links"]}   # [id, src, sslot, dst, dslot, type]

    # ---------------------------------------------------------------- helpers
    def _widgets(self, node):
        """widget name -> value, as the API expects them."""
        named = node.get("widgets_values_named") or {}
        return dict(named)

    def _api_id(self, scope, node_id):
        return f"{scope}_{node_id}" if scope else f"n{node_id}"

    # ---------------------------------------------------------------- expand
    def expand(self):
        for n in self.wf["nodes"]:
            if n["type"] in self.subgraphs:
                self._expand_instance(n)
            elif n["type"] != "MarkdownNote":
                self._add_plain(n, scope="")
        self._wire_top_level()
        return self.prompt

    def _add_plain(self, node, scope):
        self.prompt[self._api_id(scope, node["id"])] = {
            "class_type": node["type"],
            "inputs": self._widgets(node),
            "_meta": {"title": node.get("title") or node["type"]},
        }

    def _expand_instance(self, inst):
        """Inline one subgraph instance; returns nothing, fills self.prompt."""
        sg = self.subgraphs[inst["type"]]
        scope = f"sg{inst['id']}"

        # The instance's `inputs` array is ordered promoted-widgets-first, while the
        # definition orders image/string inputs before widget inputs. The two orders
        # therefore do NOT line up slot by slot -- bind them BY NAME.
        inst_by_name = {i.get("name"): i for i in inst.get("inputs", [])}
        boundary = {}          # definition input slot -> [api_id, out_slot]
        for slot, dinp in enumerate(sg["inputs"]):
            iinp = inst_by_name.get(dinp["name"])
            if iinp is None:
                continue
            link = iinp.get("link")
            if link is None:
                continue
            L = self.top_links.get(link)
            if L is None:
                continue
            boundary[slot] = [f"n{L[1]}", L[2]]

        interior = {l["id"]: l for l in sg["links"]}
        for n in sg["nodes"]:
            self._add_plain(n, scope)
            for inp in (n.get("inputs") or []):
                link = inp.get("link")
                if link is None:
                    continue
                L = interior.get(link)
                if L is None:
                    raise SystemExit(f"interior node {n['id']} input '{inp.get('name')}' "
                                     f"-> unknown interior link {link}")
                if L["origin_id"] == BOUNDARY_IN:
                    src = boundary.get(L["origin_slot"])
                    if src is None:
                        # widget-promoted input with nothing wired outside: the UI
                        # leaves the interior node's own value in place, so the API
                        # prompt simply omits the input and the node default applies
                        continue
                else:
                    src = [self._api_id(scope, L["origin_id"]), L["origin_slot"]]
                self.prompt[self._api_id(scope, n["id"])]["inputs"][inp["name"]] = src

        self._instances = getattr(self, "_instances", {})
        self._instances[inst["id"]] = (sg, scope)

        # the subgraph output boundary (-20) is not a real node in the backend
        # prompt: the frontend splices whatever feeds it straight into the consumer.
        # Record that source so top-level consumers can be wired to it.
        for lid in (sg.get("outputs") or [{}])[0].get("linkIds", []) or []:
            L = interior.get(lid)
            if L is None:
                continue
            # link runs interior_node -> -20, so the real source is the origin
            self._out_src = {inst["id"]: [self._api_id(scope, L["origin_id"]), L["origin_slot"]]}

    def _wire_top_level(self):
        """Wire links whose endpoints are ordinary top-level nodes."""
        for lid, src, sslot, dst, dslot, _typ in self.wf["links"]:
            src_node = self.top_by_id.get(src)
            dst_node = self.top_by_id.get(dst)
            if src_node is None or dst_node is None:
                continue
            inputs = dst_node.get("inputs") or []
            if dslot >= len(inputs):
                continue
            name = inputs[dslot]["name"]
            if src_node["type"] in self.subgraphs:
                # an instance output feeding a top-level node: splice the real source
                spliced = getattr(self, "_out_src", {}).get(src)
                if spliced is None:
                    continue
                self.prompt[f"n{dst}"]["inputs"][name] = spliced
                continue
            if dst_node["type"] in self.subgraphs:
                continue                      # handled by _expand_instance
            self.prompt[f"n{dst}"]["inputs"][name] = [f"n{src}", sslot]

    def subgraph_output(self, inst_id, out_slot=0):
        """API source for a subgraph instance's output slot."""
        return [f"sg{inst_id}_{BOUNDARY_OUT}", out_slot]


def localise_loadimages(prompt, first, last):
    """Swap the placeholder filenames for images that exist on this machine."""
    names = [v for k, v in sorted(prompt.items()) if v["class_type"] == "LoadImage"]
    for entry, fname in zip(names, [first, last]):
        entry["inputs"]["image"] = fname
    return names


def main():
    path = sys.argv[1]
    first = sys.argv[2] if len(sys.argv) > 2 else "Anime_T2I_4K_20261006_143812.png"
    last = sys.argv[3] if len(sys.argv) > 3 else "QwenEdit2511_20261006_150458.png"

    wf = json.load(open(path, encoding="utf-8"))
    ex = Expander(wf)
    prompt = ex.expand()
    localise_loadimages(prompt, first, last)

    print("expanded to %d API nodes" % len(prompt))
    for k in sorted(prompt):
        v = prompt[k]
        print("  %-10s %-30s" % (k, v["class_type"]))

    body = json.dumps({"prompt": prompt, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(BASE + "/prompt", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=60)
        payload = json.loads(r.read().decode())
        print("\nHTTP", r.status, payload)
        print("=> VALIDATION PASSED (all %d nodes accepted)" % len(prompt))
        return 0
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        print("\nHTTP", e.code, "=> VALIDATION FAILED")
        try:
            j = json.loads(raw)
            err = j.get("error", {})
            print("  type   :", err.get("type"))
            print("  message:", err.get("message"))
            print("  details:", str(err.get("details"))[:300])
            for nid, ne in (j.get("node_errors") or {}).items():
                print("  node %s (%s)" % (nid, ne.get("class_type")))
                for r2 in ne.get("errors", []):
                    print("     -", r2.get("message"), "|", str(r2.get("details"))[:300])
        except Exception:
            print(raw[:1500])
        return 1


if __name__ == "__main__":
    sys.exit(main())
