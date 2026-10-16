import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// MiniMax H3 Project: shows this project's cache and manages every cache on disk.

const TARGET = "MiniMaxH3Project";

function sizeText(bytes) {
    return bytes >= 1024 ** 3 ? `${(bytes / 1024 ** 3).toFixed(2)} GB` : `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

function timeText(seconds) {
    const date = new Date(seconds * 1000);
    return date.toDateString() === new Date().toDateString()
        ? date.toLocaleTimeString([], { timeStyle: "short" })
        : date.toLocaleDateString([], { dateStyle: "short" });
}

// Grow the node so its buttons fit and the status line is not cut off.
function fitNode(node) {
    const ctx = app.canvas?.ctx;
    let width = node.computeSize()[0];
    if (ctx && node.__h3ProjectStatus) {
        ctx.save();
        ctx.font = "12px Arial";
        width = Math.max(width, ctx.measureText(node.__h3ProjectStatus.label).width + 60);
        ctx.restore();
    }
    const height = node.computeSize()[1];
    if (node.size[0] < width || node.size[1] < height) {
        node.setSize([Math.max(node.size[0], width), Math.max(node.size[1], height)]);
    }
}

async function listCaches() {
    const response = await api.fetchApi("/h3_extender/caches");
    return (await response.json()).caches || [];
}

async function post(path, body) {
    const response = await api.fetchApi(path, { method: "POST", body: JSON.stringify(body) });
    const result = await response.json();
    if (!result.ok) throw new Error(result.error || `${path} failed`);
    return result;
}

function nameWidget(node) {
    return node.widgets?.find((w) => w.name === "name");
}

async function refreshStatus(node) {
    const status = node.__h3ProjectStatus;
    if (!status) return;
    const name = String(nameWidget(node)?.value || "").trim();
    try {
        const cache = (await listCaches()).find((c) => c.name === name);
        status.label = !name
            ? "Give the project a name"
            : cache
                ? `${cache.validated}/${cache.clips} validated · ${sizeText(cache.size_bytes)} · ${timeText(cache.updated_at)}`
                : "No cache yet";
    } catch (error) {
        status.label = "Cache status unavailable";
    }
    fitNode(node);
    node.setDirtyCanvas(true, true);
}

function refreshAll() {
    for (const node of app.graph?.nodes || []) {
        if (node.type === TARGET) refreshStatus(node);
    }
}

async function wipe(name) {
    if (!confirm(
        `Delete the cache '${name}'?\n\n` +
        "Its clips, audio and previews are deleted for good; the next run samples every clip again."
    )) return false;
    try {
        const result = await post("/h3_extender/caches/wipe", { name });
        alert(`Deleted '${name}', ${sizeText(result.freed_bytes)} freed.`);
        return true;
    } catch (error) {
        alert(error.message);
        return false;
    }
}

async function renameProject(node) {
    const widget = nameWidget(node);
    const old = String(widget?.value || "").trim();
    const name = prompt(
        `Rename the project '${old}' to:

` +
        "Its clip chain, its refined chain and its Latent Caches are renamed too, so nothing runs again.",
        old,
    );
    if (name === null || !name.trim() || name.trim() === old) return;
    try {
        const result = old ? await post("/h3_extender/caches/rename_project", { old, new: name.trim() }) : { name: name.trim() };
        widget.value = result.name;
        widget.callback?.(result.name);
        app.graph.setDirtyCanvas(true, true);
    } catch (error) {
        alert(error.message);
    }
}

async function archiveProject(node) {
    const name = String(nameWidget(node)?.value || "").trim();
    if (!name) return;
    if (!confirm(
        `Archive the project '${name}'?\n\n` +
        "Its cache, this workflow as it is now, and the input files it loads go into one .h3project file " +
        "in output/h3_projects. Previews and decoded videos are left out; the next run makes them again."
    )) return;
    try {
        const result = await post("/h3_extender/caches/archive", { name, workflow: (app.rootGraph ?? app.graph).serialize() });
        alert(`Archived '${name}' with ${result.inputs.length} input files (${sizeText(result.size_bytes)}):\n${result.path}`);
    } catch (error) {
        alert(error.message);
    }
}

function button(text, onClick) {
    const el = document.createElement("button");
    el.textContent = text;
    el.style.cssText = "margin-left:6px;padding:2px 8px;cursor:pointer;";
    el.addEventListener("click", onClick);
    return el;
}

function dialog(onClose) {
    const overlay = document.createElement("div");
    overlay.style.cssText = "position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:10000;display:flex;align-items:center;justify-content:center;";
    const panel = document.createElement("div");
    panel.style.cssText = "background:var(--comfy-menu-bg,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#555);" +
        "border-radius:8px;padding:16px;min-width:560px;max-width:90vw;max-height:80vh;overflow:auto;font:13px sans-serif;";
    overlay.appendChild(panel);
    const close = () => {
        overlay.remove();
        document.removeEventListener("keydown", onKey);
        onClose?.();
    };
    const onKey = (event) => { if (event.key === "Escape") close(); };
    overlay.addEventListener("click", (event) => { if (event.target === overlay) close(); });
    document.addEventListener("keydown", onKey);
    document.body.appendChild(overlay);
    return { panel, close };
}

function dialogHeader(panel, text, close) {
    const header = document.createElement("div");
    header.style.cssText = "display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;";
    const title = document.createElement("b");
    title.textContent = text;
    header.append(title, button("Close", close));
    panel.appendChild(header);
}

async function showArchives(node) {
    const { panel, close } = dialog(refreshAll);
    dialogHeader(panel, "Restore a MiniMax H3 project", close);
    let listing;
    try {
        listing = await (await api.fetchApi("/h3_extender/caches/archives")).json();
    } catch (error) {
        panel.append("Could not list the archives.");
        return;
    }
    const where = document.createElement("div");
    where.style.cssText = "opacity:.75;margin-bottom:8px;";
    where.textContent = `Archives in ${listing.folder}`;
    panel.appendChild(where);
    if (!listing.archives.length) panel.append("No archives yet.");
    for (const archive of listing.archives) {
        const row = document.createElement("div");
        row.style.cssText = "display:flex;align-items:center;gap:12px;padding:6px 0;border-top:1px solid var(--border-color,#444);";
        const label = document.createElement("span");
        label.style.cssText = "flex:1;";
        label.textContent = `${archive.name} · ${sizeText(archive.size_bytes)} · ${timeText(archive.created_at)}`;
        const input = document.createElement("input");
        input.placeholder = "restore as (optional)";
        input.style.cssText = "width:160px;";
        row.append(label, input, button("Restore", async () => {
            try {
                const result = await post("/h3_extender/caches/restore", { file: archive.file, name: input.value.trim() });
                const renamed = Object.entries(result.renamed_inputs)
                    .map(([from, to]) => `\n  ${from} → ${to} (a different file had its name)`).join("");
                alert(`Restored '${result.name}'. Open '${result.workflow}' from the Workflows panel.${renamed}`);
                close();
            } catch (error) {
                alert(error.message);
            }
        }));
        panel.appendChild(row);
    }
}

async function showCaches(node) {
    const { panel, close } = dialog(refreshAll);

    const render = async () => {
        panel.replaceChildren();
        dialogHeader(panel, "MiniMax H3 clip caches", close);

        let caches;
        try {
            caches = await listCaches();
        } catch (error) {
            panel.append("Could not list the caches.");
            return;
        }
        if (!caches.length) panel.append("No caches on disk.");
        const current = String(nameWidget(node)?.value || "").trim();
        for (const cache of caches) {
            const row = document.createElement("div");
            row.style.cssText = "display:flex;align-items:center;gap:12px;padding:6px 0;border-top:1px solid var(--border-color,#444);";
            const label = document.createElement("span");
            label.style.cssText = "flex:1;";
            label.textContent = `${cache.name}${cache.name === current ? " (this project)" : ""}`;
            const info = document.createElement("span");
            info.style.cssText = "opacity:.75;white-space:nowrap;";
            info.textContent = `${cache.validated}/${cache.clips} validated · ${sizeText(cache.size_bytes)} · ${timeText(cache.updated_at)}`;
            row.append(label, info);
            row.append(button("Use", () => {
                const widget = nameWidget(node);
                widget.value = cache.name;
                widget.callback?.(cache.name);
                close();
            }));
            row.append(button("Rename", () => {
                const input = document.createElement("input");
                input.value = cache.name;
                input.style.cssText = "flex:1;";
                const confirmRename = async () => {
                    try {
                        const result = await post("/h3_extender/caches/rename", { old: cache.name, new: input.value });
                        if (cache.name === current) {
                            const widget = nameWidget(node);
                            widget.value = result.name;
                            widget.callback?.(result.name);
                        }
                        render();
                    } catch (error) {
                        alert(error.message);
                    }
                };
                input.addEventListener("keydown", (event) => { if (event.key === "Enter") confirmRename(); });
                row.replaceChildren(input, button("Rename", confirmRename), button("Cancel", render));
                input.focus();
            }));
            row.append(button("Wipe", async () => { if (await wipe(cache.name)) render(); }));
            panel.appendChild(row);
        }
    };
    render();
}

app.registerExtension({
    name: "MiniMaxH3.Project.CacheManager",

    setup() {
        api.addEventListener("execution_success", refreshAll);
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== TARGET) return;

        const oldCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = oldCreated ? oldCreated.apply(this, arguments) : undefined;
            const node = this;
            node.__h3ProjectStatus = node.addWidget("button", "cache_status", null, () => refreshStatus(node), { serialize: false });
            node.__h3ProjectStatus.label = "…";
            node.addWidget("button", "Wipe this cache…", null, async () => {
                const name = String(nameWidget(node)?.value || "").trim();
                if (name && await wipe(name)) refreshStatus(node);
            }, { serialize: false });
            node.addWidget("button", "Rename project…", null, () => renameProject(node), { serialize: false });
            node.addWidget("button", "All caches…", null, () => showCaches(node), { serialize: false });
            node.addWidget("button", "Archive…", null, () => archiveProject(node), { serialize: false });
            node.addWidget("button", "Restore…", null, () => showArchives(node), { serialize: false });
            const widget = nameWidget(node);
            if (widget) {
                const oldCallback = widget.callback;
                widget.callback = function () {
                    const value = oldCallback ? oldCallback.apply(this, arguments) : undefined;
                    refreshStatus(node);
                    return value;
                };
            }
            setTimeout(() => refreshStatus(node), 0);
            return result;
        };

        const oldConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = oldConfigure ? oldConfigure.apply(this, arguments) : undefined;
            setTimeout(() => refreshStatus(this), 0);
            return result;
        };
    },
});
