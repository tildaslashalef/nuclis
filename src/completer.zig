//! The completion side of a conversation: renders messages through the
//! model's profile, continues the live session when the render extends what
//! it consumed, else restores the longest cached state (`cache.zig`), else
//! prefills, then runs `inference.engine.complete`. A template that rewrites
//! the model's past turns never extends what the session consumed: there
//! the session moves back to the prompt's end (`Mark`). The agent loop and the
//! API's language-model service drive it through the same `Model` seam; it
//! owns no conversation, only the session's consumed-text bookkeeping
//! (docs/engine/session.md § The agent's token cache).
const std = @import("std");
const inference = @import("inference");
const engine = @import("engine.zig");
const config = @import("config.zig");
const model_mod = @import("model.zig");
pub const cache = @import("completer/cache.zig");

const Allocator = std.mem.Allocator;
const Profile = inference.profiles;

/// Adapts state that has a `send`-like method for inference events. The
/// agent loop installs one whose handler parses the answer channel; tests
/// install a recorder.
pub const Sink = struct {
    context: *anyopaque,
    call: *const fn (*anyopaque, inference.events.Event) anyerror!void,

    pub fn send(self: *Sink, event: inference.events.Event) !void {
        try self.call(self.context, event);
    }
};

/// One image attached to a user message: where it came from, its decoded
/// size, and the projector's rows the completer substitutes at prefill.
/// Owned by the history item that carries it.
pub const Image = struct {
    path: []u8,
    /// The file's size in bytes, for the transcript's detail row.
    bytes: u64,
    prepared: inference.engine.PreparedImage,

    pub fn ref(self: Image) Profile.ImageRef {
        return .{ .width_tokens = self.prepared.width_tokens, .height_tokens = self.prepared.height_tokens };
    }
    pub fn deinit(self: *Image, alloc: Allocator) void {
        alloc.free(self.path);
        alloc.free(self.prepared.features);
        self.* = undefined;
    }
};

/// Frees a list of images and the list.
pub fn freeImages(alloc: Allocator, images: []Image) void {
    for (images) |*image| image.deinit(alloc);
    if (images.len != 0) alloc.free(images);
}

/// One open model: the engine and everything sized by its vocabulary or
/// keyed by its files. A switch or a context change replaces it in place, so
/// the pointers a completer holds into it stay valid.
pub const Open = struct {
    eng: engine.Engine,
    settings: config.Resolved,
    /// Owned: the file, its draft source, and the digest its sidecar
    /// verified (null without one).
    model_path: []u8,
    draft_path: ?[]u8,
    digest: ?[]u8,
    logits: []f32,
    /// Sized for sampling whatever the current temperature: the effort can
    /// switch profiles mid-session.
    candidates: []inference.sampling.Candidate,
    generated: []u32,
    history: inference.sampling.History,
    /// False once unloaded: a switch whose fallback also failed leaves
    /// nothing to free.
    alive: bool = true,

    pub fn init(alloc: std.mem.Allocator, io: std.Io, model_path: []const u8, settings: config.Resolved) !Open {
        if (settings.ctx_size == 0 or settings.ctx_size > config.max_context or settings.max_tokens == 0 or settings.max_tokens > config.max_output_tokens) return error.InvalidGenerationBudget;
        const path = try alloc.dupe(u8, model_path);
        errdefer alloc.free(path);
        const draft_path = try engine.draftPath(alloc, path, if (settings.entry) |entry| entry.mtp else null);
        errdefer if (draft_path) |d| alloc.free(d);
        const draft: inference.engine.DraftRequest = if (settings.speculative) .{ .preferred = draft_path } else .none;
        var eng = try engine.Engine.open(alloc, io, path, settings.backend, settings.ctx_size, settings.kv_precision, settings.forced_profile, draft);
        errdefer eng.deinit();
        // The file's own profile from here on: the configuration guessed one
        // from the catalogue name without opening the file (`config show`).
        if (eng.profile == null) return error.UnsupportedPromptTemplate;
        const vocab = eng.vocab.tokens.len;
        const logits = try alloc.alloc(f32, vocab);
        errdefer alloc.free(logits);
        const candidates = try alloc.alloc(inference.sampling.Candidate, vocab);
        errdefer alloc.free(candidates);
        const generated = try alloc.alloc(u32, settings.max_tokens);
        errdefer alloc.free(generated);
        var history = try inference.sampling.History.init(alloc, vocab);
        errdefer history.deinit();
        return .{ .eng = eng, .settings = settings, .model_path = path, .draft_path = draft_path, .digest = try readDigest(alloc, io, path), .logits = logits, .candidates = candidates, .generated = generated, .history = history };
    }

    pub fn deinit(self: *Open, alloc: std.mem.Allocator) void {
        if (!self.alive) return;
        self.alive = false;
        self.history.deinit();
        alloc.free(self.generated);
        alloc.free(self.candidates);
        alloc.free(self.logits);
        self.eng.deinit();
        if (self.digest) |d| alloc.free(d);
        if (self.draft_path) |d| alloc.free(d);
        alloc.free(self.model_path);
    }

    pub fn profile(self: *const Open) Profile.Profile {
        return self.eng.profile.?;
    }
};

/// The digest the model's sidecar verified, or null without a readable one.
fn readDigest(alloc: std.mem.Allocator, io: std.Io, model_path: []const u8) !?[]u8 {
    // The sidecar's strings live in the arena; only the digest outlives it.
    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const a = arena.allocator();
    const sidecar_path = model_mod.sidecarPath(a, model_path) catch return null;
    const sidecar = model_mod.readSidecar(a, io, .cwd(), sidecar_path) catch return null;
    return if (sidecar) |record| try alloc.dupe(u8, record.sha256) else null;
}

/// What one completion step reported. `outcome` is the engine's; `replay`
/// is set when the conversation could not continue the session where it
/// stood and was prefilled again from a cached state or from empty.
pub const Reply = struct {
    outcome: inference.engine.Outcome,
    replay: ?Replay = null,
    /// The rendered conversation's tokens, and how many of them the session
    /// already held (continued or restored) rather than prefilled.
    prompt_tokens: usize = 0,
    reused_tokens: usize = 0,
};

/// Why a step re-prefilled the conversation, and how much of it a cached
/// state restored instead.
pub const Replay = struct {
    cause: Cause,
    /// Tokens restored from the cache rather than prefilled; 0 on a miss.
    restored: usize = 0,

    pub const Cause = enum {
        /// A stored conversation is rendered into a fresh session.
        resumed,
        /// The previous step was cancelled and its state discarded.
        cancel,
        /// An earlier message renders differently (compaction, elision).
        rewrite,
        /// The reasoning effort changed the system block.
        effort,
        /// The engine was re-opened at another context size.
        context,
        /// Another model was loaded; its profile renders the conversation.
        model,
        /// A failed turn reset the session.
        failure,
    };
};

/// The completion side of a step. `messages` is the conversation to render
/// (system first) and `tools` are the definitions the profile renders into
/// that system message; the implementation forwards every engine event to
/// `sink`.
pub const Model = struct {
    context: *anyopaque,
    /// `images` are the attachments of every rendered message, in order;
    /// their `ImageRef`s are already on the messages.
    run: *const fn (*anyopaque, messages: []const Profile.Message, definitions: []const Profile.ToolDefinition, images: []const Image, sink: *Sink) anyerror!Reply,
    /// Tokens `text` costs in the model's vocabulary; what the result budget
    /// is measured in. Empty text costs zero, never an error: tool results
    /// may be empty.
    count: *const fn (*anyopaque, text: []const u8) anyerror!usize,
    /// Runs the projector over an image's encoded bytes; `NoVision` without
    /// one. The features are allocated with the given allocator.
    encode_image: *const fn (*anyopaque, Allocator, bytes: []const u8) anyerror!inference.engine.PreparedImage = noVision,
    /// Called when a turn ends in an answer: a boundary the next turn's
    /// render starts with, worth keeping. Returns where the state was saved
    /// to disk, which the session records for a resume; null when it was
    /// not. Best effort, never an error.
    checkpoint: *const fn (*anyopaque) ?cache.Boundary = noCheckpoint,
};

fn noCheckpoint(_: *anyopaque) ?cache.Boundary {
    return null;
}

fn noVision(_: *anyopaque, _: Allocator, _: []const u8) anyerror!inference.engine.PreparedImage {
    return error.NoVision;
}

/// What did not fit when a completion reported `ContextFull`: the tokens the
/// step needed against the window. For the message the user reads.
pub const Overflow = struct { needed: usize, capacity: usize };

/// The text still to prefill for `full` given what the model has consumed, or
/// null when the conversation must be replayed from an empty session.
pub fn increment(seen: []const u8, full: []const u8) ?[]const u8 {
    if (!std.mem.startsWith(u8, full, seen)) return null;
    return full[seen.len..];
}

/// The fewest tokens a state saved inside a step's prompt holds: below this a
/// Qwen snapshot (150 MB before its first row) is not worth keeping.
pub const min_save_tokens = 64;

/// Where a step's prompt ended on an attention-only model, which can move
/// back there by position (`inference.engine.rewindSession`) when the next
/// render shares only that much: always, where the profile rewrites its
/// turns (`Profile.rewritesTurn`); else on a request sent again. Owns
/// `history`, the penalty set at that point.
pub const Mark = struct {
    bytes: usize,
    position: usize,
    /// The token fed at `position - 1`, which a drafter is re-fed.
    last: u32,
    history: ?std.bit_set.Dynamic,
};

/// The real `Model`: renders through the artifact's profile and completes one
/// step, keeping the session's consumed-text bookkeeping so a growing
/// conversation prefills only its remainder when it can.
pub const Completer = struct {
    alloc: Allocator,
    eng: *engine.Engine,
    effort: Profile.Effort,
    sampler: *inference.sampling.Sampler,
    /// Mirrors the session for the sampler's penalties; reset wherever the
    /// session is.
    history: ?*inference.sampling.History,
    buffers: inference.engine.CompletionBuffers,
    /// `agent.thinking_budget`: applied per step at the current effort.
    thinking_budget: usize = 0,
    /// Speculative decoding for this turn, resolved from the configuration
    /// and flags; the engine must have been opened with a matching drafter.
    speculative: inference.engine.Speculative = .{},
    observer: ?inference.observer.Observer = null,
    /// Rendered prompt plus generated text the session has consumed. Empty
    /// after a reset, which is what makes the next render replay.
    seen: std.ArrayList(u8) = .empty,
    /// The effort `seen` was rendered at: a render at another one differs
    /// in the system block, which names the replay's cause.
    seen_effort: Profile.Effort = .off,
    /// Bytes of `seen` that are the primed prefix; more means a conversation.
    primed_len: usize = 0,
    /// Set when `run` last reported `ContextFull`, for the caller's message.
    overflow: ?Overflow = null,
    /// States at turn boundaries and primed prefixes, restored instead of
    /// re-prefilled when a render starts with one (`cache.zig`).
    cache: cache.Memory,
    /// Primed prefixes and turn ends across processes; null when disabled
    /// or unavailable.
    disk: ?cache.Disk = null,
    /// Save each turn's end to disk: only where a session file is written,
    /// since nothing else can resume it.
    save_turns: bool = false,
    /// The disk key of this conversation's last saved turn, replaced by the
    /// next one.
    saved_turn: ?inference.prefix_cache.Key = null,
    /// The resumed conversation's last saved turn, tried once by the next
    /// step's fallback.
    hint: ?cache.Boundary = null,
    /// Why the next step will not continue where the session stands, when
    /// the reason is known before the render (a reset, a cancel, a re-prime).
    cause: ?Replay.Cause = null,
    /// An image prefill ran in this session: the drafter's cache is stale
    /// from then on, so speculation stays off until `reset`.
    images_fed: bool = false,
    /// Where the last step's prompt ended, when the next render cannot
    /// continue past it (`Mark`).
    mark: ?Mark = null,
    /// Cap the output budget at the space the prompt leaves instead of
    /// refusing it: a client sizes `max_tokens` from its own idea of the
    /// window. The agent keeps the whole budget or reports `ContextFull`.
    clamp_budget: bool = false,

    pub fn model(self: *Completer) Model {
        return .{ .context = self, .run = run, .count = count, .encode_image = encodeImage, .checkpoint = checkpoint };
    }

    /// The projector over one image's bytes, with the features moved to the
    /// caller's allocator. `NoVision` when no projector is loaded.
    fn encodeImage(context: *anyopaque, alloc: Allocator, bytes: []const u8) anyerror!inference.engine.PreparedImage {
        const self: *Completer = @ptrCast(@alignCast(context));
        var prepared = try self.eng.encodeImage(bytes);
        const engine_owned = prepared.features;
        defer self.eng.alloc.free(engine_owned);
        prepared.features = try alloc.dupe(f32, engine_owned);
        return prepared;
    }

    /// Where a primed prefix came from.
    pub const Primed = struct {
        tokens: usize = 0,
        from: enum { prefill, memory, disk } = .prefill,
    };

    /// Puts the system block and tools in the session now, so the first turn
    /// — and every session that starts from the same prefix — pays only its
    /// own message: restored from the memory tier, else the disk tier, else
    /// prefilled and saved to both. On failure the session is reset;
    /// `ContextFull` (with `overflow` set) means the window cannot hold the
    /// prefix plus the output budget. `tokens` is 0 when the profile has no
    /// prefix to prime.
    pub fn prime(self: *Completer, system: []const u8, definitions: []const Profile.ToolDefinition) !Primed {
        const text = try self.eng.prefix(&.{.{ .role = .system, .content = system }}, definitions, self.effort);
        defer self.alloc.free(text);
        if (text.len == 0) return .{};
        const tokens = try self.eng.encode(text);
        defer self.alloc.free(tokens);
        const session = self.eng.model.session();
        if (tokens.len + self.buffers.generated.len > session.capacity) {
            self.overflow = .{ .needed = tokens.len + self.buffers.generated.len, .capacity = session.capacity };
            return error.ContextFull;
        }
        // Re-priming under a conversation replays it on the next step.
        if (self.seen.items.len > self.primed_len) self.cause = self.cause orelse .effort;
        if (self.cache.exact(text)) |entry| {
            if (self.restoreEntry(entry)) {
                self.primed_len = text.len;
                return .{ .tokens = tokens.len, .from = .memory };
            } else |_| self.cache.remove(entry);
        }
        // A per-layer observer is a per-token contract: those runs prefill.
        const layer_observer = self.observer != null and self.observer.?.layer != null;
        const disk_key = if (self.disk) |*d| d.key(cache.textDigest(text), session.layout_digest) else undefined;
        if (!layer_observer) if (self.disk) |*d| {
            if (d.load(self.alloc, disk_key) catch null) |loaded| {
                var snap = loaded;
                if (self.restoreSnapshot(&snap, text, tokens)) {
                    self.primed_len = text.len;
                    return .{ .tokens = tokens.len, .from = .disk };
                } else |_| snap.deinit();
            }
        };
        self.dropMark();
        self.images_fed = false;
        self.eng.model.reset();
        if (self.history) |h| h.reset();
        self.seen.clearRetainingCapacity();
        self.primed_len = 0;
        // With a drafter, prime through `commitPrompt` so the primed snapshot
        // carries the block's cache rows; a per-layer observer keeps the
        // ordinary prefill (speculation is off then).
        if (self.eng.model.drafter() != null and !layer_observer) {
            inference.engine.commitPrompt(self.eng, tokens, self.buffers.logits, self.observer) catch |err| {
                self.eng.model.reset();
                return err;
            };
        } else {
            self.eng.model.prefill(tokens, self.buffers.logits, null, null, null, null, self.observer) catch |err| {
                self.eng.model.reset();
                return err;
            };
        }
        if (self.history) |h| for (tokens) |token| try h.observe(token);
        try self.seen.appendSlice(self.alloc, text);
        self.seen_effort = self.effort;
        self.primed_len = text.len;
        var snap = try self.eng.model.snapshot(self.alloc);
        // A failed save costs the next process a prefill, nothing more.
        if (!layer_observer) if (self.disk) |*d| d.save(self.alloc, disk_key, &snap) catch {};
        self.keep(snap);
        return .{ .tokens = tokens.len };
    }

    /// Restores a disk snapshot of `text` (`tokens`) and keeps it in memory.
    /// The snapshot is consumed on success, the caller's on failure.
    fn restoreSnapshot(self: *Completer, snap: *inference.session.Snapshot, text: []const u8, tokens: []const u32) !void {
        self.dropMark();
        self.eng.model.reset();
        errdefer self.eng.model.reset();
        try self.eng.model.restore(snap);
        if (self.history) |h| {
            h.reset();
            for (tokens) |token| try h.observe(token);
        }
        self.seen.clearRetainingCapacity();
        try self.seen.appendSlice(self.alloc, text);
        self.seen_effort = self.effort;
        self.images_fed = false;
        self.keep(snap.*);
    }

    /// Hands `snap`, taken where the session stands, to the memory tier
    /// under `seen` and the current history. Consumes `snap` either way.
    fn keep(self: *Completer, snap: inference.session.Snapshot) void {
        self.keepAs(snap, self.seen.items);
    }

    /// `keep` under `consumed`, the text the session has consumed when that
    /// is not `seen` (a state saved inside a step's prompt).
    fn keepAs(self: *Completer, snap: inference.session.Snapshot, consumed: []const u8) void {
        var owned = snap;
        const text = self.alloc.dupe(u8, consumed) catch {
            owned.deinit();
            return;
        };
        const bits: ?std.bit_set.Dynamic = if (self.history) |h| h.seen.clone(self.alloc) catch {
            self.alloc.free(text);
            owned.deinit();
            return;
        } else null;
        _ = self.cache.insert(.{ .text = text, .tokens = owned.position, .history = bits, .snapshot = owned });
    }

    /// Puts the session at `entry`'s state. On failure the session is reset.
    fn restoreEntry(self: *Completer, entry: *const cache.Entry) !void {
        self.dropMark();
        // A reset first: `restore` needs a ready session, and a cancelled
        // step leaves a failed one.
        self.eng.model.reset();
        errdefer self.eng.model.reset();
        try self.eng.model.restore(&entry.snapshot);
        if (self.history) |h| {
            h.reset();
            if (entry.history) |bits| {
                h.seen.setUnion(bits);
                h.revision += 1;
            }
        }
        self.seen.clearRetainingCapacity();
        try self.seen.appendSlice(self.alloc, entry.text);
        self.seen_effort = self.effort;
        self.images_fed = false;
    }

    /// The turn ended in an answer: keep the state under what it consumed,
    /// in memory and, when turns are saved, on disk in place of this
    /// conversation's previous turn. With a `Mark`, the session first moves
    /// back to the prompt's end, the last point the next render shares. Skipped when images are in it, since
    /// their placeholder text does not tell one image from another.
    fn checkpoint(context: *anyopaque) ?cache.Boundary {
        const self: *Completer = @ptrCast(@alignCast(context));
        if (self.images_fed or self.seen.items.len == 0) return null;
        // The next render cannot continue past the prompt's end: go back
        // there first, so the state kept is one it can use.
        if (self.mark) |m| if (self.eng.profile.?.rewritesTurn(self.effort)) self.rewindToMark(m) catch return null;
        const to_disk = self.save_turns and self.disk != null;
        const in_memory = self.cache.budget > 0 and self.cache.exact(self.seen.items) == null;
        if (!to_disk and !in_memory) return null;
        var snap = self.eng.model.snapshot(self.alloc) catch return null;
        if (snap.span_count != 0) {
            snap.deinit();
            return null;
        }
        var saved: ?cache.Boundary = null;
        if (to_disk) {
            const boundary: cache.Boundary = .of(self.seen.items);
            const d = &self.disk.?;
            const k = d.key(boundary.digest, snap.layout_digest);
            if (d.save(self.alloc, k, &snap)) {
                if (self.saved_turn) |old| if (!std.meta.eql(old, k)) d.remove(old);
                self.saved_turn = k;
                saved = boundary;
            } else |_| {}
        }
        if (in_memory) self.keep(snap) else snap.deinit();
        return saved;
    }

    /// Continues a stored conversation: `reset(.resumed)`, and the next
    /// step tries `boundary` (its last saved turn) on disk first.
    pub fn resumeFrom(self: *Completer, boundary: ?cache.Boundary) void {
        self.reset(.resumed);
        self.hint = boundary;
        self.saved_turn = null;
    }

    /// Restores the disk state at `b` when `full` starts with it; the tokens
    /// restored, or null on a miss. The penalty history is rebuilt from the
    /// text's encoding.
    fn restoreBoundary(self: *Completer, b: cache.Boundary, full: []const u8) ?usize {
        if (!b.prefixes(full)) return null;
        const d = &(self.disk orelse return null);
        const text = full[0..b.bytes];
        var snap = (d.load(self.alloc, d.key(b.digest, self.eng.model.session().layout_digest)) catch return null) orelse return null;
        const tokens = self.eng.encode(text) catch {
            snap.deinit();
            return null;
        };
        defer self.alloc.free(tokens);
        const position = snap.position;
        self.restoreSnapshot(&snap, text, tokens) catch {
            snap.deinit();
            return null;
        };
        return position;
    }

    /// Forgets every cached state (a re-opened engine cannot restore them).
    pub fn dropCache(self: *Completer) void {
        self.cache.clear();
    }

    fn count(context: *anyopaque, text: []const u8) anyerror!usize {
        const self: *Completer = @ptrCast(@alignCast(context));
        // The encoder refuses an empty prompt; an empty result (a glob with
        // no match, a command with no output) simply costs nothing.
        if (text.len == 0) return 0;
        const tokens = try self.eng.encode(text);
        defer self.alloc.free(tokens);
        return tokens.len;
    }

    /// Forgets what the session consumed, so the next step renders from an
    /// empty session (or a cached prefix). `cause` names the replay when a
    /// conversation continues (`/resume`, a re-opened engine, a failed
    /// turn); null for a new one. The session itself is reset on the next
    /// render.
    pub fn reset(self: *Completer, cause: ?Replay.Cause) void {
        self.dropMark();
        self.images_fed = false;
        self.seen.clearRetainingCapacity();
        self.primed_len = 0;
        self.cause = cause;
        // A new conversation keeps the last one's saved turn: it can still
        // be resumed.
        if (cause == null) self.saved_turn = null;
    }

    pub fn deinit(self: *Completer) void {
        self.dropMark();
        self.cache.deinit();
        if (self.disk) |*d| d.close();
        self.seen.deinit(self.alloc);
    }

    /// Moves the session back to `m` (`Mark`) and forgets it. On failure the
    /// session is reset, as after a failed restore.
    fn rewindToMark(self: *Completer, m: Mark) !void {
        self.mark = null;
        var bits = m.history;
        defer if (bits) |*b| b.deinit(self.alloc);
        inference.engine.rewindSession(self.eng, m.position, m.last) catch |err| {
            self.eng.model.reset();
            if (self.history) |h| h.reset();
            self.seen.clearRetainingCapacity();
            self.primed_len = 0;
            return err;
        };
        if (self.history) |h| {
            h.reset();
            if (bits) |b| {
                h.seen.setUnion(b);
                h.revision += 1;
            }
        }
        self.seen.shrinkRetainingCapacity(m.bytes);
    }

    fn dropMark(self: *Completer) void {
        if (self.mark) |*m| if (m.history) |*b| b.deinit(self.alloc);
        self.mark = null;
    }

    /// Marks where this step's prompt ends (`Mark`) when the model can move
    /// back by position. `tokens` encodes `full` from `seen`, at session
    /// position `reused`.
    fn markPromptEnd(self: *Completer, full: []const u8, tokens: []const u32, reused: usize) !void {
        const profile = self.eng.profile orelse return;
        if (self.eng.model.hasRecurrentState()) return;
        const start = self.seen.items.len;
        const end = profile.promptEnd(full);
        if (end <= start) return;
        const fed = if (end == full.len) tokens.len else blk: {
            const head = try self.eng.encode(full[start..end]);
            defer self.alloc.free(head);
            // A cut before a control token encodes alike; anything else is
            // not a mark.
            if (head.len >= tokens.len or !std.mem.eql(u32, head, tokens[0..head.len])) return;
            break :blk head.len;
        };
        var bits: ?std.bit_set.Dynamic = if (self.history) |h| try h.seen.clone(self.alloc) else null;
        if (bits) |*b| for (tokens[0..fed]) |token| if (token < b.bit_length) b.set(token);
        self.mark = .{ .bytes = end, .position = reused + fed, .last = tokens[fed - 1], .history = bits };
    }

    /// A state kept inside a step's prompt: the system block's end, the
    /// boundary every conversation under the same system prompt shares
    /// (the agent primes it; a server request does not). Fed to `complete`
    /// as `inference.engine.Saves`.
    const Saving = struct {
        completer: *Completer,
        full: []const u8,
        offset: usize = 0,
        bytes: usize = 0,

        /// Picks the system block's end when it lies inside the remainder,
        /// the memory tier would keep it and does not hold it, and the
        /// state holds `min_save_tokens`.
        fn plan(self: *Saving, messages: []const Profile.Message, definitions: []const Profile.ToolDefinition, tokens: []const u32, reused: usize) !void {
            const c = self.completer;
            if (c.cache.budget == 0) return;
            if (c.observer) |o| if (o.layer != null) return;
            const start = c.seen.items.len;
            const system = try c.eng.prefix(messages[0..Profile.leadingSystemCount(messages)], definitions, c.effort);
            defer c.alloc.free(system);
            if (system.len <= start or system.len >= self.full.len or !std.mem.startsWith(u8, self.full, system)) return;
            if (c.cache.exact(system) != null) return;
            const head = try c.eng.encode(self.full[start..system.len]);
            defer c.alloc.free(head);
            if (head.len >= tokens.len or !std.mem.eql(u32, head, tokens[0..head.len])) return;
            if (reused + head.len < min_save_tokens) return;
            self.offset = head.len;
            self.bytes = system.len;
        }

        fn saves(self: *Saving) ?inference.engine.Saves {
            if (self.offset == 0) return null;
            return .{ .offsets = (&self.offset)[0..1], .context = self, .save = save };
        }

        fn save(context: *anyopaque, _: usize) void {
            const self: *Saving = @ptrCast(@alignCast(context));
            const c = self.completer;
            var snap = c.eng.model.snapshot(c.alloc) catch return;
            if (snap.span_count != 0) return snap.deinit();
            c.keepAs(snap, self.full[0..self.bytes]);
        }
    };

    fn run(context: *anyopaque, messages: []const Profile.Message, definitions: []const Profile.ToolDefinition, images: []const Image, sink: *Sink) anyerror!Reply {
        const self: *Completer = @ptrCast(@alignCast(context));
        var replay: ?Replay = null;
        const full = try self.eng.render(messages, definitions, self.effort);
        defer self.alloc.free(full);
        const remainder = blk: {
            var continued: ?[]const u8 = if (self.seen.items.len > 0) increment(self.seen.items, full) else null;
            // A request sent again, or a step not followed by `checkpoint`
            // (an answer cut at its budget), finds the session past its mark.
            if (continued == null and self.cause == null) if (self.mark) |m| {
                if (m.bytes < full.len and std.mem.startsWith(u8, full, self.seen.items[0..m.bytes])) {
                    if (self.rewindToMark(m)) {
                        continued = full[self.seen.items.len..];
                    } else |_| {}
                }
            };
            // Only a conversation that was in the session counts as a replay.
            const cause: ?Replay.Cause = self.cause orelse if (continued != null or self.seen.items.len == 0)
                null
            else if (self.seen_effort != self.effort) .effort else .rewrite;
            // A resumed conversation's last turn, from disk, unless memory
            // holds as much.
            if (self.hint) |b| {
                self.hint = null;
                const held = if (self.cache.longest(full)) |e| e.text.len else 0;
                if (b.bytes > held and b.bytes > self.seen.items.len) if (self.restoreBoundary(b, full)) |restored| {
                    if (cause) |c| replay = .{ .cause = c, .restored = restored };
                    break :blk full[self.seen.items.len..];
                };
            }
            // A cached state further along than the session wins even when
            // the session could continue: a re-prime under a conversation
            // left it at the system block.
            if (self.cache.longest(full)) |entry| if (continued == null or entry.text.len > self.seen.items.len) {
                const restored = entry.tokens;
                if (self.restoreEntry(entry)) {
                    if (cause) |c| replay = .{ .cause = c, .restored = restored };
                    break :blk full[self.seen.items.len..];
                } else |_| {
                    self.cache.remove(entry);
                    // The failed restore reset the session.
                    continued = null;
                }
            };
            if (continued) |rest| {
                if (cause) |c| replay = .{ .cause = c };
                break :blk rest;
            }
            // The flag describes the live session, which now starts empty.
            self.images_fed = false;
            self.eng.model.reset();
            if (self.history) |h| h.reset();
            self.seen.clearRetainingCapacity();
            if (cause) |c| replay = .{ .cause = c };
            break :blk full;
        };
        self.cause = null;
        self.primed_len = @min(self.primed_len, self.seen.items.len);
        const tokens = try self.eng.encode(remainder);
        defer self.alloc.free(tokens);
        const session = self.eng.model.session();
        const reused = session.position;
        const limit = outputBudget(session.position, tokens.len, self.buffers.generated.len, session.capacity, self.clamp_budget) orelse {
            const output: usize = if (self.clamp_budget) 1 else self.buffers.generated.len;
            self.overflow = .{ .needed = session.position + tokens.len + output, .capacity = session.capacity };
            return error.ContextFull;
        };
        // The remainder's placeholder runs are the last images rendered:
        // consumed text is a prefix of the render and compaction drops whole
        // earlier turns, so the images still rendered are a suffix.
        var prefill = try self.imagePrefill(tokens, images);
        defer prefill.deinit(self.alloc);
        if (prefill.value != null) self.images_fed = true;
        self.dropMark();
        var saving: Saving = .{ .completer = self, .full = full };
        if (!self.images_fed) {
            saving.plan(messages, definitions, tokens, reused) catch {};
            // Allocation is the only failure: no mark, a replay next turn.
            self.markPromptEnd(full, tokens, reused) catch {};
        }
        errdefer self.dropMark();
        // The effort is the one this render used (Ctrl-T changes it between
        // turns): it decides whether the completion opens in reasoning.
        var buffers = self.buffers;
        buffers.effort = self.effort;
        buffers.thinking_budget = config.thinkingBudget(self.thinking_budget, self.effort, limit);
        buffers.saves = saving.saves();
        const settings: inference.engine.Speculative = if (self.images_fed) .{ .enabled = false, .draft_length = self.speculative.draft_length } else self.speculative;
        const outcome = try inference.engine.complete(
            self.eng,
            tokens,
            limit,
            self.sampler,
            self.history,
            settings,
            buffers,
            prefill.value,
            self.observer,
            sink,
        );
        // The model consumed the prompt and every generated token except the
        // last one sampled (the stop token or the budget's final token).
        try self.seen.appendSlice(self.alloc, remainder);
        self.seen_effort = self.effort;
        const timing = outcome.timing;
        const fed = if (timing.generated_tokens > 0) self.buffers.generated[0 .. timing.generated_tokens - 1] else self.buffers.generated[0..0];
        const fed_text = try inference.bpe.decode(self.alloc, &self.eng.vocab, fed, true, .{});
        defer self.alloc.free(fed_text);
        try self.seen.appendSlice(self.alloc, fed_text);
        if (outcome.stop == .cancelled) self.reset(.cancel);
        return .{ .outcome = outcome, .replay = replay, .prompt_tokens = reused + tokens.len, .reused_tokens = reused };
    }

    const Prefill = struct {
        value: ?inference.engine.ImagePrefill = null,
        spans: []inference.vision.Span = &.{},
        features: []f32 = &.{},
        fn deinit(self: *Prefill, alloc: Allocator) void {
            if (self.spans.len != 0) alloc.free(self.spans);
            if (self.features.len != 0) alloc.free(self.features);
        }
    };

    /// The spans and concatenated features for the placeholder runs in
    /// `tokens`, paired with the tail of `images`; none when there are no
    /// runs. `ImageSpanMismatch` when the runs outnumber the images or a run
    /// is not an image's token count.
    fn imagePrefill(self: *Completer, tokens: []const u32, images: []const Image) !Prefill {
        const pad = self.eng.imagePadId() orelse return .{};
        const runs = placeholderRuns(tokens, pad);
        if (runs == 0) return .{};
        if (runs > images.len) return error.ImageSpanMismatch;
        const tail = images[images.len - runs ..];
        const prepared = try self.alloc.alloc(inference.engine.PreparedImage, tail.len);
        defer self.alloc.free(prepared);
        var total: usize = 0;
        for (tail, prepared) |image, *p| {
            p.* = image.prepared;
            total += image.prepared.features.len;
        }
        const spans = try self.eng.locateImageSpans(tokens, prepared);
        errdefer self.alloc.free(spans);
        const features = try self.alloc.alloc(f32, total);
        var at: usize = 0;
        for (tail) |image| {
            @memcpy(features[at..][0..image.prepared.features.len], image.prepared.features);
            at += image.prepared.features.len;
        }
        return .{ .value = .{ .spans = spans, .features = features }, .spans = spans, .features = features };
    }
};

/// The tokens a completion may generate after a prompt of `prompt` tokens
/// fed at `position` in a window of `capacity`, from an output buffer of
/// `slice`: the whole slice when it fits; with `clamp`, what the window has
/// left, at least 1; null when it does not fit.
pub fn outputBudget(position: usize, prompt: usize, slice: usize, capacity: usize, clamp: bool) ?usize {
    if (position + prompt > capacity) return null;
    const left = capacity - position - prompt;
    if (slice <= left) return slice;
    if (!clamp or left == 0) return null;
    return left;
}

/// How many maximal runs of `pad` the tokens hold: one per image span.
pub fn placeholderRuns(tokens: []const u32, pad: u32) usize {
    var runs: usize = 0;
    var inside = false;
    for (tokens) |token| {
        const is_pad = token == pad;
        if (is_pad and !inside) runs += 1;
        inside = is_pad;
    }
    return runs;
}

// ----- tests -----

const testing = std.testing;

test "increment is the unseen suffix, or null when the conversation must replay" {
    try testing.expectEqualStrings("<|im_end|>\n<|im_start|>user\nmore", increment("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nanswer", "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nanswer<|im_end|>\n<|im_start|>user\nmore").?);
    try testing.expectEqualStrings("whole", increment("", "whole").?);
    try testing.expect(increment("system A", "system B") == null);
}

test "placeholder runs count image spans, not tokens" {
    try testing.expectEqual(@as(usize, 0), placeholderRuns(&.{ 1, 2, 3 }, 9));
    try testing.expectEqual(@as(usize, 1), placeholderRuns(&.{ 1, 9, 9, 9, 2 }, 9));
    try testing.expectEqual(@as(usize, 2), placeholderRuns(&.{ 9, 9, 1, 9 }, 9));
    try testing.expectEqual(@as(usize, 1), placeholderRuns(&.{9}, 9));
}

test "the output budget is the slice when it fits, what is left when clamped, else nothing" {
    try testing.expectEqual(@as(?usize, 100), outputBudget(0, 900, 100, 1000, false));
    try testing.expectEqual(@as(?usize, null), outputBudget(0, 901, 100, 1000, false));
    try testing.expectEqual(@as(?usize, 99), outputBudget(0, 901, 100, 1000, true));
    try testing.expectEqual(@as(?usize, 1), outputBudget(500, 499, 16384, 1000, true));
    try testing.expectEqual(@as(?usize, null), outputBudget(500, 500, 16384, 1000, true));
    try testing.expectEqual(@as(?usize, null), outputBudget(500, 501, 1, 1000, true));
}
